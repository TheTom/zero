//! The imperative shell around the pure input/output models: raw-mode terminal
//! control via libc, with zero crates. We declare the handful of `termios` /
//! `ioctl` symbols ourselves rather than pulling in the `libc` crate — they are
//! stable C ABI and this keeps Zero's dependency count at zero.
//!
//! Only this module is `unsafe`; the rest of the TUI is pure and tested. Raw
//! mode is RAII: [`RawTerminal`] restores the original tty settings on drop, so
//! a panic or early return never leaves the user's shell wedged.

use std::io;
use std::os::raw::{c_int, c_ulong, c_void};
use std::os::unix::io::RawFd;

const STDIN_FD: RawFd = 0;
/// `tcsetattr` action: flush pending I/O, then apply (same value on macOS+Linux).
const TCSAFLUSH: c_int = 2;

// --- termios struct, per-platform layout -----------------------------------

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone)]
struct Termios {
    c_iflag: c_ulong,
    c_oflag: c_ulong,
    c_cflag: c_ulong,
    c_lflag: c_ulong,
    c_cc: [u8; 20], // NCCS == 20 on Darwin
    c_ispeed: c_ulong,
    c_ospeed: c_ulong,
}

#[cfg(target_os = "macos")]
const VMIN: usize = 16;
#[cfg(target_os = "macos")]
const VTIME: usize = 17;

#[cfg(target_os = "linux")]
#[repr(C)]
#[derive(Clone)]
struct Termios {
    c_iflag: u32,
    c_oflag: u32,
    c_cflag: u32,
    c_lflag: u32,
    c_line: u8,
    c_cc: [u8; 32], // NCCS == 32 on Linux
    c_ispeed: u32,
    c_ospeed: u32,
}

#[cfg(target_os = "linux")]
const VMIN: usize = 6;
#[cfg(target_os = "linux")]
const VTIME: usize = 5;

#[cfg(target_os = "macos")]
const TIOCGWINSZ: c_ulong = 0x4008_7468;
#[cfg(target_os = "linux")]
const TIOCGWINSZ: c_ulong = 0x5413;

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct WinSize {
    ws_row: u16,
    ws_col: u16,
    ws_xpixel: u16,
    ws_ypixel: u16,
}

extern "C" {
    fn tcgetattr(fd: c_int, termios: *mut Termios) -> c_int;
    fn tcsetattr(fd: c_int, optional_actions: c_int, termios: *const Termios) -> c_int;
    fn cfmakeraw(termios: *mut Termios);
    fn ioctl(fd: c_int, request: c_ulong, argp: *mut WinSize) -> c_int;
    fn read(fd: c_int, buf: *mut c_void, count: usize) -> isize;
    fn isatty(fd: c_int) -> c_int;
}

/// Terminal dimensions in character cells.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Size {
    pub cols: u16,
    pub rows: u16,
}

/// Is stdin connected to a real terminal? False under pipes / CI.
pub fn stdin_is_tty() -> bool {
    // SAFETY: isatty takes an fd and has no preconditions.
    unsafe { isatty(STDIN_FD) == 1 }
}

/// Query the current terminal size, falling back to 80x24 if the ioctl fails.
pub fn terminal_size() -> Size {
    let mut ws = WinSize::default();
    // SAFETY: ws is a valid, sized-out WinSize for the kernel to fill.
    let rc = unsafe { ioctl(STDIN_FD, TIOCGWINSZ, &mut ws) };
    if rc == 0 && ws.ws_col > 0 && ws.ws_row > 0 {
        Size {
            cols: ws.ws_col,
            rows: ws.ws_row,
        }
    } else {
        Size { cols: 80, rows: 24 }
    }
}

/// RAII guard that puts the terminal in raw mode and restores it on drop.
pub struct RawTerminal {
    original: Termios,
}

impl RawTerminal {
    /// Enter raw mode. Errors if stdin is not a tty or `termios` calls fail.
    pub fn enable() -> io::Result<Self> {
        if !stdin_is_tty() {
            return Err(io::Error::other("stdin is not a terminal"));
        }
        // SAFETY: zeroed Termios is a valid out-param; tcgetattr fills it.
        let mut original: Termios = unsafe { std::mem::zeroed() };
        if unsafe { tcgetattr(STDIN_FD, &mut original) } != 0 {
            return Err(io::Error::last_os_error());
        }
        let mut raw = original.clone();
        // SAFETY: raw is a valid Termios; cfmakeraw only writes to it.
        unsafe { cfmakeraw(&mut raw) };
        // VMIN=0, VTIME=1 → read returns after up to 100ms even with no input,
        // which lets the event loop poll for resize and disambiguate a lone ESC.
        raw.c_cc[VMIN] = 0;
        raw.c_cc[VTIME] = 1;
        if unsafe { tcsetattr(STDIN_FD, TCSAFLUSH, &raw) } != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(RawTerminal { original })
    }

    /// Read available input bytes into `buf`. Returns 0 on a poll timeout (no
    /// input within VTIME) — not EOF. Callers loop and re-render.
    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        // SAFETY: buf is a valid, writable slice of len buf.len().
        let n = unsafe { read(STDIN_FD, buf.as_mut_ptr() as *mut c_void, buf.len()) };
        if n < 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(n as usize)
        }
    }
}

impl crate::app::Input for RawTerminal {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        RawTerminal::read(self, buf)
    }
}

impl Drop for RawTerminal {
    fn drop(&mut self) {
        // Best-effort restore; nothing useful to do if it fails during unwind.
        // SAFETY: self.original is the saved, valid pre-raw termios.
        unsafe {
            tcsetattr(STDIN_FD, TCSAFLUSH, &self.original);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminal_size_is_never_zero() {
        // Whether or not a tty is attached, we must return a usable size.
        let s = terminal_size();
        assert!(s.cols > 0 && s.rows > 0);
    }

    #[test]
    fn enabling_raw_mode_without_tty_errors_cleanly() {
        // Under `cargo test` stdin is usually not a tty; enable() must Err, not
        // panic or corrupt the terminal. (If a tty *is* attached, skip.)
        if !stdin_is_tty() {
            assert!(RawTerminal::enable().is_err());
        }
    }
}
