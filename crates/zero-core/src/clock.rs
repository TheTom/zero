// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2026 Zero Contributors

//! Honest time. Zero's north star on timestamps: **measure, never estimate.**
//! Nothing here predicts how long anything "will take" — it only reports real
//! elapsed durations from a monotonic source, formatted for humans.
//!
//! Wall-clock time (for log filenames / display) comes from `SystemTime`;
//! elapsed measurement uses `Instant` so it is immune to clock adjustments.

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// A running measurement. Start one when a turn begins, read `elapsed()` to get
/// the *actual* time spent — the only kind of duration Zero ever shows.
#[derive(Debug, Clone, Copy)]
pub struct Stopwatch {
    started: Instant,
}

impl Stopwatch {
    pub fn start() -> Self {
        Stopwatch {
            started: Instant::now(),
        }
    }

    pub fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }

    /// Elapsed time, already formatted for display.
    pub fn elapsed_human(&self) -> String {
        format_duration(self.elapsed())
    }
}

/// Seconds since the Unix epoch (UTC), as a wall-clock instant. Used for log
/// timestamps and session filenames — not for measuring intervals.
pub fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Milliseconds since the Unix epoch (UTC).
pub fn unix_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

/// Format a real elapsed duration compactly and honestly.
///
/// Sub-second → `"850ms"`; seconds → `"1.4s"`; minutes/hours roll up
/// (`"2m03s"`, `"1h04m"`). No rounding that inflates: 1499ms is `"1.5s"`.
pub fn format_duration(d: Duration) -> String {
    let total_ms = d.as_millis();
    if total_ms < 1000 {
        return format!("{total_ms}ms");
    }
    let total_secs = d.as_secs();
    if total_secs < 60 {
        // One decimal of seconds, e.g. 1.4s.
        let tenths = (total_ms % 1000) / 100;
        return format!("{total_secs}.{tenths}s");
    }
    if total_secs < 3600 {
        let m = total_secs / 60;
        let s = total_secs % 60;
        return format!("{m}m{s:02}s");
    }
    let h = total_secs / 3600;
    let m = (total_secs % 3600) / 60;
    format!("{h}h{m:02}m")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_sub_second() {
        assert_eq!(format_duration(Duration::from_millis(0)), "0ms");
        assert_eq!(format_duration(Duration::from_millis(850)), "850ms");
        assert_eq!(format_duration(Duration::from_millis(999)), "999ms");
    }

    #[test]
    fn formats_seconds_with_one_decimal() {
        assert_eq!(format_duration(Duration::from_millis(1000)), "1.0s");
        assert_eq!(format_duration(Duration::from_millis(1499)), "1.4s");
        assert_eq!(format_duration(Duration::from_millis(59900)), "59.9s");
    }

    #[test]
    fn formats_minutes() {
        assert_eq!(format_duration(Duration::from_secs(60)), "1m00s");
        assert_eq!(format_duration(Duration::from_secs(123)), "2m03s");
        assert_eq!(format_duration(Duration::from_secs(3599)), "59m59s");
    }

    #[test]
    fn formats_hours() {
        assert_eq!(format_duration(Duration::from_secs(3600)), "1h00m");
        assert_eq!(format_duration(Duration::from_secs(3600 + 4 * 60)), "1h04m");
    }

    #[test]
    fn stopwatch_measures_forward() {
        let sw = Stopwatch::start();
        let e1 = sw.elapsed();
        let e2 = sw.elapsed();
        assert!(e2 >= e1, "elapsed must be monotonic non-decreasing");
    }

    #[test]
    fn stopwatch_elapsed_human_is_formatted() {
        let sw = Stopwatch::start();
        let s = sw.elapsed_human();
        // Freshly started → sub-second, so it ends in "ms".
        assert!(s.ends_with("ms"), "got {s}");
    }

    #[test]
    fn wall_clock_is_after_2020() {
        // 2020-01-01 in unix seconds — sanity that the clock isn't zeroed.
        assert!(unix_seconds() > 1_577_836_800);
        assert!(unix_millis() > 1_577_836_800_000);
    }
}
