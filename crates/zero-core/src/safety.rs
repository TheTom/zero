//! Destructive-command guard.
//!
//! The hard lesson from every other harness is that *soft* rules in a prompt or
//! a `CLAUDE.md` do not stop an agent (or a fat-fingered human) from running
//! `rm -rf ~` — only a hard technical constraint at the execution boundary does.
//! So this is a pure, dependency-free classifier that every command must pass
//! through before it runs: the `!` shell mode today, and agent tool calls later.
//!
//! It is deliberately **conservative** — it would rather flag a harmless command
//! and ask for confirmation than let a catastrophic one through silently. The
//! classifier never executes anything; it only labels.

/// How risky a command looks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Risk {
    /// No known destructive pattern — safe to run.
    Safe,
    /// Matches a destructive pattern — require explicit confirmation first.
    Dangerous,
}

/// A classification plus a short human reason (for the confirmation prompt).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Verdict {
    pub risk: Risk,
    /// Why it was flagged; `None` when `Safe`.
    pub reason: Option<&'static str>,
}

impl Verdict {
    pub fn is_dangerous(&self) -> bool {
        self.risk == Risk::Dangerous
    }
}

/// Classify a shell command line. Splits on `;`, `|`, `&`, and newlines and
/// flags the command if *any* sub-command matches a destructive pattern.
pub fn classify(cmd: &str) -> Verdict {
    for seg in split_segments(cmd) {
        if let Some(reason) = segment_danger(seg) {
            return Verdict {
                risk: Risk::Dangerous,
                reason: Some(reason),
            };
        }
    }
    Verdict {
        risk: Risk::Safe,
        reason: None,
    }
}

/// Convenience: just the boolean.
pub fn is_dangerous(cmd: &str) -> bool {
    classify(cmd).is_dangerous()
}

/// Break a command line into independently-classified segments.
fn split_segments(cmd: &str) -> impl Iterator<Item = &str> {
    cmd.split([';', '|', '&', '\n'])
        .filter(|s| !s.trim().is_empty())
}

/// Inspect one segment; return a reason string if it is destructive.
fn segment_danger(seg: &str) -> Option<&'static str> {
    let s = seg.trim();
    let tokens: Vec<&str> = s.split_whitespace().collect();
    // argv[0] is the first token that is NOT a leading `NAME=val` environment
    // assignment — `FOO=bar rm -rf /` runs `rm`, not a command named `FOO=bar`.
    // (The rules matcher already skips these; safety must agree or it is bypassed.)
    let first = tokens
        .iter()
        .find(|t| !is_env_assign(t))
        .copied()
        .unwrap_or("");
    let has = |t: &str| tokens.contains(&t);
    let any_risky_path = || tokens.iter().any(|t| is_risky_path(t));

    // Fork bomb.
    if s.contains(":(){") || s.contains(":|:&") {
        return Some("looks like a fork bomb");
    }
    // Elevated privileges — always confirm.
    if has("sudo") || has("doas") {
        return Some("runs with elevated privileges (sudo)");
    }
    // Power state.
    if matches!(first, "shutdown" | "reboot" | "halt" | "poweroff") {
        return Some("powers off or reboots the machine");
    }
    // Raw disk writes.
    if first == "dd" && tokens.iter().any(|t| t.starts_with("of=")) {
        return Some("dd can overwrite a whole disk");
    }
    if tokens.iter().any(|t| t.starts_with("mkfs")) {
        return Some("formats a filesystem");
    }
    if ["/dev/sd", "/dev/disk", "/dev/nvme", "/dev/hd"]
        .iter()
        .any(|dev| {
            s.contains(&format!("> {dev}"))
                || s.contains(&format!(">{dev}"))
                || s.contains(&format!("of={dev}"))
        })
    {
        return Some("writes directly to a disk device");
    }
    // Clobbering a system path.
    if ["/etc", "/usr", "/bin", "/boot", "/sys", "/lib"]
        .iter()
        .any(|p| s.contains(&format!("> {p}")) || s.contains(&format!(">{p}")))
    {
        return Some("overwrites a system path");
    }
    // rm.
    if first == "rm" {
        let recursive = tokens.iter().any(|t| is_rm_recursive_flag(t));
        let risky = any_risky_path();
        return match (recursive, risky) {
            (true, true) => Some("recursively deletes a critical path"),
            (true, false) => Some("recursively deletes files"),
            (false, true) => Some("deletes a critical path"),
            (false, false) => None,
        };
    }
    // git, the other classic foot-gun.
    if first == "git" {
        if has("reset") && has("--hard") {
            return Some("git reset --hard discards uncommitted changes");
        }
        if has("clean") && tokens.iter().any(|t| t.starts_with('-') && t.contains('f')) {
            return Some("git clean deletes untracked files");
        }
        if has("checkout") && (has("--") || has(".") || has("-f") || has("--force")) {
            return Some("git checkout discards uncommitted changes");
        }
        if has("push") && (has("--force") || has("-f") || tokens.iter().any(|t| t.starts_with('+')))
        {
            return Some("force push rewrites remote history");
        }
    }
    // Recursive permission/ownership changes on a critical path.
    if matches!(first, "chmod" | "chown")
        && tokens.iter().any(|t| *t == "-R" || *t == "--recursive")
        && any_risky_path()
    {
        return Some("recursively changes permissions on a critical path");
    }
    // find that deletes.
    if first == "find" && (has("-delete") || (has("-exec") && has("rm"))) {
        return Some("find deletes matched files");
    }
    // Moving into the void.
    if first == "mv" && has("/dev/null") {
        return Some("moves files into /dev/null (destroys them)");
    }
    None
}

/// Is `tok` a leading `NAME=value` shell environment assignment (so it precedes
/// argv[0] rather than being the command)? Name must be a valid shell identifier.
fn is_env_assign(tok: &str) -> bool {
    let Some(eq) = tok.find('=') else {
        return false;
    };
    let name = &tok[..eq];
    !name.is_empty()
        && !name.as_bytes()[0].is_ascii_digit()
        && name.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_')
}

/// Does this `rm` flag request recursion (`-r`, `-R`, `-rf`, `--recursive`)?
fn is_rm_recursive_flag(tok: &str) -> bool {
    if tok == "--recursive" {
        return true;
    }
    if tok.starts_with("--") || !tok.starts_with('-') {
        return false;
    }
    tok[1..].contains(['r', 'R'])
}

/// Paths that are catastrophic to delete or recurse over.
fn is_risky_path(tok: &str) -> bool {
    matches!(
        tok,
        "/" | "/*" | "~" | "~/" | "*" | "." | "./" | "$HOME" | "${HOME}" | "~/*" | "$HOME/*"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn danger(cmd: &str) -> bool {
        is_dangerous(cmd)
    }

    #[test]
    fn ordinary_commands_are_safe() {
        for cmd in [
            "ls -la",
            "echo hello",
            "cargo test --workspace",
            "git status",
            "git commit -m 'wip'",
            "grep -r foo src",
            "rm file.txt",
            "rm -f stale.log",
            "cat /etc/hosts",
            "mv a.txt b.txt",
            "find . -name '*.rs'",
        ] {
            assert!(!danger(cmd), "should be safe: {cmd}");
        }
    }

    #[test]
    fn recursive_rm_is_flagged() {
        assert!(danger("rm -rf node_modules"));
        assert!(danger("rm -r build"));
        assert!(danger("rm -R something"));
        assert!(danger("rm --recursive build"));
    }

    #[test]
    fn rm_of_critical_paths_is_flagged() {
        assert!(danger("rm -rf /"));
        assert!(danger("rm -rf ~"));
        assert!(danger("rm -rf ~/"));
        assert!(danger("rm -rf *"));
        assert!(danger("rm -rf $HOME"));
        assert!(danger("rm -fr ."));
        assert!(danger("rm /*"));
    }

    #[test]
    fn sudo_and_power_are_flagged() {
        assert!(danger("sudo rm file"));
        assert!(danger("sudo apt-get install foo"));
        assert!(danger("shutdown now"));
        assert!(danger("reboot"));
    }

    #[test]
    fn disk_and_fs_destroyers_are_flagged() {
        assert!(danger("dd if=/dev/zero of=/dev/sda"));
        assert!(danger("mkfs.ext4 /dev/sdb1"));
        assert!(danger("echo x > /dev/sda"));
        assert!(danger("cat img > /dev/disk2"));
    }

    #[test]
    fn fork_bomb_is_flagged() {
        assert!(danger(":(){ :|:& };:"));
    }

    #[test]
    fn destructive_git_is_flagged() {
        assert!(danger("git reset --hard HEAD~3"));
        assert!(danger("git clean -fd"));
        assert!(danger("git checkout -- ."));
        assert!(danger("git checkout ."));
        assert!(danger("git push --force origin main"));
        assert!(danger("git push -f"));
    }

    #[test]
    fn non_destructive_git_stays_safe() {
        assert!(!danger("git checkout -b feature"));
        assert!(!danger("git push origin main"));
        assert!(!danger("git reset HEAD~1")); // soft reset keeps changes
    }

    #[test]
    fn recursive_chmod_on_root_is_flagged() {
        assert!(danger("chmod -R 777 /"));
        assert!(danger("chown -R me ~"));
        assert!(!danger("chmod -R 755 ./scripts")); // scoped path is fine
    }

    #[test]
    fn find_delete_and_overwrite_system_are_flagged() {
        assert!(danger("find . -name '*.tmp' -delete"));
        assert!(danger("find / -exec rm {} ;"));
        assert!(danger("echo '' > /etc/passwd"));
        assert!(danger("mv secrets /dev/null"));
    }

    #[test]
    fn env_assignment_prefix_does_not_hide_the_command() {
        // argv[0] detection must skip leading `NAME=val` env assignments, or a
        // dangerous command smuggles past the classifier behind a benign-looking
        // variable. (Regression: these all returned safe before the env-skip.)
        assert!(danger("FOO=bar rm -rf /"));
        assert!(danger("A=1 B=2 rm -rf ~"));
        assert!(danger("EDITOR=vi shutdown now"));
        assert!(danger("TMPDIR=/tmp dd if=/dev/zero of=/dev/sda"));
        // A bare `NAME=val` with no command is not itself dangerous.
        assert!(!danger("FOO=bar"));
        assert!(!danger("PATH=/usr/bin ls"));
        // `=` inside a real argument must not be mistaken for an env assignment.
        assert!(danger("rm -rf / --opt=val"));
    }

    #[test]
    fn danger_anywhere_in_a_chain_is_caught() {
        assert!(danger("cargo build && rm -rf /"));
        assert!(danger("ls | sudo tee /etc/hosts"));
        assert!(danger("echo ok; git reset --hard"));
    }

    #[test]
    fn classify_reports_a_reason() {
        let v = classify("rm -rf /");
        assert!(v.is_dangerous());
        assert!(v.reason.unwrap().contains("critical"));
        let safe = classify("ls");
        assert_eq!(safe.risk, Risk::Safe);
        assert!(safe.reason.is_none());
    }

    #[test]
    fn empty_and_whitespace_are_safe() {
        assert!(!danger(""));
        assert!(!danger("   "));
        assert!(!danger(";;;"));
    }
}
