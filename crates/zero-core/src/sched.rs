// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2026 Zero Contributors

//! Scheduling time math — the pure half of the loop scheduler: parse an absolute
//! deadline (`parse_rfc3339`) and compute a launch trigger's next fire time
//! (`next_fire` for `every <dur>` / `at <rfc3339>` / `daily HH:MM`). All in `std`
//! on civil-date arithmetic (Howard Hinnant's `days_from_civil`), no chrono.
//!
//! Everything here is **UTC**. `std` exposes no local timezone, so `daily 08:00`
//! means 08:00 UTC — documented, and the only honest std-only choice. The
//! scheduler *thread* (a `BinaryHeap` + `Condvar` in the binary) consumes these
//! fire times; the math is here so it's unit-tested without sleeping.

use crate::loop_config::parse_duration;

const MS_PER_DAY: u64 = 86_400_000;
/// Floor on a recurring `every <dur>` interval — a `0s`/`1ms` spec must not become
/// a wake storm. The shortest a trigger can recur is one second.
pub const MIN_INTERVAL_MS: u64 = 1000;

/// Days from 1970-01-01 to the civil date `y-m-d` (proleptic Gregorian). Negative
/// before the epoch. Hinnant's algorithm — exact, branch-light, no leap tables.
pub fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400; // [0, 399]
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) as i64 + 2) / 5 + d as i64 - 1; // [0,365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146097 + doe - 719468
}

/// Inverse of [`days_from_civil`]: civil `(year, month, day)` from days-since-epoch.
pub fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Parse an RFC3339-ish timestamp to unix **milliseconds**. Accepts
/// `YYYY-MM-DDTHH:MM:SS[.fff][Z|±HH:MM]` (space instead of `T` allowed; missing
/// zone treated as UTC). Returns `None` on anything malformed — never panics.
pub fn parse_rfc3339(s: &str) -> Option<u64> {
    let s = s.trim();
    let (date, rest) = s.split_once(['T', ' '])?;
    let mut dp = date.split('-');
    let y: i64 = dp.next()?.parse().ok()?;
    let mo: u32 = dp.next()?.parse().ok()?;
    let d: u32 = dp.next()?.parse().ok()?;
    if dp.next().is_some() || !(1..=12).contains(&mo) || !(1..=31).contains(&d) {
        return None;
    }

    // Split the zone suffix off the time.
    let (time, offset_secs) = split_zone(rest)?;
    let mut tp = time.split(':');
    let hh: i64 = tp.next()?.parse().ok()?;
    let mm: i64 = tp.next()?.parse().ok()?;
    // Seconds may carry a fractional part (.fff); we keep ms precision.
    let sec_field = tp.next().unwrap_or("0");
    if tp.next().is_some() {
        return None;
    }
    let (ss, frac_ms) = parse_seconds(sec_field)?;
    if !(0..=23).contains(&hh) || !(0..=59).contains(&mm) || !(0..=60).contains(&ss) {
        return None;
    }

    let days = days_from_civil(y, mo, d);
    let secs = days * 86_400 + hh * 3_600 + mm * 60 + ss - offset_secs;
    let ms = secs.checked_mul(1000)?.checked_add(frac_ms as i64)?;
    u64::try_from(ms).ok()
}

/// Split a time-with-zone into `(time, offset_seconds_east_of_utc)`.
fn split_zone(rest: &str) -> Option<(&str, i64)> {
    if let Some(t) = rest.strip_suffix('Z').or_else(|| rest.strip_suffix('z')) {
        return Some((t, 0));
    }
    // Find a +/- that introduces the offset (after the time digits). Search from
    // the end so the date's dashes aren't mistaken for it.
    if let Some(i) = rest.rfind(['+', '-']) {
        // Must be positioned where an offset would be (HH:MM:SS is 5..8 chars in).
        if i >= 5 {
            let (time, off) = rest.split_at(i);
            let sign = if off.starts_with('-') { -1 } else { 1 };
            let body = &off[1..];
            let mut op = body.split(':');
            let oh: i64 = op.next()?.parse().ok()?;
            let om: i64 = op.next().unwrap_or("0").parse().ok()?;
            return Some((time, sign * (oh * 3600 + om * 60)));
        }
    }
    // No zone → UTC.
    Some((rest, 0))
}

/// Parse a seconds field that may carry a fraction: `"07"` → `(7, 0)`,
/// `"07.250"` → `(7, 250)`.
fn parse_seconds(field: &str) -> Option<(i64, u32)> {
    match field.split_once('.') {
        Some((s, frac)) => {
            let secs = s.parse().ok()?;
            // Take up to 3 fractional digits as ms.
            let mut ms = 0u32;
            for (i, c) in frac.chars().take(3).enumerate() {
                let d = c.to_digit(10)?;
                ms += d * 10u32.pow(2 - i as u32);
            }
            Some((secs, ms))
        }
        None => Some((field.parse().ok()?, 0)),
    }
}

/// The next time (unix ms) a launch trigger `when` fires strictly after
/// `after_ms`. Forms:
/// - `every <dur>` — `after_ms + dur` (e.g. `every 6h`).
/// - `at <rfc3339>` — the one-shot instant, or `None` if it's already past.
/// - `daily HH:MM` — the next occurrence of that UTC wall-clock time.
///
/// Returns `None` for an unparseable spec or a one-shot already in the past.
pub fn next_fire(when: &str, after_ms: u64) -> Option<u64> {
    let when = when.trim();
    if let Some(dur) = when.strip_prefix("every ") {
        let d = parse_duration(dur.trim())?;
        // Clamp to the floor so `every 0s` / `every 1ms` can't storm the scheduler.
        let interval = (d.as_millis() as u64).max(MIN_INTERVAL_MS);
        return after_ms.checked_add(interval);
    }
    if let Some(ts) = when.strip_prefix("at ") {
        let fire = parse_rfc3339(ts.trim())?;
        return (fire > after_ms).then_some(fire);
    }
    if let Some(hhmm) = when.strip_prefix("daily ") {
        let (h, m) = hhmm.trim().split_once(':')?;
        let hh: u64 = h.parse().ok()?;
        let mm: u64 = m.parse().ok()?;
        if hh > 23 || mm > 59 {
            return None;
        }
        let target_of_day = (hh * 3600 + mm * 60) * 1000;
        let day = after_ms / MS_PER_DAY;
        let today = day * MS_PER_DAY + target_of_day;
        return Some(if today > after_ms {
            today
        } else {
            today + MS_PER_DAY
        });
    }
    None
}

/// Milliseconds from `now_ms` until `target_ms` (0 if already due).
pub fn ms_until(target_ms: u64, now_ms: u64) -> u64 {
    target_ms.saturating_sub(now_ms)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn civil_day_math_roundtrips_known_dates() {
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(days_from_civil(1970, 1, 2), 1);
        assert_eq!(days_from_civil(1969, 12, 31), -1);
        assert_eq!(days_from_civil(2000, 3, 1), 11017);
        // Round-trip a spread of dates through both directions.
        for &(y, m, d) in &[
            (1970, 1, 1),
            (2026, 6, 13),
            (2000, 2, 29),
            (1999, 12, 31),
            (2100, 1, 1),
        ] {
            let z = days_from_civil(y, m, d);
            assert_eq!(civil_from_days(z), (y, m, d), "roundtrip {y}-{m}-{d}");
        }
    }

    #[test]
    fn parse_rfc3339_utc_and_offsets() {
        // 1970-01-01T00:00:00Z = 0.
        assert_eq!(parse_rfc3339("1970-01-01T00:00:00Z"), Some(0));
        // A known epoch: 2021-01-01T00:00:00Z = 1609459200 s.
        assert_eq!(
            parse_rfc3339("2021-01-01T00:00:00Z"),
            Some(1_609_459_200_000)
        );
        // Offset is applied: 11:00:00-05:00 == 16:00:00Z.
        let east = parse_rfc3339("2026-06-13T16:00:00Z").unwrap();
        let west = parse_rfc3339("2026-06-13T11:00:00-05:00").unwrap();
        assert_eq!(east, west);
        // A positive offset.
        let utc = parse_rfc3339("2026-06-13T10:00:00Z").unwrap();
        let plus2 = parse_rfc3339("2026-06-13T12:00:00+02:00").unwrap();
        assert_eq!(utc, plus2);
    }

    #[test]
    fn parse_rfc3339_space_and_fraction_and_no_zone() {
        // Space separator + fractional seconds + implicit UTC.
        let a = parse_rfc3339("2026-06-13 00:00:00.250").unwrap();
        let b = parse_rfc3339("2026-06-13T00:00:00Z").unwrap();
        assert_eq!(a, b + 250);
    }

    #[test]
    fn parse_rfc3339_rejects_garbage() {
        assert_eq!(parse_rfc3339("not a date"), None);
        assert_eq!(parse_rfc3339("2026-13-01T00:00:00Z"), None); // month 13
        assert_eq!(parse_rfc3339("2026-06-32T00:00:00Z"), None); // day 32
        assert_eq!(parse_rfc3339("2026-06-13T25:00:00Z"), None); // hour 25
        assert_eq!(parse_rfc3339(""), None);
    }

    #[test]
    fn next_fire_every_duration() {
        let base = 1_000_000_000_000;
        assert_eq!(next_fire("every 6h", base), Some(base + 6 * 3600 * 1000));
        assert_eq!(next_fire("every 30m", base), Some(base + 1800 * 1000));
        assert_eq!(next_fire("every nonsense", base), None);
    }

    #[test]
    fn next_fire_every_clamps_to_a_minimum_interval() {
        let base = 1_000_000_000_000;
        // `every 0s` / `every 1ms` must not storm — clamped to the 1s floor.
        assert_eq!(next_fire("every 0s", base), Some(base + MIN_INTERVAL_MS));
        assert_eq!(next_fire("every 0", base), Some(base + MIN_INTERVAL_MS));
    }

    #[test]
    fn next_fire_at_one_shot() {
        let future = parse_rfc3339("2030-01-01T00:00:00Z").unwrap();
        assert_eq!(
            next_fire("at 2030-01-01T00:00:00Z", future - 1000),
            Some(future)
        );
        // Already in the past → None (a one-shot doesn't recur).
        assert_eq!(next_fire("at 2030-01-01T00:00:00Z", future + 1000), None);
    }

    #[test]
    fn next_fire_daily_picks_next_occurrence() {
        // Midnight UTC on a known day.
        let midnight = parse_rfc3339("2026-06-13T00:00:00Z").unwrap();
        // At 07:00 the day's 08:00 is still ahead → today.
        let at7 = midnight + 7 * 3600 * 1000;
        assert_eq!(
            next_fire("daily 08:00", at7),
            Some(midnight + 8 * 3600 * 1000)
        );
        // At 09:00 the 08:00 has passed → tomorrow's 08:00.
        let at9 = midnight + 9 * 3600 * 1000;
        assert_eq!(
            next_fire("daily 08:00", at9),
            Some(midnight + MS_PER_DAY + 8 * 3600 * 1000)
        );
        assert_eq!(next_fire("daily 25:00", midnight), None);
    }

    #[test]
    fn ms_until_saturates() {
        assert_eq!(ms_until(100, 40), 60);
        assert_eq!(ms_until(40, 100), 0); // already due, never negative
    }
}
