//! Single source of truth for the product name.
//!
//! The name "Zero" is **not final**. Everything user-facing — the banner, the
//! assistant prompt label, the config/session directory — derives from the two
//! constants here, so renaming the product is a one-file edit (plus renaming the
//! crates, which is mechanical; see README "Renaming the project").
//!
//! Override at runtime without recompiling via the `ZERO_NAME` / `ZERO_SLUG`
//! env vars — handy while the name is still in flux.

/// Display name, e.g. shown in the banner. Title-cased.
pub const DEFAULT_NAME: &str = "Zero";

/// Lowercase slug used for the binary, the `.<slug>` config dir, and the
/// assistant prompt label.
pub const DEFAULT_SLUG: &str = "zero";

/// Resolved display name, honoring the `ZERO_NAME` override.
pub fn name() -> String {
    std::env::var("ZERO_NAME").unwrap_or_else(|_| DEFAULT_NAME.to_string())
}

/// Resolved lowercase slug, honoring the `ZERO_SLUG` override.
pub fn slug() -> String {
    std::env::var("ZERO_SLUG").unwrap_or_else(|_| DEFAULT_SLUG.to_string())
}

/// Name of the per-user config/session directory, e.g. `.zero`.
pub fn dot_dir() -> String {
    format!(".{}", slug())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_consistent() {
        // Slug is the lowercase of the name by convention.
        assert_eq!(DEFAULT_SLUG, DEFAULT_NAME.to_lowercase());
    }

    #[test]
    fn dot_dir_is_hidden_slug() {
        // Can't assume env is unset in CI, so just check the shape.
        assert!(dot_dir().starts_with('.'));
        assert!(dot_dir().ends_with(&slug()));
    }
}
