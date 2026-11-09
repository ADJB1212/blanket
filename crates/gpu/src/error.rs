//! Typed GPU failures with owned diagnostic chains for detached execution.

use std::error::Error;
use std::fmt;

/// Stable failure categories shared by GPU preparation and execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuErrorKind {
    InvalidInput,
    Allocation,
    Unavailable,
    Execution,
}

/// A GPU failure and a snapshot of its native diagnostic chain.
#[derive(Clone, Debug)]
#[must_use]
pub struct GpuError {
    kind: GpuErrorKind,
    message: String,
    source: Option<Box<Diagnostic>>,
}

#[derive(Clone, Debug)]
struct Diagnostic {
    message: String,
    source: Option<Box<Self>>,
}

impl Diagnostic {
    fn snapshot(error: &dyn Error) -> Self {
        Self { message: error.to_string(), source: error.source().map(|error| Box::new(Self::snapshot(error))) }
    }
}

impl GpuError {
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new(GpuErrorKind::InvalidInput, message)
    }

    pub fn allocation(message: impl Into<String>) -> Self {
        Self::new(GpuErrorKind::Allocation, message)
    }

    pub fn unavailable(message: impl Into<String>) -> Self {
        Self::new(GpuErrorKind::Unavailable, message)
    }

    pub fn execution(message: impl Into<String>) -> Self {
        Self::new(GpuErrorKind::Execution, message)
    }

    fn new(kind: GpuErrorKind, message: impl Into<String>) -> Self {
        Self { kind, message: message.into(), source: None }
    }

    /// Copy diagnostics so foreign objects need not cross execution boundaries.
    pub fn with_source(kind: GpuErrorKind, message: impl Into<String>, source: &dyn Error) -> Self {
        Self { source: Some(Box::new(Diagnostic::snapshot(source))), ..Self::new(kind, message) }
    }

    #[must_use]
    pub fn kind(&self) -> GpuErrorKind {
        self.kind
    }
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for GpuError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.source.as_deref().map(|source| source as _)
    }
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for Diagnostic {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.source.as_deref().map(|source| source as _)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn categories_and_native_sources_survive_cloning() {
        let original = "invalid".parse::<u32>().unwrap_err();
        let nested = GpuError::with_source(GpuErrorKind::Execution, "native execution failed", &original);
        let error = GpuError::with_source(GpuErrorKind::Execution, "GPU operation failed", &nested);
        let cloned = error.clone();
        assert_eq!(cloned.kind(), error.kind());
        assert_eq!(error.kind(), GpuErrorKind::Execution);
        assert_eq!(error.to_string(), "GPU operation failed");
        let source = cloned.source().unwrap();
        assert_eq!(source.to_string(), "native execution failed");
        assert_eq!(source.source().unwrap().to_string(), original.to_string());
        assert!(source.source().unwrap().source().is_none());
    }
}
