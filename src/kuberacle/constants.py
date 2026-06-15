"""Cross-cutting constants shared across the pipeline."""

#: Sentinel that marks an abstention. An answer that begins with this token means
#: the system could not answer from the docs corpus. It is the single source of
#: truth for the token the generator falls back to, the QA abstention messages,
#: and the eval metric that detects abstention. The prompt files instruct the
#: model to emit the same token and must be kept in sync by hand.
ABSTENTION_SENTINEL = "INSUFFICIENT_EVIDENCE"
