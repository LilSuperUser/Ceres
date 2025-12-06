# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **BREAKING**: Refactored Gemini error handling to use structured types instead of strings
  - Replaced `AppError::GeminiError(String)` with `AppError::GeminiError(GeminiErrorDetails)`
  - Added `GeminiErrorKind` enum for type-safe error classification (Authentication, RateLimit, QuotaExceeded, ServerError, NetworkError, Unknown)
  - Added `GeminiErrorDetails` struct with error kind, message, and HTTP status code
  - Replaced string parsing in `user_message()` with pattern matching for better performance and maintainability
  - Updated `is_retryable()` to intelligently handle different Gemini error types
  - Added `classify_gemini_error()` helper function for centralized error classification

### Added
- Structured error handling for Gemini API calls with detailed error information
- HTTP status codes are now captured and exposed in Gemini errors
- Better error messages for different types of Gemini API failures
- 7 new unit tests for Gemini error classification
- 5 new unit tests for structured error handling

## [0.0.1] - Initial Release

### Added
- Semantic search engine for CKAN open data portals
- Integration with Google Gemini API for text embeddings
- PostgreSQL database with pgvector extension for vector similarity search
- CLI commands: harvest, search, export, stats
- Support for multiple CKAN portals
- Concurrent dataset processing
- CSV and JSONL export formats