// src/lib.rs
pub mod clients;
pub mod config;
pub mod error;
pub mod models;
pub mod storage;

// Re-export commonly used items for easier access
pub use error::AppError;
