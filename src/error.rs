use thiserror::Error;

#[derive(Error, Debug)]
pub enum AppError {
    #[error("Database error: {0}")]
    DatabaseError(#[from] sqlx::Error),

    #[error("API Client error: {0}")]
    ClientError(#[from] reqwest::Error),

    #[error("OpenAI error: {0}")]
    OpenAiError(#[from] async_openai::error::OpenAIError),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    #[error("Generic error: {0}")]
    Generic(String),
}