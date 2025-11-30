use chrono::{DateTime, Utc};
use pgvector::Vector;
use serde::{Deserialize, Serialize};
use sqlx::prelude::FromRow;
use sqlx::types::Json;
use uuid::Uuid;

/// Rappresentazione completa di una riga della tabella 'datasets'
#[derive(Debug, FromRow, Serialize, Deserialize)]
pub struct Dataset {
    pub id: Uuid,
    pub original_id: String,
    pub source_portal: String,
    pub url: String,
    pub title: String,
    pub description: Option<String>,

    // Mappatura automatica con la crate 'pgvector'
    pub embedding: Option<Vector>,

    // Wrapper Json per gestire il tipo JSONB di Postgres
    pub metadata: Json<serde_json::Value>,

    pub first_seen_at: DateTime<Utc>,
    pub last_updated_at: DateTime<Utc>,
}

/// Struttura usata per inserire/aggiornare dati (DTO)
#[derive(Debug, Serialize, Clone)]
pub struct NewDataset {
    pub original_id: String,
    pub source_portal: String,
    pub url: String,
    pub title: String,
    pub description: Option<String>,
    pub embedding: Option<Vec<f32>>, // Qui usiamo Vec standard per comodità
    pub metadata: serde_json::Value,
}
