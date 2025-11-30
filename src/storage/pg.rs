use crate::error::AppError; // Assumiamo tu abbia creato error.rs come da struttura precedente
use crate::models::{Dataset, NewDataset};
use pgvector::Vector;
use sqlx::{PgPool, Pool, Postgres};
use uuid::Uuid;

#[derive(Clone)]
pub struct DatasetRepository {
    pool: Pool<Postgres>,
}

impl DatasetRepository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Esegue un UPSERT: Inserisce il dataset o lo aggiorna se la coppia
    /// (source_portal, original_id) esiste già.
    pub async fn upsert(&self, new_data: &NewDataset) -> Result<Uuid, AppError> {
        // Convertiamo il Vec<f32> in pgvector::Vector se presente
        let embedding_vector = new_data.embedding.as_ref().map(Vector::from);

        let rec = sqlx::query!(
            r#"
            INSERT INTO datasets (
                original_id, 
                source_portal, 
                url, 
                title, 
                description, 
                embedding, 
                metadata,
                last_updated_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
            ON CONFLICT (source_portal, original_id) 
            DO UPDATE SET
                title = EXCLUDED.title,
                description = EXCLUDED.description,
                url = EXCLUDED.url,
                embedding = EXCLUDED.embedding,
                metadata = EXCLUDED.metadata,
                last_updated_at = NOW()
            RETURNING id
            "#,
            new_data.original_id,
            new_data.source_portal,
            new_data.url,
            new_data.title,
            new_data.description,
            embedding_vector as Option<Vector>, // Casting esplicito per sqlx
            serde_json::to_value(&new_data.metadata).unwrap_or(serde_json::json!({}))
        )
        .fetch_one(&self.pool)
        .await
        .map_err(|e| AppError::DatabaseError(e))?; // Mappa l'errore SQLx nel tuo AppError

        Ok(rec.id)
    }

    /// Recupera un dataset per ID (utile per debug o check)
    pub async fn get(&self, id: Uuid) -> Result<Option<Dataset>, AppError> {
        let result = sqlx::query_as!(
            Dataset,
            r#"
            SELECT 
                id, original_id, source_portal, url, title, description, 
                embedding, metadata, first_seen_at, last_updated_at
            FROM datasets 
            WHERE id = $1
            "#,
            id
        )
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| AppError::DatabaseError(e))?;

        Ok(result)
    }
}
