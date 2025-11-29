--- Enables the vector extension and creates the datasets table ---
CREATE EXTENSION IF NOT EXISTS vector;

--- Main table to store datasets with embeddings ---
CREATE TABLE IF NOT EXISTS datasets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Core Metadata
    original_id VARCHAR NOT NULL,      -- Original ID on the CKAN portal
    source_portal VARCHAR NOT NULL,    -- e.g. "dati.gov.it"
    url VARCHAR NOT NULL,              -- URL of the dataset page
    
    -- Content
    title TEXT NOT NULL,
    description TEXT,
    
    -- Embedding (1536 dimensions is standard for OpenAI's text-embedding-3-small)
    embedding vector(1536),
    
    -- Technical metadata (JSONB format for future flexibility)
    metadata JSONB DEFAULT '{}'::jsonb,
    
    -- Audit
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Uniqueness constraint: same ID on the same portal = same record
    CONSTRAINT uk_portal_original_id UNIQUE (source_portal, original_id)
);

-- Index for vector search (HNSW is the fastest for approximate queries)
-- Note: requires data in the table to be created effectively, 
-- but we define it here for completeness.
CREATE INDEX ON datasets USING hnsw (embedding vector_cosine_ops);