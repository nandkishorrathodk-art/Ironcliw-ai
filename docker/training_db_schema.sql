-- =============================================================================
-- JARVIS Training Database Schema
-- =============================================================================
--
-- SQLite schema for the JARVIS training system. This database stores:
-- - Experiences from JARVIS interactions
-- - Scraped web content for training
-- - Learning goals and topics
-- - Training run history
-- - Model deployment records
--
-- Usage:
--   sqlite3 jarvis_training.db < training_db_schema.sql
--
-- Version: 1.0.0
-- =============================================================================

-- Drop existing tables if recreating
-- DROP TABLE IF EXISTS experiences;
-- DROP TABLE IF EXISTS scraped_content;
-- DROP TABLE IF EXISTS learning_goals;
-- DROP TABLE IF EXISTS training_runs;
-- DROP TABLE IF EXISTS model_deployments;
-- DROP TABLE IF EXISTS intelligence_insights;
-- DROP TABLE IF EXISTS agent_tasks;

-- =============================================================================
-- Experiences Table
-- =============================================================================
-- Records of JARVIS interactions that can be used for training
-- Each row represents a user input -> JARVIS output pair

CREATE TABLE IF NOT EXISTS experiences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    source TEXT NOT NULL,                    -- 'voice', 'text', 'api', 'automation'
    input_text TEXT NOT NULL,                -- User's input
    output_text TEXT NOT NULL,               -- JARVIS's response
    context TEXT,                            -- JSON: additional context (screen state, etc.)
    quality_score REAL DEFAULT 0.5,          -- 0.0 to 1.0, based on user feedback
    feedback TEXT,                           -- 'positive', 'negative', 'corrected'
    correction TEXT,                         -- If feedback is 'corrected', the correct response
    used_in_training INTEGER DEFAULT 0,      -- 0 or 1
    training_run_id INTEGER,                 -- Foreign key to training_runs
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for experiences
CREATE INDEX IF NOT EXISTS idx_experiences_timestamp ON experiences(timestamp);
CREATE INDEX IF NOT EXISTS idx_experiences_source ON experiences(source);
CREATE INDEX IF NOT EXISTS idx_experiences_used ON experiences(used_in_training);
CREATE INDEX IF NOT EXISTS idx_experiences_quality ON experiences(quality_score DESC);


-- =============================================================================
-- Scraped Content Table
-- =============================================================================
-- Web documentation and content scraped by Safe Scout

CREATE TABLE IF NOT EXISTS scraped_content (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT UNIQUE NOT NULL,                -- Source URL
    title TEXT,                              -- Page title
    content TEXT NOT NULL,                   -- Processed text content
    content_type TEXT DEFAULT 'documentation', -- 'documentation', 'tutorial', 'api_reference', 'article'
    topic TEXT,                              -- Associated learning topic
    language TEXT DEFAULT 'en',              -- Content language
    quality_score REAL DEFAULT 0.5,          -- 0.0 to 1.0, based on content quality
    word_count INTEGER,                      -- Number of words
    code_blocks INTEGER DEFAULT 0,           -- Number of code blocks
    scraped_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    used_in_training INTEGER DEFAULT 0,
    training_run_id INTEGER
);

-- Indexes for scraped content
CREATE INDEX IF NOT EXISTS idx_scraped_url ON scraped_content(url);
CREATE INDEX IF NOT EXISTS idx_scraped_topic ON scraped_content(topic);
CREATE INDEX IF NOT EXISTS idx_scraped_quality ON scraped_content(quality_score DESC);
CREATE INDEX IF NOT EXISTS idx_scraped_used ON scraped_content(used_in_training);


-- =============================================================================
-- Learning Goals Table
-- =============================================================================
-- Topics JARVIS should learn about, either auto-discovered or user-specified

CREATE TABLE IF NOT EXISTS learning_goals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT UNIQUE NOT NULL,              -- Topic name
    description TEXT,                        -- Optional description
    priority INTEGER DEFAULT 5,              -- 1-10, higher is more important
    source TEXT DEFAULT 'auto',              -- 'auto', 'user', 'trending', 'correction'
    urls TEXT,                               -- Comma-separated list of URLs to scrape
    keywords TEXT,                           -- Comma-separated keywords for searching
    discovered_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    started_at DATETIME,                     -- When we started learning this topic
    completed INTEGER DEFAULT 0,             -- 0 or 1
    completed_at DATETIME,
    experiences_count INTEGER DEFAULT 0,     -- Number of experiences for this topic
    pages_scraped INTEGER DEFAULT 0          -- Number of pages scraped for this topic
);

-- Indexes for learning goals
CREATE INDEX IF NOT EXISTS idx_goals_priority ON learning_goals(priority DESC);
CREATE INDEX IF NOT EXISTS idx_goals_completed ON learning_goals(completed);
CREATE INDEX IF NOT EXISTS idx_goals_source ON learning_goals(source);


-- =============================================================================
-- Training Runs Table
-- =============================================================================
-- History of training runs

CREATE TABLE IF NOT EXISTS training_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at DATETIME NOT NULL,
    completed_at DATETIME,
    status TEXT DEFAULT 'running',           -- 'running', 'completed', 'failed', 'cancelled'

    -- Data used
    experiences_used INTEGER DEFAULT 0,      -- Number of experiences used
    pages_used INTEGER DEFAULT 0,            -- Number of scraped pages used

    -- Training metrics
    training_steps INTEGER DEFAULT 0,
    training_epochs INTEGER DEFAULT 0,
    learning_rate REAL,
    batch_size INTEGER,
    final_loss REAL,
    final_eval_loss REAL,

    -- Model output
    model_path TEXT,                         -- Path to trained model
    gguf_path TEXT,                          -- Path to GGUF export
    gguf_size_mb REAL,                       -- GGUF file size in MB
    gguf_checksum TEXT,                      -- SHA256 checksum

    -- Deployment
    gcs_path TEXT,                           -- GCS upload path
    deployed_to_local INTEGER DEFAULT 0,     -- Deployed to JARVIS-Prime local
    deployed_to_cloud INTEGER DEFAULT 0,     -- Deployed to Cloud Run

    -- Error handling
    error TEXT,
    error_traceback TEXT,

    -- Metadata
    config_json TEXT                         -- JSON: full training config used
);

-- Indexes for training runs
CREATE INDEX IF NOT EXISTS idx_runs_status ON training_runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_started ON training_runs(started_at DESC);


-- =============================================================================
-- Model Deployments Table
-- =============================================================================
-- Record of deployed models

CREATE TABLE IF NOT EXISTS model_deployments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    training_run_id INTEGER,                 -- Foreign key to training_runs
    model_name TEXT NOT NULL,                -- Model identifier
    model_version TEXT,                      -- Semantic version
    model_path TEXT NOT NULL,                -- Local path
    gguf_path TEXT,                          -- GGUF path
    checksum TEXT,                           -- SHA256 checksum
    model_size_mb REAL,                      -- File size in MB
    deployed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    deployed_to TEXT,                        -- 'local', 'cloud_run', 'both'
    gcs_path TEXT,                           -- GCS path if uploaded
    active INTEGER DEFAULT 1,                -- 1 if currently active model
    performance_score REAL,                  -- Post-deployment performance metric
    rollback_reason TEXT                     -- If rolled back, why
);

-- Indexes for model deployments
CREATE INDEX IF NOT EXISTS idx_deployments_active ON model_deployments(active);
CREATE INDEX IF NOT EXISTS idx_deployments_name ON model_deployments(model_name);


-- =============================================================================
-- Intelligence Insights Table
-- =============================================================================
-- Insights from the CAI (Collective AI Intelligence) system

CREATE TABLE IF NOT EXISTS intelligence_insights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    insight_type TEXT NOT NULL,              -- 'pattern', 'recommendation', 'anomaly'
    topic TEXT NOT NULL,
    source_systems TEXT,                     -- Comma-separated: 'uae', 'sai', 'mas', 'cai'
    confidence REAL DEFAULT 0.5,             -- 0.0 to 1.0
    description TEXT NOT NULL,
    data_json TEXT,                          -- JSON: additional insight data
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    acted_upon INTEGER DEFAULT 0,            -- 1 if insight was used
    outcome TEXT                             -- Result of acting on insight
);

-- Indexes for intelligence insights
CREATE INDEX IF NOT EXISTS idx_insights_type ON intelligence_insights(insight_type);
CREATE INDEX IF NOT EXISTS idx_insights_topic ON intelligence_insights(topic);
CREATE INDEX IF NOT EXISTS idx_insights_confidence ON intelligence_insights(confidence DESC);


-- =============================================================================
-- Agent Tasks Table
-- =============================================================================
-- Tasks executed by the MAS (Multi-Agent System)

CREATE TABLE IF NOT EXISTS agent_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT UNIQUE NOT NULL,            -- UUID
    parent_task_id TEXT,                     -- For subtasks
    goal TEXT NOT NULL,                      -- Task description
    agent_type TEXT,                         -- 'explorer', 'creator', 'analyzer', 'scraper'
    agent_id TEXT,                           -- Assigned agent UUID
    priority INTEGER DEFAULT 5,              -- 1-10
    status TEXT DEFAULT 'pending',           -- 'pending', 'running', 'completed', 'failed'
    context_json TEXT,                       -- JSON: task context
    result_json TEXT,                        -- JSON: task result
    error TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    started_at DATETIME,
    completed_at DATETIME,
    execution_time_ms INTEGER
);

-- Indexes for agent tasks
CREATE INDEX IF NOT EXISTS idx_tasks_status ON agent_tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_priority ON agent_tasks(priority DESC);
CREATE INDEX IF NOT EXISTS idx_tasks_agent ON agent_tasks(agent_type);


-- =============================================================================
-- Triggers
-- =============================================================================

-- Auto-update word_count for scraped content
CREATE TRIGGER IF NOT EXISTS update_word_count
AFTER INSERT ON scraped_content
BEGIN
    UPDATE scraped_content
    SET word_count = (
        SELECT (LENGTH(content) - LENGTH(REPLACE(content, ' ', '')) + 1)
    )
    WHERE id = NEW.id;
END;

-- Auto-update experiences_count for learning goals
CREATE TRIGGER IF NOT EXISTS update_goal_experiences
AFTER INSERT ON experiences
WHEN NEW.context LIKE '%"topic":%'
BEGIN
    UPDATE learning_goals
    SET experiences_count = experiences_count + 1
    WHERE topic = json_extract(NEW.context, '$.topic');
END;


-- =============================================================================
-- Views
-- =============================================================================

-- View: Training-ready experiences
CREATE VIEW IF NOT EXISTS v_training_ready_experiences AS
SELECT
    id, input_text, output_text, context, quality_score
FROM experiences
WHERE used_in_training = 0
    AND quality_score >= 0.3
ORDER BY quality_score DESC, timestamp DESC;

-- View: Active learning goals
CREATE VIEW IF NOT EXISTS v_active_learning_goals AS
SELECT
    id, topic, priority, source, urls,
    experiences_count, pages_scraped
FROM learning_goals
WHERE completed = 0
ORDER BY priority DESC;

-- View: Recent training runs
CREATE VIEW IF NOT EXISTS v_recent_training_runs AS
SELECT
    id, started_at, completed_at, status,
    experiences_used, pages_used,
    training_steps, final_loss,
    gguf_path, gcs_path
FROM training_runs
ORDER BY started_at DESC
LIMIT 10;

-- View: Model deployment history
CREATE VIEW IF NOT EXISTS v_model_history AS
SELECT
    d.id, d.model_name, d.model_version,
    d.deployed_at, d.deployed_to, d.active,
    r.training_steps, r.final_loss
FROM model_deployments d
LEFT JOIN training_runs r ON d.training_run_id = r.id
ORDER BY d.deployed_at DESC;


-- =============================================================================
-- Sample Data (for testing - comment out in production)
-- =============================================================================

-- Insert sample learning goals
-- INSERT OR IGNORE INTO learning_goals (topic, priority, source) VALUES
--     ('Python asyncio patterns', 8, 'user'),
--     ('LangChain integration', 7, 'auto'),
--     ('GGUF model optimization', 6, 'auto'),
--     ('Voice biometric security', 9, 'user'),
--     ('macOS automation', 5, 'auto');
