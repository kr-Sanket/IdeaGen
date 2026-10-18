# IdeaGen Backend

Backend services for the IdeaGen project.

## Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (macOS/Linux)
source .venv/bin/activate

# IdeaGen Backend

Backend services for the IdeaGen project.

## Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate (macOS/Linux)
source .venv/bin/activate
# Windows: .venv\Scripts\activate

# Install in editable mode with development and ML dependencies
pip install -e ".[dev,ml]"

# Copy and edit environment config (optional for offline testing)
cp .env.example .env
```

## Configuration

All settings use the `IDEAGEN_` environment variable prefix.
See [`.env.example`](.env.example) for available options.

## Testing Architecture

The backend test suite is divided into two distinct execution tiers:

### 1. Offline Tests (No Database Required)
- **Unit & Logic Suites**: `tests/opportunity/test_state_machine.py`, `tests/opportunity/test_classifier.py`, `tests/opportunity/test_discovery.py`, `tests/ingestion/`, `tests/rag/`, `tests/retrieval/`, `tests/generation/`, and `tests/test_config.py`.
- **In-Memory SQLite Suites**: `tests/api/test_upload.py`, `tests/api/test_phase6_api.py`, `tests/api/test_demo.py`, `tests/api/test_search.py`, and `tests/generation/test_opportunity_persistence.py`.
- If `IDEAGEN_DATABASE_URL` is unset, all database-dependent tests are automatically skipped with a clear notice, allowing offline tests to run without PostgreSQL or network access.

### 2. PostgreSQL Integration Tests
- **Suites**: `tests/db/test_session_and_foundation.py`, `tests/db/test_persistence.py`, `tests/opportunity/test_service.py`, `tests/opportunity/test_integration.py`, `tests/opportunity/test_temporal.py`, `tests/pipeline/test_end_to_end.py`, `tests/discovery/test_discovery_pipeline.py`, `tests/understanding/`, and `tests/ingestion/test_pdf_real.py`.
- **Prerequisites**:
  1. Start the PostgreSQL service:
     ```bash
     docker compose up -d postgres
     ```
  2. Apply database migrations to the test database:
     ```bash
     alembic upgrade head
     ```
  3. Provide the test database URL in `.env` or via environment variable:
     ```bash
     export IDEAGEN_DATABASE_URL="postgresql+psycopg2://ideagen:ideagen_dev@localhost:5432/ideagen_db"
     ```
- Test fixtures employ nested transaction savepoints (`join_transaction_mode="create_savepoint"`), rolling back test data upon completion to prevent database contamination.

## Running Tests

```bash
# Run all collectable tests (PostgreSQL suites auto-skip if IDEAGEN_DATABASE_URL is not set)
pytest

# Run only pure unit tests
pytest tests/opportunity/test_state_machine.py tests/test_config.py
```
