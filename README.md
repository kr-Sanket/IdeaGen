# IdeaGen

**Evidence-Grounded Research Opportunity Discovery and Evolution Tracking using Deep Learning**

## Project Structure

```
IdeaGen/
├── backend/         # Python/FastAPI backend
│   ├── ideagen/     # Installable package
│   │   ├── core/    # Configuration and shared utilities
│   │   └── ...
│   ├── tests/       # Backend test suite
│   ├── pyproject.toml
│   ├── .env.example
│   └── README.md
├── frontend/        # React + Vite + TypeScript explorer frontend
├── data/
│   ├── raw/         # Raw data from sources (e.g., OpenAlex)
│   ├── processed/   # Cleaned and transformed data
│   └── indexes/     # Search indexes and embeddings
├── docs/            # Project documentation and specifications
├── notebooks/       # Jupyter notebooks for exploration and analysis
└── README.md
```

## Quick Start

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -e ".[dev]"
cp .env.example .env
pytest
```

## Running Locally

```bash
make install   # backend/.venv + frontend node_modules (existing .env untouched)
make dev       # backend :8000 + frontend :5173; Ctrl+C stops both
make status    # who owns ports 8000/5173 (Makefile-managed or external)
make test      # backend suite on temporary databases + frontend type check
```

- `make backend` / `make dev` default `IDEAGEN_DATABASE_URL` to `data/demo/ideagen_demo_20261022.db` (writable).
  The test suite refuses to run against any SQLite file under `data/`.
- **Start the servers from a regular terminal.** A backend launched from a sandboxed agent inherits its
  `HTTP(S)_PROXY`; that policy proxy rejects OpenAlex ("not allowed by policy") and even the local Ollama port
  ("Port 11434 not allowed"). A Vite server launched in a sandbox may be unable to connect to the backend
  (`connect EPERM 127.0.0.1:8000`), which the UI reports as **API Offline**. The Makefile warns when proxy
  variables are set; `NO_PROXY=localhost,127.0.0.1` keeps local Ollama traffic off a proxy.
- Settings → *Application Settings & Diagnostics* shows the configured models and whether Ollama and the
  required models are actually available (`GET /api/v1/system/status`).
- Point a second frontend at another backend with `IDEAGEN_API_PROXY_TARGET=http://127.0.0.1:<port> npx vite --port <p>`.

## Current Status & Limitations

- **Backend Pipeline & Explorer UI:** Core evidence-grounded research opportunity discovery, synthesis, and temporal evolution pipelines are implemented. A read-only React + Vite explorer interface is available for inspecting runs, opportunities, and statement provenance.
- **No Authentication or Multi-Tenant Workspaces:** User accounts, authentication, session isolation, and multi-tenant workspaces are **not implemented**. API endpoints (including `/api/v1/stats`) report global database aggregates across all persisted records. The current frontend is an exploratory viewer, not a multi-user account/workspace system.
- **Exploratory Model Evaluation:** LLM relationship classification and statement extraction evaluations are exploratory and conducted on small prototype sets; passing automated test suites demonstrates software correctness and deterministic pipeline mechanics, not broad scientific accuracy on open-domain literature.
- **Database Environments:** PostgreSQL is the target production datastore with Alembic migrations; automated CI tests run against SQLite. PostgreSQL-specific features require a live database service to be verified.
- Detailed status, verified test outputs, and known limitations are maintained in [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md).
