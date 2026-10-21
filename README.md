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

## Current Status & Limitations

- **Backend Pipeline & Explorer UI:** Core evidence-grounded research opportunity discovery, synthesis, and temporal evolution pipelines are implemented. A read-only React + Vite explorer interface is available for inspecting runs, opportunities, and statement provenance.
- **No Authentication or Multi-Tenant Workspaces:** User accounts, authentication, session isolation, and multi-tenant workspaces are **not implemented**. API endpoints (including `/api/v1/stats`) report global database aggregates across all persisted records. The current frontend is an exploratory viewer, not a multi-user account/workspace system.
- **Exploratory Model Evaluation:** LLM relationship classification and statement extraction evaluations are exploratory and conducted on small prototype sets; passing automated test suites demonstrates software correctness and deterministic pipeline mechanics, not broad scientific accuracy on open-domain literature.
- **Database Environments:** PostgreSQL is the target production datastore with Alembic migrations; automated CI tests run against SQLite. PostgreSQL-specific features require a live database service to be verified.
- Detailed status, verified test outputs, and known limitations are maintained in [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md).
