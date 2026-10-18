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
├── frontend/        # Next.js frontend (not yet implemented)
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

## Status

✅ **Phase 0 — Backend Foundation** complete. See [`docs/06_STATE_AND_SESSION.md`](docs/06_STATE_AND_SESSION.md) for details.
