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

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Copy and edit environment config
cp .env.example .env
```

## Configuration

All settings use the `IDEAGEN_` environment variable prefix.
See [`.env.example`](.env.example) for available options.

## Running Tests

```bash
pytest
```
