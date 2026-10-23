# ==============================================================================
# IdeaGen — Local Development Makefile
# Evidence-Grounded Research Opportunity Discovery & Evolution Tracking
# ==============================================================================

.DEFAULT_GOAL := help
.PHONY: help install backend frontend dev stop status test lint clean-run

# ── Project Directory Resolution ──────────────────────────────────────────────
ROOT_DIR     := $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
BACKEND_DIR  := $(ROOT_DIR)/backend
FRONTEND_DIR := $(ROOT_DIR)/frontend
RUN_DIR      := $(ROOT_DIR)/.run

# ── Host & Port Configuration ─────────────────────────────────────────────────
BACKEND_HOST  ?= 127.0.0.1
BACKEND_PORT  ?= 8000
FRONTEND_HOST ?= 127.0.0.1
FRONTEND_PORT ?= 5173

# ── Environment & Database ────────────────────────────────────────────────────
# Preserve existing IDEAGEN_DATABASE_URL if already set; otherwise default to demo DB
DEFAULT_DEMO_DB := $(ROOT_DIR)/data/demo/ideagen_demo_20261022.db
export IDEAGEN_DATABASE_URL ?= sqlite:///$(DEFAULT_DEMO_DB)

# ── Tooling Paths (Prefer existing backend virtual environment) ───────────────
# $(wildcard) splits on spaces, and this repository path contains one ("IdeaGen F"), so probe with
# the shell instead. Recipes quote these paths.
VENV_DIR     := $(BACKEND_DIR)/.venv
venv_tool     = $(shell [ -x "$(VENV_DIR)/bin/$(1)" ] && printf '%s' "$(VENV_DIR)/bin/$(1)" || printf '%s' "$(2)")
PYTHON       := $(call venv_tool,python,python3)
UVICORN      := $(call venv_tool,uvicorn,uvicorn)
PIP          := $(call venv_tool,pip,pip)
PYTEST       := $(call venv_tool,pytest,pytest)
NPM          := npm

# Warn when the shell routes HTTP through a proxy: the backend inherits it for OpenAlex and for the
# local Ollama server. A sandbox/policy proxy there yields "not allowed by policy" (OpenAlex) or
# "Port 11434 not allowed" (Ollama). The backend honours NO_PROXY for hosts that must bypass it.
define PROXY_WARNING
	@if [ -n "$$HTTPS_PROXY$$https_proxy$$HTTP_PROXY$$http_proxy$$ALL_PROXY$$all_proxy" ]; then \
		echo "WARNING: proxy variables are set in this shell (HTTPS_PROXY=$${HTTPS_PROXY:-$$https_proxy} HTTP_PROXY=$${HTTP_PROXY:-$$http_proxy} NO_PROXY=$${NO_PROXY:-$$no_proxy})."; \
		echo "         The backend will send OpenAlex and Ollama requests through that proxy. If it is a sandbox"; \
		echo "         or policy proxy, start IdeaGen from a regular terminal or set NO_PROXY=localhost,127.0.0.1."; \
	fi
endef

# ── Targets ───────────────────────────────────────────────────────────────────

## help: Show this help message and service configuration
help:
	@echo "======================================================================"
	@echo "                      IdeaGen Development Makefile"
	@echo "======================================================================"
	@echo "Usage: make <target>"
	@echo ""
	@echo "Primary Targets:"
	@echo "  help        Display this help message with available commands"
	@echo "  install     Install backend and frontend dependencies without altering configs"
	@echo "  backend     Start FastAPI backend server on http://$(BACKEND_HOST):$(BACKEND_PORT)"
	@echo "  frontend    Start Vite frontend dev server on http://$(FRONTEND_HOST):$(FRONTEND_PORT)"
	@echo "  dev         Start both backend and frontend concurrently with prefixed logs"
	@echo "  stop        Safely stop only processes started by this Makefile"
	@echo "  status      Check listener status on ports $(BACKEND_PORT) and $(FRONTEND_PORT)"
	@echo ""
	@echo "Quality Targets:"
	@echo "  test        Run backend pytest suite (temporary DBs) and frontend type check"
	@echo "  lint        Run frontend oxlint check"
	@echo ""
	@echo "Current Configuration:"
	@echo "  Backend URL:   http://$(BACKEND_HOST):$(BACKEND_PORT)"
	@echo "  API Docs:      http://$(BACKEND_HOST):$(BACKEND_PORT)/docs"
	@echo "  Frontend URL:  http://$(FRONTEND_HOST):$(FRONTEND_PORT)"
	@echo "  Database URL:  $(IDEAGEN_DATABASE_URL)"
	@echo "  Python venv:   $(VENV_DIR)"
	@echo "======================================================================"

## install: Install backend (.venv + pip) and frontend (npm) dependencies
install:
	@echo "--> Checking backend virtual environment..."
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Creating virtual environment at $(VENV_DIR)..."; \
		python3 -m venv "$(VENV_DIR)"; \
	else \
		echo "Using existing virtual environment at $(VENV_DIR)."; \
	fi
	@echo "--> Installing backend dependencies in editable mode..."
	@cd "$(BACKEND_DIR)" && "$(VENV_DIR)/bin/pip" install -e ".[dev]"
	@echo "--> Installing frontend dependencies..."
	@cd "$(FRONTEND_DIR)" && "$(NPM)" install
	@echo "--> Dependencies installed successfully."
	@echo "    Note: Existing configuration (.env) was preserved and untouched."

## backend: Start the backend server in foreground
backend:
	@pid=$$(lsof -iTCP:$(BACKEND_PORT) -sTCP:LISTEN -t 2>/dev/null | head -n 1); \
	if [ -n "$$pid" ]; then \
		echo "ERROR: Backend port $(BACKEND_PORT) is already occupied by PID $$pid."; \
		echo "To avoid conflicts, stop the existing process before starting (or run 'make status')."; \
		exit 1; \
	fi
	$(PROXY_WARNING)
	@echo "--> Starting IdeaGen backend on http://$(BACKEND_HOST):$(BACKEND_PORT)..."
	@echo "    Database: $(IDEAGEN_DATABASE_URL)"
	@cd "$(BACKEND_DIR)" && exec "$(UVICORN)" ideagen.api.main:app --host $(BACKEND_HOST) --port $(BACKEND_PORT) --reload

## frontend: Start the frontend server in foreground
frontend:
	@pid=$$(lsof -iTCP:$(FRONTEND_PORT) -sTCP:LISTEN -t 2>/dev/null | head -n 1); \
	if [ -n "$$pid" ]; then \
		echo "ERROR: Frontend port $(FRONTEND_PORT) is already occupied by PID $$pid."; \
		echo "To avoid conflicts, stop the existing process before starting (or run 'make status')."; \
		exit 1; \
	fi
	@echo "--> Starting IdeaGen frontend on http://$(FRONTEND_HOST):$(FRONTEND_PORT)..."
	@cd "$(FRONTEND_DIR)" && exec "$(NPM)" run dev -- --host $(FRONTEND_HOST) --port $(FRONTEND_PORT)

## dev: Start both services concurrently with prefixed logs and graceful shutdown
dev:
	@mkdir -p "$(RUN_DIR)"
	@# Preflight check for occupied ports
	@b_pid=$$(lsof -iTCP:$(BACKEND_PORT) -sTCP:LISTEN -t 2>/dev/null | head -n 1); \
	if [ -n "$$b_pid" ]; then \
		echo "ERROR: Backend port $(BACKEND_PORT) is already occupied by PID $$b_pid."; \
		echo "Run 'make status' to inspect, or terminate PID $$b_pid before starting."; \
		exit 1; \
	fi
	@f_pid=$$(lsof -iTCP:$(FRONTEND_PORT) -sTCP:LISTEN -t 2>/dev/null | head -n 1); \
	if [ -n "$$f_pid" ]; then \
		echo "ERROR: Frontend port $(FRONTEND_PORT) is already occupied by PID $$f_pid."; \
		echo "Run 'make status' to inspect, or terminate PID $$f_pid before starting."; \
		exit 1; \
	fi
	$(PROXY_WARNING)
	@echo "======================================================================"
	@echo "               Starting IdeaGen Concurrent Development"
	@echo "======================================================================"
	@echo "  Backend:   http://$(BACKEND_HOST):$(BACKEND_PORT)"
	@echo "  Frontend:  http://$(FRONTEND_HOST):$(FRONTEND_PORT)"
	@echo "  Database:  $(IDEAGEN_DATABASE_URL)"
	@echo "  PIDs dir:  $(RUN_DIR)"
	@echo "  Press Ctrl+C to cleanly stop both services."
	@echo "======================================================================"
	@trap 'echo ""; echo "--> Stopping development processes..."; \
		if [ -f "$(RUN_DIR)/backend.pid" ]; then \
			bpid=$$(cat "$(RUN_DIR)/backend.pid"); \
			pkill -P $$bpid 2>/dev/null || true; \
			kill -TERM $$bpid 2>/dev/null || true; \
			rm -f "$(RUN_DIR)/backend.pid"; \
		fi; \
		if [ -f "$(RUN_DIR)/frontend.pid" ]; then \
			fpid=$$(cat "$(RUN_DIR)/frontend.pid"); \
			pkill -P $$fpid 2>/dev/null || true; \
			kill -TERM $$fpid 2>/dev/null || true; \
			rm -f "$(RUN_DIR)/frontend.pid"; \
		fi; \
		rmdir "$(RUN_DIR)" 2>/dev/null || true; \
		echo "--> IdeaGen services stopped cleanly."; \
		exit 0' INT TERM EXIT; \
	(cd "$(BACKEND_DIR)" && "$(UVICORN)" ideagen.api.main:app --host $(BACKEND_HOST) --port $(BACKEND_PORT) --reload 2>&1 | awk '{print "[backend]  " $$0; fflush()}') & \
	echo $$! > "$(RUN_DIR)/backend.pid"; \
	(cd "$(FRONTEND_DIR)" && "$(NPM)" run dev -- --host $(FRONTEND_HOST) --port $(FRONTEND_PORT) 2>&1 | awk '{print "[frontend] " $$0; fflush()}') & \
	echo $$! > "$(RUN_DIR)/frontend.pid"; \
	wait

## stop: Safely stop only processes recorded by this Makefile
stop:
	@echo "--> Checking for Makefile-managed processes in $(RUN_DIR)..."
	@stopped=0; \
	if [ -f "$(RUN_DIR)/backend.pid" ]; then \
		bpid=$$(cat "$(RUN_DIR)/backend.pid"); \
		if kill -0 $$bpid 2>/dev/null; then \
			echo "Stopping managed backend process (PID $$bpid)..."; \
			pkill -P $$bpid 2>/dev/null || true; \
			kill -TERM $$bpid 2>/dev/null || true; \
			stopped=1; \
		else \
			echo "Backend PID $$bpid is not running (stale pidfile)."; \
		fi; \
		rm -f "$(RUN_DIR)/backend.pid"; \
	else \
		echo "No managed backend PID file found."; \
	fi; \
	if [ -f "$(RUN_DIR)/frontend.pid" ]; then \
		fpid=$$(cat "$(RUN_DIR)/frontend.pid"); \
		if kill -0 $$fpid 2>/dev/null; then \
			echo "Stopping managed frontend process (PID $$fpid)..."; \
			pkill -P $$fpid 2>/dev/null || true; \
			kill -TERM $$fpid 2>/dev/null || true; \
			stopped=1; \
		else \
			echo "Frontend PID $$fpid is not running (stale pidfile)."; \
		fi; \
		rm -f "$(RUN_DIR)/frontend.pid"; \
	else \
		echo "No managed frontend PID file found."; \
	fi; \
	rmdir "$(RUN_DIR)" 2>/dev/null || true; \
	if [ $$stopped -eq 1 ]; then \
		echo "--> Managed services stopped."; \
	fi
	@# Inspect if ports are still listening by non-Makefile processes
	@b_active=$$(lsof -iTCP:$(BACKEND_PORT) -sTCP:LISTEN -t 2>/dev/null | head -n 1); \
	if [ -n "$$b_active" ]; then \
		echo "NOTE: Port $(BACKEND_PORT) is currently listening by PID $$b_active (unmanaged / external process). It was preserved."; \
	fi
	@f_active=$$(lsof -iTCP:$(FRONTEND_PORT) -sTCP:LISTEN -t 2>/dev/null | head -n 1); \
	if [ -n "$$f_active" ]; then \
		echo "NOTE: Port $(FRONTEND_PORT) is currently listening by PID $$f_active (unmanaged / external process). It was preserved."; \
	fi

## status: Inspect port listeners and managed PID state
status:
	@echo "======================================================================"
	@echo "                   IdeaGen Service Status"
	@echo "======================================================================"
	@# Backend check
	@b_pid=$$(lsof -iTCP:$(BACKEND_PORT) -sTCP:LISTEN -t 2>/dev/null | head -n 1); \
	if [ -n "$$b_pid" ]; then \
		b_cmd=$$(ps -p $$b_pid -o comm= 2>/dev/null || echo "unknown"); \
		echo "  Backend (Port $(BACKEND_PORT)):   LISTENING (PID $$b_pid: $$b_cmd)"; \
		echo "    URL:        http://$(BACKEND_HOST):$(BACKEND_PORT)"; \
		echo "    API Docs:   http://$(BACKEND_HOST):$(BACKEND_PORT)/docs"; \
	else \
		echo "  Backend (Port $(BACKEND_PORT)):   STOPPED (no listener)"; \
	fi
	@# Frontend check
	@f_pid=$$(lsof -iTCP:$(FRONTEND_PORT) -sTCP:LISTEN -t 2>/dev/null | head -n 1); \
	if [ -n "$$f_pid" ]; then \
		f_cmd=$$(ps -p $$f_pid -o comm= 2>/dev/null || echo "unknown"); \
		echo "  Frontend (Port $(FRONTEND_PORT)):  LISTENING (PID $$f_pid: $$f_cmd)"; \
		echo "    URL:        http://$(FRONTEND_HOST):$(FRONTEND_PORT)"; \
	else \
		echo "  Frontend (Port $(FRONTEND_PORT)):  STOPPED (no listener)"; \
	fi
	@echo ""
	@echo "  PID Tracking:"
	@if [ -f "$(RUN_DIR)/backend.pid" ]; then \
		echo "    Managed Backend PID file:  $$(cat "$(RUN_DIR)/backend.pid")"; \
	else \
		echo "    Managed Backend PID file:  None"; \
	fi
	@if [ -f "$(RUN_DIR)/frontend.pid" ]; then \
		echo "    Managed Frontend PID file: $$(cat "$(RUN_DIR)/frontend.pid")"; \
	else \
		echo "    Managed Frontend PID file: None"; \
	fi
	@echo ""
	@echo "  Database Configuration:"
	@echo "    Target URL: $(IDEAGEN_DATABASE_URL)"
	@if [ -f "$(DEFAULT_DEMO_DB)" ]; then \
		echo "    Demo DB:    Found at $(DEFAULT_DEMO_DB)"; \
	else \
		echo "    Demo DB:    Not found at $(DEFAULT_DEMO_DB)"; \
	fi
	@echo "======================================================================"

## test: Run backend automated tests and frontend type checking
# Tests must never see the demo database URL exported above: several fixtures run DDL and
# transactions against IDEAGEN_DATABASE_URL. Unsetting it makes them use temporary SQLite databases.
test:
	@echo "--> Running backend pytest suite (IDEAGEN_DATABASE_URL unset; tests use temporary databases)..."
	@cd "$(BACKEND_DIR)" && env -u IDEAGEN_DATABASE_URL "$(PYTEST)" tests/
	@echo "--> Type-checking frontend..."
	@cd "$(FRONTEND_DIR)" && npx tsc -b --noEmit

## lint: Run frontend linter
lint:
	@echo "--> Running frontend oxlint..."
	@cd "$(FRONTEND_DIR)" && "$(NPM)" run lint
