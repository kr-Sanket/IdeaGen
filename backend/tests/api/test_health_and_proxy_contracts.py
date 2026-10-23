from unittest.mock import patch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError
from ideagen.api.main import app, get_db

client = TestClient(app)

test_engine = create_engine("sqlite:///:memory:")
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def test_health_endpoints_return_healthy():
    """Verify all health endpoints return HTTP 200 with 'healthy' status and connected DB."""
    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    try:
        for path in ["/api/health", "/api/v1/health", "/health"]:
            response = client.get(path)
            assert response.status_code == 200, f"Endpoint {path} failed with {response.status_code}"
            data = response.json()
            assert data["status"] == "healthy"
            assert "connected" in data["database"]
    finally:
        app.dependency_overrides.pop(get_db, None)


def test_health_endpoint_db_failure():
    """Verify that if the database is unreachable, health endpoint returns 503."""
    class FailingDBSession:
        bind = None

        def execute(self, stmt):
            raise OperationalError("SELECT 1", {}, Exception("Connection refused"))

    def override_get_db():
        yield FailingDBSession()

    app.dependency_overrides[get_db] = override_get_db
    try:
        response = client.get("/api/health")
        assert response.status_code == 503
        data = response.json()
        assert "Database connection error" in data["detail"]
    finally:
        app.dependency_overrides.pop(get_db, None)


def test_client_connectivity_contract_simulation():
    """Simulates frontend client checkBackendConnection behavior across responses."""
    def simulate_check(status_code: int, throws: bool = False) -> bool:
        if throws:
            return False
        return 200 <= status_code < 300

    # HTTP 200 -> backend is online
    assert simulate_check(200) is True

    # HTTP 502 (Vite proxy bad gateway when backend or proxy socket fails) -> offline
    assert simulate_check(502) is False

    # HTTP 503 (FastAPI reports database connection error) -> offline
    assert simulate_check(503) is False

    # HTTP 500 (Internal server error) -> offline
    assert simulate_check(500) is False

    # Network / socket drop (Fetch throws TypeError) -> offline
    assert simulate_check(0, throws=True) is False
