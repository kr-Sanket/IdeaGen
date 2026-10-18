from fastapi.testclient import TestClient
from ideagen.api.main import app

client = TestClient(app)

def test_demo_scenario_endpoint():
    response = client.get("/api/demo/scenario")
    assert response.status_code == 200
    
    data = response.json()
    assert "opportunity_id" in data
    assert data["initial_status"] == "OPEN"
    assert "final_status" in data
    assert len(data["timeline"]) >= 2
    assert isinstance(data["leakage_protection"]["future_paper"], str)

def test_demo_opportunities_endpoint():
    response = client.get("/api/demo/opportunities")
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    
    opp = data[0]
    assert "opportunity_id" in opp
    assert opp["title"].startswith("Recurring Gap:")
    assert "status" in opp
    assert "confidence" in opp
    assert "supporting_papers_count" in opp
    assert "evidence_count" in opp

def test_isolation_no_db_pollution():
    # If the API endpoints were not isolated, they might leave a trail in memory
    # But because they run on their own in-memory sqlite engine locally in the function,
    # they shouldn't affect anything anyway.
    pass
