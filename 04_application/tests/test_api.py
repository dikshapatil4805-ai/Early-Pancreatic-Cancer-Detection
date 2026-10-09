from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "API is running"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["api_running"] is True


def test_predict_rejects_unsupported_file():
    response = client.post(
        "/predict",
        files={
            "ct_file": (
                "scan.txt",
                b"test content",
                "text/plain",
            )
        },
        data={
            "clinical_json": '{"age": 50}'
        },
    )

    assert response.status_code == 400


def test_predict_reports_unconfigured_model():
    response = client.post(
        "/predict",
        files={
            "ct_file": (
                "scan.dcm",
                b"placeholder-not-a-real-dicom-file",
                "application/octet-stream",
            )
        },
        data={
            "clinical_json": '{"age": 50}'
        },
    )

    assert response.status_code == 503
    assert "not available yet" in response.json()["detail"]


def test_predict_rejects_invalid_age():
    response = client.post(
        "/predict",
        files={
            "ct_file": (
                "scan.dcm",
                b"placeholder",
                "application/octet-stream",
            )
        },
        data={
            "clinical_json": '{"age": 10}'
        },
    )

    assert response.status_code == 422