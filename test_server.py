import pytest
import tempfile
import os
import shutil
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from server import app, download_file_from_github, download_file_from_gcp_bucket, analyze_code_with_gemini
import requests
from google.cloud import storage
from fastapi import HTTPException

# Helper function to create a temporary file
def create_temp_file(content, suffix=".py"):
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(content)
        return tmp_file.name


# Unit tests
@patch("requests.get")
def test_download_file_from_github_success(mock_get):
    mock_response = MagicMock()
    mock_response.text = "Test file content"
    mock_response.status_code = 200  # Explicitly set status code
    mock_get.return_value = mock_response
    content = download_file_from_github("https://github.com/test/repo/blob/main/test.py")
    assert content == "Test file content"


@patch("requests.get")
def test_download_file_from_github_error(mock_get):
    mock_get.side_effect = requests.exceptions.RequestException("Test error")
    with pytest.raises(HTTPException) as exc_info:
        download_file_from_github("https://github.com/test/repo/blob/main/test.py")
    assert "Error downloading file from GitHub" in str(exc_info.value)


@patch.object(storage, "Client")
def test_download_file_from_gcp_bucket_success(mock_client):
    mock_blob = MagicMock()
    mock_blob.download_as_text.return_value = "Test file content"
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    mock_client.return_value.bucket.return_value = mock_bucket
    content = download_file_from_gcp_bucket("gs://test-bucket/test.py")
    assert content == "Test file content"


@patch.object(storage, "Client")
def test_download_file_from_gcp_bucket_invalid_url(mock_client):
    with pytest.raises(HTTPException) as exc_info:
        download_file_from_gcp_bucket("invalid-url")
    assert "Invalid GCP Bucket URL format" in str(exc_info.value)


@patch.object(storage, "Client")
def test_download_file_from_gcp_bucket_error(mock_client):
    mock_client.side_effect = Exception("Test error")
    with pytest.raises(HTTPException) as exc_info:
        download_file_from_gcp_bucket("gs://test-bucket/test.py")
    assert "Error downloading file from GCP Bucket" in str(exc_info.value)


@patch("server.model.generate_content")
def test_analyze_code_with_gemini_success(mock_generate_content):
    mock_response = MagicMock()
    mock_response.text = """
    {
        "analysis": "Good code",
        "performance_evaluation": "Fast",
        "coding_standards_evaluation": "Compliant",
        "no_go": false
    }
    """
    mock_response.status_code = 200
    mock_generate_content.return_value = mock_response
    result = analyze_code_with_gemini("print('hello')", "python")
    assert result["analysis"] == "Good code"
    assert result["no_go"] == False


@patch("server.model.generate_content")
def test_analyze_code_with_gemini_error(mock_generate_content):
    mock_generate_content.side_effect = Exception("Gemini API error")
    result = analyze_code_with_gemini("print('hello')", "text/x-python")
    assert "Error during Gemini API call" in result["analysis"]
    assert result["no_go"] is True


# Integration tests
client = TestClient(app)

@patch("server.analyze_code_with_gemini")
def test_analyze_code_endpoint_success(mock_analyze_code_with_gemini):
    mock_analyze_code_with_gemini.return_value = {
        "analysis": "Good code",
        "performance_evaluation": "Fast",
        "coding_standards_evaluation": "Compliant",
        "no_go": False,
        "raw_gemini_response": "Raw response"
    }

    response = client.post(
        "/analyze/", json={"code": "print('hello')", "file_type": "text/x-python"}
    )
    assert response.status_code == 200
    assert "analysis" in response.json()


def test_analyze_code_endpoint_no_input():
    response = client.post("/analyze/")
    assert response.status_code == 400
    assert "No input provided" in response.json()["detail"]


def test_analyze_code_endpoint_invalid_request():
    response = client.post("/analyze/", json={})
    assert response.status_code == 400
    assert "Please provide code, github_url or gcp_bucket_url in request data." in response.json()["detail"]


def test_analyze_code_endpoint_code_without_file_type():
    response = client.post("/analyze/", json={"code": "print('hello')"})
    assert response.status_code == 400
    assert "file_type is required when using code." in response.json()["detail"]


@patch("server.download_file_from_github")
def test_analyze_code_endpoint_github_url_success(mock_download):
    mock_download.return_value = "print('hello')"
    response = client.post(
        "/analyze/", json={"github_url": "https://github.com/test/repo/blob/main/test.py"}
    )
    assert response.status_code == 200
    assert "analysis" in response.json()


@patch("server.download_file_from_gcp_bucket")
def test_analyze_code_endpoint_gcp_bucket_url_success(mock_download):
    mock_download.return_value = "print('hello')"
    response = client.post(
        "/analyze/", json={"gcp_bucket_url": "gs://test-bucket/test.py"}
    )
    assert response.status_code == 200
    assert "analysis" in response.json()

def test_analyze_code_endpoint_github_url_live_success():
    response = client.post(
        "/analyze/", json={"github_url": "https://github.com/gpapp/notebooks/blob/main/Clean up transcription.ipynb"}
    )
    assert response.status_code == 200
    assert "analysis" in response.json()

