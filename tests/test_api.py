"""Tests for the FastAPI endpoints."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.server import app

client = TestClient(app)


class TestConfigEndpoints:
    def test_get_config(self):
        response = client.get("/config")
        assert response.status_code == 200
        data = response.json()
        assert "env" in data
        assert "train" in data
        assert "redis" in data

    def test_update_config(self):
        response = client.put(
            "/config",
            json={"env": {"lambda_a": 0.8, "lambda_b": 0.5}},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["env"]["lambda_a"] == 0.8
        assert data["env"]["lambda_b"] == 0.5


class TestTrainStatus:
    def test_train_status(self):
        response = client.get("/train/status")
        assert response.status_code == 200
        assert "status" in response.json()
