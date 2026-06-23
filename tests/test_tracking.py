from recsys import tracking


def test_default_tracking_uri(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert tracking.get_tracking_uri() == "./mlruns"


def test_env_tracking_uri(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    assert tracking.get_tracking_uri() == "http://mlflow:5000"
