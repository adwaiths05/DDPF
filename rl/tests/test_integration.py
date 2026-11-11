# tests/test_integration.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pytest
from src.integration_adapter import IntegrationAdapter


def test_adapter_initialize_ppo():
    adapter = IntegrationAdapter(policy_type="ppo")
    obs = adapter.initialize(seed=0)
    assert obs is not None
    assert len(obs) > 0


def test_adapter_step_ppo():
    adapter = IntegrationAdapter(policy_type="ppo")
    adapter.initialize(seed=0)
    result = adapter.step()
    assert "action" in result
    assert "reward" in result
    assert "done" in result
    assert "observation" in result


def test_adapter_reset():
    adapter = IntegrationAdapter(policy_type="ppo")
    adapter.initialize(seed=0)
    obs = adapter.reset(seed=1)
    assert obs is not None


def test_adapter_close():
    adapter = IntegrationAdapter(policy_type="ppo")
    adapter.initialize(seed=0)
    adapter.close()
    assert adapter.env is None
