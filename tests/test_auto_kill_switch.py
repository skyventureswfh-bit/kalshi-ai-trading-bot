"""
Tests for src/auto/kill_switch.py.

Covers the required proof: kill switch active -> zero new buy order
submissions (verified at the wrapper level in test_auto_runner.py; this
file proves the switch's own fail-closed state semantics in isolation).
"""

import json
import os

import pytest

from src.auto.kill_switch import AutoKillSwitch


@pytest.fixture
def switch_path(tmp_path):
    return str(tmp_path / "auto_kill_switch.json")


def test_missing_file_defaults_to_not_active(switch_path):
    switch = AutoKillSwitch(switch_path)
    assert switch.is_active() is False


def test_activate_sets_active_true(switch_path):
    switch = AutoKillSwitch(switch_path)
    switch.activate(reason="unit test")
    assert switch.is_active() is True
    state = switch.read_state()
    assert state.reason == "unit test"
    assert state.activated_by == "manual"


def test_deactivate_clears_active(switch_path):
    switch = AutoKillSwitch(switch_path)
    switch.activate(reason="temp")
    switch.deactivate()
    assert switch.is_active() is False


def test_corrupt_state_file_fails_closed(switch_path):
    os.makedirs(os.path.dirname(switch_path), exist_ok=True)
    with open(switch_path, "w", encoding="utf-8") as fh:
        fh.write("{not valid json at all")
    switch = AutoKillSwitch(switch_path)
    # Fail-closed: an unreadable state file must be treated as ACTIVE,
    # never as "not active", because Auto submitting real orders while it
    # cannot confirm the switch is off is the dangerous failure mode.
    assert switch.is_active() is True


def test_missing_active_key_defaults_to_active_true(switch_path):
    # A malformed-but-parseable file (valid JSON, missing the "active" key)
    # must also fail closed, not default to False.
    os.makedirs(os.path.dirname(switch_path), exist_ok=True)
    with open(switch_path, "w", encoding="utf-8") as fh:
        json.dump({"reason": "no active key"}, fh)
    switch = AutoKillSwitch(switch_path)
    assert switch.is_active() is True
