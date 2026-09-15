"""Issue #72 invariants for terminal, subthreshold replay sanitation."""

from __future__ import annotations

import json
import logging
import time
from copy import deepcopy
from unittest.mock import Mock

import pytest

import hermes_lcm.engine as lcm_engine
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens


def _engine(tmp_path, name: str, **overrides) -> LCMEngine:
    config = LCMConfig(
        database_path=str(tmp_path / f"{name}.db"),
        large_output_externalization_path=str(tmp_path / f"{name}-externalized"),
        fresh_tail_count=2,
        leaf_chunk_tokens=1,
        context_threshold=0.95,
        sensitive_patterns_enabled=True,
        sensitive_patterns=[
            "api_key",
            "bearer_token",
            "password_assignment",
            "private_key",
        ],
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / f"{name}-home"))
    engine.on_session_start(
        f"{name}-session",
        platform="synthetic",
        conversation_id=f"{name}-conversation",
        context_length=100_000,
    )
    return engine


def _content_text(messages) -> str:
    return json.dumps(messages, ensure_ascii=False, sort_keys=True)


def _claim_sanitation(engine, messages, *, generation: int = 1):
    engine._compression_attempt_generation = generation
    prepared = engine.prepare_compression_operation(
        deepcopy(messages),
        session_id=engine.bound_session_id,
        attempt_generation=generation,
    )
    assert prepared is not None
    operation, claim = prepared
    assert operation == "sanitize"
    assert claim is not None
    return claim


def test_claimed_sanitation_returns_exact_opaque_claim_once(tmp_path, monkeypatch):
    engine = _engine(tmp_path, "claimed-happy")
    engine.threshold_tokens = 90_000
    messages = [
        {
            "role": "user",
            "content": "api_key=sk-synthetic-claimed-happy-000000000000",
        },
        {"role": "assistant", "content": "fresh answer"},
    ]
    monkeypatch.setattr(
        lcm_engine,
        "summarize_with_escalation",
        Mock(side_effect=AssertionError("sanitation must not summarize")),
    )

    assert engine.should_compress_preflight(deepcopy(messages)) is True
    claim = _claim_sanitation(engine, messages, generation=17)

    sanitized, returned_claim = engine.compress(
        deepcopy(messages),
        current_tokens=count_messages_tokens(messages),
        operation_claim=claim,
    )

    assert returned_claim is claim
    assert engine.last_compression_status == "sanitized"
    assert "sk-synthetic-claimed-happy" not in _content_text(sanitized)
    assert engine.should_compress_preflight(deepcopy(sanitized)) is False
    assert (
        engine.prepare_compression_operation(
            deepcopy(messages),
            session_id=engine.bound_session_id,
            attempt_generation=17,
        )
        is None
    )


@pytest.mark.parametrize(
    ("session_id", "attempt_generation"),
    [
        ("different-host-session", 7),
        (None, 7),
        ("bound", 6),
        ("bound", None),
        ("bound", True),
    ],
)
def test_prepare_rejects_wrong_host_session_or_attempt(
    tmp_path,
    session_id,
    attempt_generation,
):
    engine = _engine(tmp_path, f"invalid-host-binding-{session_id}-{attempt_generation}")
    messages = [
        {
            "role": "user",
            "content": "api_key=sk-synthetic-invalid-host-binding-000000000",
        }
    ]
    assert engine.should_compress_preflight(deepcopy(messages)) is True
    engine._compression_attempt_generation = 7
    if session_id == "bound":
        session_id = engine.bound_session_id

    assert (
        engine.prepare_compression_operation(
            deepcopy(messages),
            session_id=session_id,
            attempt_generation=attempt_generation,
        )
        is None
    )


def test_second_prepare_consumes_handoff_and_invalidates_first_claim(tmp_path):
    engine = _engine(tmp_path, "claim-second-prepare")
    messages = [
        {
            "role": "user",
            "content": "api_key=sk-synthetic-second-prepare-000000000000",
        }
    ]
    assert engine.should_compress_preflight(deepcopy(messages)) is True
    claim = _claim_sanitation(engine, messages, generation=8)

    assert (
        engine.prepare_compression_operation(
            deepcopy(messages),
            session_id=engine.bound_session_id,
            attempt_generation=8,
        )
        is None
    )
    assert isinstance(
        engine.compress(deepcopy(messages), operation_claim=claim),
        list,
    )


def test_equal_but_nonidentical_claim_is_rejected_and_consumes_real_claim(tmp_path):
    class EqualClaim:
        def __eq__(self, _other):
            return True

    engine = _engine(tmp_path, "claim-identity")
    messages = [
        {
            "role": "user",
            "content": "api_key=sk-synthetic-claim-identity-0000000000000",
        }
    ]
    assert engine.should_compress_preflight(deepcopy(messages)) is True
    claim = _claim_sanitation(engine, messages, generation=9)

    assert isinstance(
        engine.compress(deepcopy(messages), operation_claim=EqualClaim()),
        list,
    )
    assert isinstance(
        engine.compress(deepcopy(messages), operation_claim=claim),
        list,
    )


def test_attempt_generation_advance_invalidates_claim(tmp_path):
    engine = _engine(tmp_path, "claim-attempt-advance")
    messages = [
        {
            "role": "user",
            "content": "api_key=sk-synthetic-attempt-advance-00000000000",
        }
    ]
    assert engine.should_compress_preflight(deepcopy(messages)) is True
    claim = _claim_sanitation(engine, messages, generation=10)
    engine._compression_attempt_generation = 11

    assert isinstance(
        engine.compress(deepcopy(messages), operation_claim=claim),
        list,
    )


def test_intervening_preflight_invalidates_claim(tmp_path):
    engine = _engine(tmp_path, "claim-intervening-preflight")
    engine.threshold_tokens = 90_000
    messages = [
        {
            "role": "user",
            "content": "api_key=sk-synthetic-intervening-preflight-0000000",
        },
        {"role": "assistant", "content": "fresh answer"},
    ]
    assert engine.should_compress_preflight(deepcopy(messages)) is True
    claim = _claim_sanitation(engine, messages)

    assert engine.should_compress_preflight(
        [{"role": "user", "content": "intervening benign input"}]
    ) is False
    result = engine.compress(deepcopy(messages), operation_claim=claim)

    assert isinstance(result, list)


def test_session_change_and_rebind_invalidates_claim(tmp_path):
    engine = _engine(tmp_path, "claim-session-rebind")
    engine.threshold_tokens = 90_000
    messages = [
        {
            "role": "user",
            "content": "api_key=sk-synthetic-session-rebind-000000000000",
        },
        {"role": "assistant", "content": "fresh answer"},
    ]
    assert engine.should_compress_preflight(deepcopy(messages)) is True
    claim = _claim_sanitation(engine, messages, generation=2)

    engine.on_session_start(
        "temporary-session",
        platform="synthetic",
        conversation_id="temporary-conversation",
        context_length=100_000,
    )
    engine.on_session_start(
        "claim-session-rebind-session",
        platform="synthetic",
        conversation_id="claim-session-rebind-conversation",
        context_length=100_000,
    )
    result = engine.compress(deepcopy(messages), operation_claim=claim)

    assert isinstance(result, list)


def test_session_reset_invalidates_claim(tmp_path):
    engine = _engine(
        tmp_path,
        "claim-session-reset",
        new_session_retain_depth=-1,
    )
    engine.threshold_tokens = 90_000
    messages = [
        {
            "role": "user",
            "content": "api_key=sk-synthetic-session-reset-0000000000000",
        },
        {"role": "assistant", "content": "fresh answer"},
    ]
    assert engine.should_compress_preflight(deepcopy(messages)) is True
    claim = _claim_sanitation(engine, messages, generation=3)

    engine.on_session_reset()
    result = engine.compress(deepcopy(messages), operation_claim=claim)

    assert isinstance(result, list)


def test_storage_rebind_invalidates_claim(tmp_path):
    engine = _engine(tmp_path, "claim-storage-rebind")
    messages = [
        {
            "role": "user",
            "content": "api_key=sk-synthetic-storage-rebind-000000000000",
        }
    ]
    assert engine.should_compress_preflight(deepcopy(messages)) is True
    claim = _claim_sanitation(engine, messages, generation=4)

    assert engine._rebind_storage_for_home(str(tmp_path / "different-home")) is True
    assert isinstance(
        engine.compress(deepcopy(messages), operation_claim=claim),
        list,
    )


def test_claim_is_consumed_when_compression_raises(tmp_path, monkeypatch):
    engine = _engine(tmp_path, "claim-exception")
    messages = [
        {
            "role": "user",
            "content": "api_key=sk-synthetic-claim-exception-0000000000000",
        }
    ]
    assert engine.should_compress_preflight(deepcopy(messages)) is True
    claim = _claim_sanitation(engine, messages, generation=5)
    monkeypatch.setattr(
        engine,
        "_ingest_messages",
        Mock(side_effect=RuntimeError("synthetic claimed failure")),
    )

    with pytest.raises(RuntimeError, match="synthetic claimed failure"):
        engine.compress(deepcopy(messages), operation_claim=claim)
    assert engine._pending_sanitation_claim is None


@pytest.mark.parametrize("mode", ["force", "overflow"])
def test_force_and_overflow_never_echo_sanitation_claim(
    tmp_path,
    monkeypatch,
    mode,
):
    engine = _engine(tmp_path, f"claim-{mode}", fresh_tail_count=1)
    messages = [
        {"role": "user", "content": "eligible backlog " * 20},
        {
            "role": "assistant",
            "content": f"api_key=sk-synthetic-claim-{mode}-00000000000000",
        },
        {"role": "user", "content": "fresh"},
    ]
    assert engine.should_compress_preflight(deepcopy(messages)) is True
    claim = _claim_sanitation(engine, messages, generation=6)
    if mode == "overflow":
        monkeypatch.setattr(
            engine,
            "_should_force_overflow_recovery",
            lambda **_kwargs: True,
        )
    monkeypatch.setattr(
        lcm_engine,
        "summarize_with_escalation",
        Mock(return_value=("summary", 1)),
    )

    result = engine.compress(
        deepcopy(messages),
        current_tokens=count_messages_tokens(messages),
        force=mode == "force",
        operation_claim=claim,
    )

    assert isinstance(result, list)


def test_generic_compression_never_echoes_unrecognized_claim(tmp_path):
    engine = _engine(
        tmp_path,
        "generic-no-claim-echo",
        sensitive_patterns_enabled=False,
    )
    messages = [{"role": "user", "content": "ordinary below-threshold input"}]

    result = engine.compress(messages, operation_claim=object())

    assert isinstance(result, list)


def test_direct_exact_preflight_consumes_sanitation_without_claim_echo(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path, "direct-exact")
    engine.threshold_tokens = 90_000
    messages = [
        {"role": "user", "content": "old eligible backlog " * 20},
        {
            "role": "assistant",
            "content": "api_key=sk-synthetic-direct-exact-000000000000000",
        },
        {"role": "user", "content": "fresh"},
    ]
    monkeypatch.setattr(
        lcm_engine,
        "summarize_with_escalation",
        Mock(side_effect=AssertionError("direct sanitation must not summarize")),
    )

    assert engine.should_compress_preflight(deepcopy(messages)) is True
    result = engine.compress(
        deepcopy(messages),
        current_tokens=count_messages_tokens(messages),
    )

    assert isinstance(result, list)
    assert engine.last_compression_status == "sanitized"
    assert engine._dag.get_session_node_count(engine.current_session_id) == 0
    assert "sk-synthetic-direct-exact" not in _content_text(result)


def test_direct_handoff_message_mismatch_remains_generic(tmp_path, monkeypatch):
    engine = _engine(tmp_path, "direct-message-mismatch", fresh_tail_count=1)
    engine.threshold_tokens = 90_000
    cleanup_messages = [
        {
            "role": "user",
            "content": "api_key=sk-synthetic-direct-mismatch-000000000000",
        },
        {"role": "assistant", "content": "fresh cleanup answer"},
    ]
    unrelated_messages = [
        {"role": "user", "content": "unrelated eligible backlog " * 20},
        {"role": "assistant", "content": "unrelated eligible answer"},
        {"role": "user", "content": "fresh"},
    ]
    summary_spy = Mock(return_value=("generic mismatch summary", 1))
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summary_spy)

    assert engine.should_compress_preflight(deepcopy(cleanup_messages)) is True
    result = engine.compress(
        deepcopy(unrelated_messages),
        current_tokens=count_messages_tokens(unrelated_messages),
    )

    assert isinstance(result, list)
    assert engine.last_compression_status == "compacted"
    summary_spy.assert_called()


def test_direct_handoff_session_change_remains_generic(tmp_path, monkeypatch):
    engine = _engine(tmp_path, "direct-session-change", fresh_tail_count=1)
    engine.threshold_tokens = 90_000
    cleanup_messages = [
        {
            "role": "user",
            "content": "api_key=sk-synthetic-direct-session-0000000000000",
        }
    ]
    next_session_messages = [
        {"role": "user", "content": "next-session eligible backlog " * 20},
        {"role": "assistant", "content": "next-session eligible answer"},
        {"role": "user", "content": "fresh"},
    ]
    summary_spy = Mock(return_value=("next-session summary", 1))
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summary_spy)

    assert engine.should_compress_preflight(deepcopy(cleanup_messages)) is True
    engine.on_session_start(
        "direct-session-change-next",
        platform="synthetic",
        conversation_id="direct-session-change-next-conversation",
        context_length=100_000,
    )
    result = engine.compress(
        deepcopy(next_session_messages),
        current_tokens=count_messages_tokens(next_session_messages),
    )

    assert isinstance(result, list)
    assert engine.last_compression_status == "compacted"
    summary_spy.assert_called()


@pytest.mark.parametrize("mode", ["threshold", "manual", "overflow"])
def test_direct_handoff_compaction_trigger_remains_generic(
    tmp_path,
    monkeypatch,
    mode,
):
    engine = _engine(
        tmp_path,
        f"direct-{mode}",
        fresh_tail_count=1,
        threshold_full_sweep_enabled=False,
    )
    engine.threshold_tokens = 90_000
    messages = [
        {"role": "user", "content": f"{mode} eligible backlog " * 20},
        {
            "role": "assistant",
            "content": f"api_key=sk-synthetic-direct-{mode}-000000000000000",
        },
        {"role": "user", "content": "fresh"},
    ]
    rough = count_messages_tokens(messages)
    summary_spy = Mock(return_value=(f"{mode} summary", 1))
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summary_spy)

    assert engine.should_compress_preflight(deepcopy(messages)) is True
    if mode == "threshold":
        engine.threshold_tokens = rough
    elif mode == "overflow":
        monkeypatch.setattr(
            engine,
            "_should_force_overflow_recovery",
            lambda **_kwargs: True,
        )
    result = engine.compress(
        deepcopy(messages),
        current_tokens=rough,
        force=mode == "manual",
    )

    assert isinstance(result, list)
    assert engine.last_compression_status == "compacted"
    summary_spy.assert_called()


def test_below_floor_replay_cleanup_is_pure_sanitation(tmp_path, monkeypatch):
    engine = _engine(tmp_path, "below-floor")
    engine.threshold_tokens = 90_000
    messages = [
        {"role": "system", "content": "stable system"},
        {"role": "user", "content": "eligible old request " * 20},
        {"role": "assistant", "content": "eligible old answer " * 20},
        {
            "role": "user",
            "content": "use api_key=sk-synthetic-subthreshold-0000000000000000",
        },
        {"role": "assistant", "content": "fresh answer"},
    ]
    summary_spy = Mock(side_effect=AssertionError("sanitation must not summarize"))
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summary_spy)

    assert count_messages_tokens(messages) < engine.threshold_tokens
    assert engine.should_compress_preflight(deepcopy(messages)) is True
    claim = _claim_sanitation(engine, messages, generation=12)
    sanitized, returned_claim = engine.compress(
        deepcopy(messages),
        current_tokens=count_messages_tokens(messages),
        operation_claim=claim,
    )

    assert returned_claim is claim
    assert engine.last_compression_status == "sanitized"
    assert engine._dag.get_session_node_count(engine.current_session_id) == 0
    assert "sk-synthetic-subthreshold" not in _content_text(sanitized)
    assert any("eligible old request" in str(message.get("content")) for message in sanitized)
    summary_spy.assert_not_called()


def test_no_leaf_scaffold_reassembly_has_distinct_status(tmp_path, monkeypatch):
    engine = _engine(
        tmp_path,
        "reassembled-status",
        fresh_tail_count=2,
        leaf_chunk_tokens=10,
        sensitive_patterns_enabled=False,
    )
    summary_text = "backed compressed details\nExpand for details about: prior"
    node_id = engine._dag.add_node(
        SummaryNode(
            session_id=engine.current_session_id,
            depth=0,
            summary=summary_text,
            token_count=3,
            source_token_count=50,
            source_ids=[1],
            source_type="messages",
            created_at=time.time(),
            earliest_at=time.time(),
            latest_at=time.time(),
            expand_hint="prior",
        )
    )
    engine._ingest_cursor = 3
    engine._ingest_cursor_needs_reconcile = False
    monkeypatch.setattr(
        lcm_engine,
        "summarize_with_escalation",
        Mock(side_effect=AssertionError("reassembly must not summarize")),
    )
    messages = [
        {
            "role": "user",
            "content": (
                f"[Recent Summary (d0, node {node_id})]\n"
                f"{summary_text}\n[Expand for details: prior]"
            ),
        },
        {"role": "user", "content": "fresh question"},
        {"role": "assistant", "content": "fresh answer"},
    ]

    claim = object()
    assert engine.compress(messages, operation_claim=claim) == messages
    assert engine.last_compression_status == "reassembled"


def test_ignored_backlog_below_effective_floor_is_deferred(tmp_path, monkeypatch):
    engine = _engine(tmp_path, "ignored-floor", sensitive_patterns_enabled=False)
    engine.threshold_tokens = 90_000
    original = [{"role": "user", "content": "provider-visible normalization source"}]
    replay = [{"role": "user", "content": "benign normalized source"}]
    monkeypatch.setattr(engine, "_ingest_messages", lambda _messages: replay)
    monkeypatch.setattr(
        engine,
        "_leaf_compaction_candidate_status",
        lambda *_args, **_kwargs: (False, "no eligible leaf"),
    )
    monkeypatch.setattr(engine, "_has_ignored_backlog_outside_fresh_tail", lambda _messages: True)

    assert engine.should_compress_preflight(original) is False


def test_effective_floor_allows_only_threshold_or_critical_pressure(tmp_path):
    engine = _engine(
        tmp_path,
        "effective-floor",
        fresh_tail_count=1,
        leaf_chunk_tokens=1,
        critical_budget_pressure_ratio=0.8,
        sensitive_patterns_enabled=False,
    )
    engine.context_length = 1_000
    engine.threshold_tokens = 900
    below_floor = [
        {"role": "user", "content": "eligible backlog " * 20},
        {"role": "assistant", "content": "eligible answer"},
        {"role": "user", "content": "fresh"},
    ]
    below_tokens = count_messages_tokens(below_floor)
    assert below_tokens < 800
    assert engine.should_compress_preflight(below_floor) is False

    critical_engine = _engine(
        tmp_path,
        "critical-floor",
        fresh_tail_count=1,
        leaf_chunk_tokens=1,
        critical_budget_pressure_ratio=0.8,
        sensitive_patterns_enabled=False,
    )
    critical_engine.context_length = 1_000
    critical_engine.threshold_tokens = 900
    pressure = "pressure "
    while True:
        above_floor = [
            {"role": "user", "content": pressure},
            {"role": "assistant", "content": "eligible answer"},
            {"role": "user", "content": "fresh"},
        ]
        above_tokens = count_messages_tokens(above_floor)
        if above_tokens >= 800:
            break
        pressure += "pressure " * 20
    assert above_tokens < engine.threshold_tokens
    assert critical_engine.should_compress_preflight(above_floor) is True


def test_replay_cleanup_does_not_swallow_critical_leaf_compaction(
    tmp_path,
    monkeypatch,
):
    engine = _engine(
        tmp_path,
        "cleanup-critical-leaf",
        fresh_tail_count=1,
        leaf_chunk_tokens=1,
        critical_budget_pressure_ratio=0.8,
    )
    engine.context_length = 100
    engine.threshold_tokens = 90_000
    messages = [
        {"role": "user", "content": "eligible critical backlog " * 30},
        {"role": "assistant", "content": "eligible critical answer"},
        {
            "role": "user",
            "content": "api_key=sk-synthetic-critical-cleanup-000000000000",
        },
    ]
    summary_spy = Mock(return_value=("critical summary", 1))
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summary_spy)

    rough = count_messages_tokens(messages)
    assert engine._critical_budget_pressure_reached(
        observed_tokens=rough,
        messages=messages,
    )
    assert engine.should_compress_preflight(deepcopy(messages)) is True
    assert engine._preflight_cleanup_only is False
    engine.compress(deepcopy(messages), current_tokens=rough)

    assert engine.last_compression_status == "compacted"
    summary_spy.assert_called()


@pytest.mark.parametrize("caller", ["host_claim", "direct"])
def test_sanitation_shrink_preserves_original_critical_pressure_operation(
    tmp_path,
    monkeypatch,
    caller,
):
    engine = _engine(
        tmp_path,
        f"cleanup-critical-shrink-{caller}",
        fresh_tail_count=1,
        leaf_chunk_tokens=1,
        critical_budget_pressure_ratio=0.8,
    )
    engine.context_length = 1_000
    engine.threshold_tokens = 90_000
    messages = [
        {"role": "user", "content": "eligible old request"},
        {"role": "assistant", "content": "eligible old answer"},
        {
            "role": "user",
            "content": f"api_key=sk-synthetic-critical-shrink-{'x' * 4_000}",
        },
    ]
    rough = count_messages_tokens(messages)
    critical_floor = int(
        engine.context_length * engine._config.critical_budget_pressure_ratio
    )
    replays = []
    real_ingest = engine._ingest_messages

    def capture_replay(candidate_messages):
        replay = real_ingest(candidate_messages)
        replays.append(deepcopy(replay))
        return replay

    monkeypatch.setattr(engine, "_ingest_messages", capture_replay)
    summary_spy = Mock(return_value=("critical shrink summary", 1))
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summary_spy)

    assert rough >= critical_floor
    assert engine.should_compress_preflight(deepcopy(messages)) is True
    assert count_messages_tokens(replays[0]) < critical_floor
    assert engine._preflight_cleanup_only is False

    operation_claim = None
    prepared = None
    if caller == "host_claim":
        engine._compression_attempt_generation = 23
        prepared = engine.prepare_compression_operation(
            deepcopy(messages),
            session_id=engine.bound_session_id,
            attempt_generation=23,
        )
        if prepared is not None:
            _operation, operation_claim = prepared

    result = engine.compress(
        deepcopy(messages),
        current_tokens=rough,
        operation_claim=operation_claim,
    )

    assert prepared is None
    assert isinstance(result, list)
    assert engine.last_compression_status == "compacted"
    summary_spy.assert_called()


def test_cooldown_preserves_unchanged_replay_critical_leaf_work(tmp_path):
    engine = _engine(
        tmp_path,
        "cooldown-critical-leaf",
        fresh_tail_count=1,
        leaf_chunk_tokens=1,
        critical_budget_pressure_ratio=0.8,
        sensitive_patterns_enabled=False,
    )
    engine.context_length = 100
    engine.threshold_tokens = 90_000
    engine._last_boundary_skip_time = time.time()
    messages = [
        {"role": "user", "content": "critical eligible backlog " * 30},
        {"role": "assistant", "content": "critical eligible answer"},
        {"role": "user", "content": "fresh question"},
    ]
    rough = count_messages_tokens(messages)

    assert engine._critical_budget_pressure_reached(
        observed_tokens=rough,
        messages=messages,
    )
    assert engine.should_compress_preflight(deepcopy(messages)) is True


def test_cooldown_preserves_unchanged_replay_critical_ignored_backlog(
    tmp_path,
    monkeypatch,
):
    engine = _engine(
        tmp_path,
        "cooldown-critical-ignored",
        critical_budget_pressure_ratio=0.8,
        sensitive_patterns_enabled=False,
    )
    engine.context_length = 10
    engine.threshold_tokens = 90_000
    engine._last_boundary_skip_time = time.time()
    messages = [{"role": "user", "content": "critical ignored backlog"}]
    monkeypatch.setattr(
        engine,
        "_leaf_compaction_candidate_status",
        lambda *_args, **_kwargs: (False, "no eligible leaf"),
    )
    monkeypatch.setattr(
        engine,
        "_has_ignored_backlog_outside_fresh_tail",
        lambda _messages: True,
    )

    assert engine.should_compress_preflight(deepcopy(messages)) is True


def test_cooldown_preserves_unchanged_replay_critical_deferred_maintenance(
    tmp_path,
    monkeypatch,
):
    engine = _engine(
        tmp_path,
        "cooldown-critical-deferred",
        critical_budget_pressure_ratio=0.8,
        sensitive_patterns_enabled=False,
    )
    engine.context_length = 10
    engine.threshold_tokens = 90_000
    engine._last_boundary_skip_time = time.time()
    messages = [{"role": "user", "content": "critical deferred backlog"}]
    monkeypatch.setattr(
        engine,
        "_leaf_compaction_candidate_status",
        lambda *_args, **_kwargs: (False, "no eligible leaf"),
    )
    monkeypatch.setattr(
        engine,
        "_has_ignored_backlog_outside_fresh_tail",
        lambda _messages: False,
    )
    monkeypatch.setattr(
        engine,
        "_should_run_deferred_maintenance",
        lambda *_args, **_kwargs: True,
    )

    assert engine.should_compress_preflight(deepcopy(messages)) is True


def test_cleanup_state_is_cleared_before_fallible_ingest_and_cannot_hijack_force(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path, "stale-state")
    engine.threshold_tokens = 90_000
    messages = [
        {"role": "user", "content": "old eligible backlog"},
        {
            "role": "assistant",
            "content": "api_key=sk-synthetic-stale-state-0000000000000000",
        },
        {"role": "user", "content": "fresh"},
    ]
    assert engine.should_compress_preflight(deepcopy(messages)) is True
    assert engine._preflight_cleanup_only is True

    real_ingest = engine._ingest_messages
    monkeypatch.setattr(
        engine,
        "_ingest_messages",
        Mock(side_effect=RuntimeError("synthetic ingest failure")),
    )
    with pytest.raises(RuntimeError, match="synthetic ingest failure"):
        engine.compress(deepcopy(messages), current_tokens=100)
    assert engine._preflight_cleanup_only is False

    summary_spy = Mock(return_value=("manual summary", 1))
    monkeypatch.setattr(engine, "_ingest_messages", real_ingest)
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summary_spy)
    engine.compress(deepcopy(messages), current_tokens=100, force=True)

    assert engine.last_compression_status == "compacted"
    summary_spy.assert_called()


def test_cleanup_handoff_cannot_sanitize_an_interleaved_message_set(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path, "handoff-cross-message")
    engine.threshold_tokens = 90_000
    cleanup_messages = [
        {
            "role": "user",
            "content": "api_key=sk-synthetic-cross-message-0000000000000000",
        },
        {"role": "assistant", "content": "fresh cleanup answer"},
    ]
    unrelated_messages = [
        {"role": "user", "content": "unrelated eligible backlog"},
        {"role": "assistant", "content": "unrelated eligible answer"},
        {"role": "user", "content": "unrelated fresh question"},
    ]
    summary_spy = Mock(return_value=("unrelated summary", 1))
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summary_spy)

    assert engine.should_compress_preflight(deepcopy(cleanup_messages)) is True
    assert engine._preflight_cleanup_only is True
    claim = _claim_sanitation(engine, cleanup_messages, generation=20)
    result = engine.compress(
        deepcopy(unrelated_messages),
        current_tokens=count_messages_tokens(unrelated_messages),
        operation_claim=claim,
    )

    assert isinstance(result, list)
    assert engine.last_compression_status == "compacted"
    summary_spy.assert_called()


def test_cleanup_handoff_cannot_cross_session_boundaries(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path, "handoff-cross-session")
    engine.threshold_tokens = 90_000
    cleanup_messages = [
        {
            "role": "user",
            "content": "api_key=sk-synthetic-cross-session-0000000000000000",
        },
        {"role": "assistant", "content": "fresh cleanup answer"},
    ]
    next_session_messages = [
        {"role": "user", "content": "next-session eligible backlog"},
        {"role": "assistant", "content": "next-session eligible answer"},
        {"role": "user", "content": "next-session fresh question"},
    ]
    summary_spy = Mock(return_value=("next-session summary", 1))
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summary_spy)

    assert engine.should_compress_preflight(deepcopy(cleanup_messages)) is True
    assert engine._preflight_cleanup_only is True
    claim = _claim_sanitation(engine, cleanup_messages, generation=21)
    engine.on_session_start(
        "handoff-cross-session-second-session",
        platform="synthetic",
        conversation_id="handoff-cross-session-second-conversation",
        context_length=100_000,
    )
    result = engine.compress(
        deepcopy(next_session_messages),
        current_tokens=count_messages_tokens(next_session_messages),
        operation_claim=claim,
    )

    assert isinstance(result, list)
    assert engine.last_compression_status == "compacted"
    summary_spy.assert_called()


def test_cleanup_handoff_cannot_sanitize_an_interleaved_threshold_call(
    tmp_path,
    monkeypatch,
):
    engine = _engine(
        tmp_path,
        "handoff-threshold",
        threshold_full_sweep_enabled=False,
    )
    engine.threshold_tokens = 90_000
    cleanup_messages = [
        {
            "role": "user",
            "content": "api_key=sk-synthetic-threshold-handoff-000000000000",
        },
        {"role": "assistant", "content": "fresh cleanup answer"},
    ]
    threshold_messages = [
        {"role": "user", "content": "threshold eligible backlog"},
        {"role": "assistant", "content": "threshold eligible answer"},
        {"role": "user", "content": "threshold fresh question"},
    ]
    threshold_tokens = count_messages_tokens(threshold_messages)
    engine.threshold_tokens = threshold_tokens
    summary_spy = Mock(return_value=("threshold summary", 1))
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summary_spy)

    engine.threshold_tokens = 90_000
    assert engine.should_compress_preflight(deepcopy(cleanup_messages)) is True
    assert engine._preflight_cleanup_only is True
    claim = _claim_sanitation(engine, cleanup_messages, generation=22)
    engine.threshold_tokens = threshold_tokens
    result = engine.compress(
        deepcopy(threshold_messages),
        current_tokens=threshold_tokens,
        operation_claim=claim,
    )

    assert isinstance(result, list)
    assert engine.last_compression_status == "compacted"
    summary_spy.assert_called()


def test_full_sanitation_round_trip_converges_without_leakage(
    tmp_path,
    monkeypatch,
    caplog,
):
    name = "round-trip"
    engine = _engine(
        tmp_path,
        name,
        fresh_tail_count=2,
        leaf_chunk_tokens=20_000,
        large_output_externalization_enabled=True,
        large_output_externalization_threshold_chars=120,
    )
    engine.threshold_tokens = 90_000
    raw_values = [
        "sk-synthetic-scalar-0000000000000000",
        "sk-synthetic-list-000000000000000000",
        "sk-synthetic-dict-000000000000000000",
        "sk-synthetic-key-0000000000000000000",
        "sk-synthetic-json-000000000000000000",
        "sk-synthetic-toolcall-00000000000000",
        "tiny77",
        "quoted synthetic passphrase",
        "sk-synthetic-adjacent-00000000000000",
        "sk-synthetic-second-adjacent-00000000",
    ]
    valid_quoted_placeholder = (
        "[LCM sensitive redaction: name=api_key; chars=32; bytes=32; "
        "sha256=0123456789abcdef]"
    )
    malformed_quoted_placeholder = (
        "[LCM sensitive redaction: name=api_key; chars=bogus; bytes=32]"
    )
    messages = [
        {"role": "system", "content": "stable system"},
        {"role": "user", "content": "old eligible production-shaped request " * 20},
        {"role": "assistant", "content": "old eligible production-shaped response " * 20},
        {
            "role": "user",
            "content": (
                f"api_key={raw_values[0]} password={raw_values[6]} "
                f'password="{raw_values[7]}" api_key={raw_values[8]} '
                f"and api_key={raw_values[9]} "
                f'quoted examples "{valid_quoted_placeholder}" '
                f'and "{malformed_quoted_placeholder}"'
            ),
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": f"api_key={raw_values[1]}"},
                {"type": "metadata", "value": {"api_key": raw_values[2]}},
                {f"api_key={raw_values[3]}": "synthetic-key-value"},
            ],
            "tool_calls": [
                {
                    "id": "call_synthetic",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": json.dumps({"api_key": raw_values[5]}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_synthetic",
            "content": (
                json.dumps({"api_key": raw_values[4]})
                + " externalized filler "
                + ("safe-filler " * 30)
            ),
        },
        {"role": "user", "content": "fresh follow-up"},
    ]
    monkeypatch.setattr(
        lcm_engine,
        "summarize_with_escalation",
        Mock(side_effect=AssertionError("sanitation must not summarize")),
    )

    with caplog.at_level(logging.INFO):
        assert engine.should_compress_preflight(deepcopy(messages)) is True
        assert engine._preflight_cleanup_only is True
        claim = _claim_sanitation(engine, messages, generation=13)
        adopted, returned_claim = engine.compress(
            deepcopy(messages),
            current_tokens=count_messages_tokens(messages),
            operation_claim=claim,
        )
    assert returned_claim is claim
    assert engine.last_compression_status == "sanitized"
    assert engine._dag.get_session_node_count(engine.current_session_id) == 0
    adopted_tool_result = next(
        message for message in adopted if message.get("role") == "tool"
    )
    assert "[LCM sensitive redaction:" in str(adopted_tool_result.get("content"))
    durable_tool_content = engine._store._conn.execute(
        "SELECT content FROM messages WHERE role = 'tool' ORDER BY store_id DESC LIMIT 1"
    ).fetchone()[0]
    assert durable_tool_content.startswith("[Externalized tool output:")
    assert durable_tool_content != adopted_tool_result.get("content")
    engine.shutdown()

    restarted = _engine(
        tmp_path,
        name,
        fresh_tail_count=2,
        leaf_chunk_tokens=20_000,
        large_output_externalization_enabled=True,
        large_output_externalization_threshold_chars=120,
    )
    restarted.threshold_tokens = 90_000
    assert restarted.should_compress_preflight(deepcopy(adopted)) is False
    restarted.shutdown()

    replay_probe = _engine(
        tmp_path,
        name,
        fresh_tail_count=2,
        leaf_chunk_tokens=20_000,
        large_output_externalization_enabled=True,
        large_output_externalization_threshold_chars=120,
    )
    replay_probe.threshold_tokens = 90_000
    second_replay = replay_probe._ingest_messages(deepcopy(adopted))

    adopted_bytes = json.dumps(
        adopted,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    second_replay_bytes = json.dumps(
        second_replay,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    assert second_replay_bytes == adopted_bytes
    third_replay_bytes = json.dumps(
        replay_probe._ingest_messages(deepcopy(adopted)),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    assert third_replay_bytes == adopted_bytes
    assert replay_probe._dag.get_session_node_count(replay_probe.current_session_id) == 0

    stored = "\n".join(
        "\n".join((str(row[0] or ""), str(row[1] or "")))
        for row in replay_probe._store._conn.execute(
            "SELECT content, tool_calls FROM messages"
        ).fetchall()
    )
    externalized = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (tmp_path / f"{name}-externalized").glob("*.json")
    )
    observed_logs = caplog.text
    for raw in raw_values:
        assert raw not in stored
        assert raw not in externalized
        assert raw not in observed_logs
        assert replay_probe._store.search(
            raw,
            session_id=replay_probe.current_session_id,
        ) == []


def test_positive_preflight_and_manual_force_logs_are_reason_coded_and_content_free(
    tmp_path,
    monkeypatch,
    caplog,
):
    engine = _engine(tmp_path, "reason-log")
    engine.threshold_tokens = 90_000
    marker = "sk-synthetic-log-marker-0000000000000000"
    messages = [
        {"role": "user", "content": f"api_key={marker}"},
        {"role": "assistant", "content": "fresh"},
    ]

    with caplog.at_level(logging.INFO):
        assert engine.should_compress_preflight(deepcopy(messages)) is True
    decisions = [
        record.getMessage()
        for record in caplog.records
        if "LCM preflight decision" in record.getMessage()
    ]
    assert decisions == [
        "LCM preflight decision operation=sanitize reason=redaction_sanitation"
    ]
    assert marker not in caplog.text

    caplog.clear()
    summary_spy = Mock(return_value=("manual summary", 1))
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summary_spy)
    with caplog.at_level(logging.INFO):
        engine.compress(
            [
                {"role": "user", "content": "manual eligible backlog"},
                {"role": "assistant", "content": "fresh"},
            ],
            current_tokens=100,
            force=True,
        )
    assert "operation=compact reason=manual_force" in caplog.text
