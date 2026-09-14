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
    sanitized = engine.compress(deepcopy(messages), current_tokens=count_messages_tokens(messages))

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

    assert engine.compress(messages) == messages
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
    engine.compress(
        deepcopy(unrelated_messages),
        current_tokens=count_messages_tokens(unrelated_messages),
    )

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
    engine.on_session_start(
        "handoff-cross-session-second-session",
        platform="synthetic",
        conversation_id="handoff-cross-session-second-conversation",
        context_length=100_000,
    )
    engine.compress(
        deepcopy(next_session_messages),
        current_tokens=count_messages_tokens(next_session_messages),
    )

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
    engine.threshold_tokens = threshold_tokens
    engine.compress(deepcopy(threshold_messages), current_tokens=threshold_tokens)

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
        adopted = engine.compress(
            deepcopy(messages),
            current_tokens=count_messages_tokens(messages),
        )
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
