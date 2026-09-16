"""Issue #72 invariants for terminal, subthreshold replay sanitation."""

from __future__ import annotations

import json
import logging
import threading
import time
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import pytest

import hermes_lcm.engine as lcm_engine
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.externalize import _externalized_summary
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


def test_should_compress_overflow_precedes_cooldown_for_public_gates(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path, "overflow-before-cooldown")
    engine.threshold_tokens = 100
    engine._last_boundary_skip_time = time.time()
    monkeypatch.setattr(
        engine,
        "_should_force_overflow_recovery",
        lambda **kwargs: kwargs["observed_tokens"] >= 1_000,
    )

    assert engine.should_compress(1_000) is True
    assert engine.should_compress(100) is False

    monkeypatch.setattr(engine, "_bypasses_lcm_context_management", lambda: True)
    assert engine.should_compress(1_000) is True
    assert engine.should_compress(100) is False


def test_overflow_is_recomputed_from_expanded_sanitized_replay(tmp_path):
    engine = _engine(tmp_path, "overflow-after-replay-sanitation")
    engine.threshold_tokens = 90_000
    messages = [{"role": "user", "content": "api_key=abcdefghijkl"}]
    sanitized = engine._redact_active_replay_messages(messages)
    original_tokens = count_messages_tokens(messages)
    sanitized_tokens = count_messages_tokens(sanitized)
    assert sanitized_tokens > original_tokens
    engine._config.max_assembly_tokens = original_tokens + 1
    assert engine._config.max_assembly_tokens <= sanitized_tokens

    assert engine.should_compress_preflight(deepcopy(messages)) is True
    result = engine.compress(deepcopy(messages))

    assert result == sanitized
    assert engine.last_compression_status == "overflow_recovery"
    assert engine._last_overflow_recovery_failed is True


def test_session_end_invalidates_only_bound_foreground_claim(tmp_path):
    engine = _engine(tmp_path, "session-end-claim")
    messages = [
        {"role": "user", "content": "api_key=sk-session-end-secret"},
        {"role": "user", "content": "fresh"},
    ]
    assert engine.should_compress_preflight(deepcopy(messages)) is True
    claim = _claim_sanitation(engine, messages)

    engine.on_session_end(engine.bound_session_id, deepcopy(messages))
    result = engine.compress(
        deepcopy(messages),
        operation_claim=claim,
        current_tokens=count_messages_tokens(messages),
    )
    assert not (isinstance(result, tuple) and result[1] is claim)


def test_handoff_identity_includes_provider_visible_name_metadata(tmp_path):
    engine = _engine(tmp_path, "handoff-name-identity")
    base = [
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "name": "lookup",
            "tool_name": "stored_lookup",
            "content": "result",
        }
    ]
    changed = deepcopy(base)
    changed[0]["name"] = "admin_lookup"
    changed_tool_name = deepcopy(base)
    changed_tool_name[0]["tool_name"] = "stored_admin_lookup"

    assert engine._cleanup_handoff_message_identity(base) != engine._cleanup_handoff_message_identity(changed)
    assert engine._cleanup_handoff_message_identity(base) != engine._cleanup_handoff_message_identity(
        changed_tool_name
    )


def test_engine_sidecar_loader_uses_configured_storage_and_rejects_traversal(tmp_path):
    engine = _engine(tmp_path, "sidecar-loader")
    try:
        storage = tmp_path / "sidecar-loader-externalized"
        storage.mkdir(parents=True, exist_ok=True)
        content = "durable tool output"
        ref = "payload.json"
        (storage / ref).write_text(
            json.dumps(
                {
                    "kind": "tool_result",
                    "tool_call_id": "call-sidecar",
                    "content": content,
                    "content_chars": len(content),
                    "content_bytes": len(content.encode("utf-8")),
                }
            ),
            encoding="utf-8",
        )

        loaded = engine.load_externalized_payload_sidecar(ref)

        assert loaded is not None
        assert loaded["kind"] == "tool_result"
        assert loaded["tool_call_id"] == "call-sidecar"
        assert loaded["content"] == content
        assert engine.load_externalized_payload_sidecar("../payload.json") is None
        assert engine.load_externalized_payload_sidecar("missing.json") is None
        (storage / "non-object.json").write_text("[]", encoding="utf-8")
        assert engine.load_externalized_payload_sidecar("non-object.json") is None
    finally:
        engine.shutdown()


def _content_text(messages) -> str:
    return json.dumps(messages, ensure_ascii=False, sort_keys=True)


def _claim_sanitation(engine, messages, *, generation: int = 1):
    engine._compression_attempt_generation = generation
    assert engine.should_compress_preflight(deepcopy(messages)) is True
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


def test_cleanup_handoff_identity_bounds_expanded_replay_payload(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path, "bounded-handoff-identity")
    payload = "large-externalized-payload-" * 50_000
    messages = [{"role": "tool", "tool_call_id": "call-large", "content": payload}]
    replay_identity = Mock(
        return_value=("tool", payload, "call-large", "")
    )
    monkeypatch.setattr(
        engine,
        "_message_replay_identity",
        replay_identity,
    )

    identity = engine._cleanup_handoff_message_identity(messages)

    assert len(identity) == 64
    assert payload not in identity
    replay_identity.assert_called_once_with(messages[0])
    replay_identity.reset_mock()
    replay_identity.return_value = ("tool", payload + "changed", "call-large", "")
    assert identity != engine._cleanup_handoff_message_identity(
        [{"role": "tool", "tool_call_id": "call-large", "content": payload + "changed"}]
    )


def test_preflight_handoff_publication_is_atomic_with_prepare(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path, "atomic-handoff-publication")
    engine.threshold_tokens = 90_000
    engine._compression_attempt_generation = 33
    messages = [
        {
            "role": "user",
            "content": "api_key=sk-synthetic-atomic-handoff-000000000000",
        }
    ]
    ingest_entered = threading.Event()
    release_ingest = threading.Event()
    prepare_done = threading.Event()
    real_ingest = engine._ingest_messages
    results = {}

    def blocked_ingest(candidate_messages):
        ingest_entered.set()
        assert release_ingest.wait(5)
        return real_ingest(candidate_messages)

    monkeypatch.setattr(engine, "_ingest_messages", blocked_ingest)
    preflight = threading.Thread(
        target=lambda: results.setdefault(
            "preflight",
            engine.should_compress_preflight(deepcopy(messages)),
        )
    )
    preflight.start()
    assert ingest_entered.wait(5)

    def prepare():
        results["prepare"] = engine.prepare_compression_operation(
            deepcopy(messages),
            session_id=engine.bound_session_id,
            attempt_generation=33,
        )
        prepare_done.set()

    preparer = threading.Thread(target=prepare)
    preparer.start()
    prepare_overtook_publication = prepare_done.wait(0.2)
    release_ingest.set()
    preflight.join(5)
    preparer.join(5)

    assert not prepare_overtook_publication
    assert results["preflight"] is True
    assert results["prepare"] is not None


def test_claim_execution_is_serialized_with_bound_session_end(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path, "claim-session-end-race")
    messages = [
        {
            "role": "user",
            "content": "api_key=sk-synthetic-session-end-race-000000000",
        }
    ]
    claim = _claim_sanitation(engine, messages, generation=34)
    ingest_entered = threading.Event()
    release_ingest = threading.Event()
    invalidation_done = threading.Event()
    session_end_done = threading.Event()
    real_ingest = engine._ingest_messages
    real_invalidate = engine._invalidate_sanitation_operation
    results = {}

    def blocked_ingest(candidate_messages):
        ingest_entered.set()
        assert release_ingest.wait(5)
        return real_ingest(candidate_messages)

    def track_invalidation():
        real_invalidate()
        invalidation_done.set()

    monkeypatch.setattr(engine, "_ingest_messages", blocked_ingest)
    monkeypatch.setattr(engine, "_invalidate_sanitation_operation", track_invalidation)
    compressor = threading.Thread(
        target=lambda: results.setdefault(
            "compress",
            engine.compress(deepcopy(messages), operation_claim=claim),
        )
    )
    compressor.start()
    assert ingest_entered.wait(5)

    session_ender = threading.Thread(
        target=lambda: (
            engine.on_session_end(engine.bound_session_id, deepcopy(messages)),
            session_end_done.set(),
        )
    )
    session_ender.start()
    lifecycle_overtook_claim_execution = invalidation_done.wait(0.2)
    release_ingest.set()
    compressor.join(5)
    session_ender.join(5)

    assert not lifecycle_overtook_claim_execution
    assert session_end_done.is_set()
    assert isinstance(results["compress"], tuple)
    assert results["compress"][1] is claim


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


def test_preflight_handoff_without_host_attempt_can_be_claimed_by_first_generation(
    tmp_path,
):
    engine = _engine(tmp_path, "preflight-before-host-attempt")
    messages = [
        {
            "role": "user",
            "content": "api_key=sk-synthetic-preflight-before-host-attempt-000000",
        }
    ]
    assert getattr(engine, "_compression_attempt_generation", None) is None
    assert engine.should_compress_preflight(deepcopy(messages)) is True

    engine._compression_attempt_generation = 1

    prepared = engine.prepare_compression_operation(
        deepcopy(messages),
        session_id=engine.bound_session_id,
        attempt_generation=1,
    )
    assert prepared is not None
    assert prepared[0] == "sanitize"


def test_preflight_handoff_cannot_be_claimed_by_later_attempt_generation(tmp_path):
    engine = _engine(tmp_path, "stale-preflight-generation")
    messages = [
        {
            "role": "user",
            "content": "api_key=sk-synthetic-stale-preflight-generation-000000",
        }
    ]
    engine._compression_attempt_generation = 40
    assert engine.should_compress_preflight(deepcopy(messages)) is True

    engine._compression_attempt_generation = 41

    assert (
        engine.prepare_compression_operation(
            deepcopy(messages),
            session_id=engine.bound_session_id,
            attempt_generation=41,
        )
        is None
    )


def test_preflight_after_host_persistence_rebinds_handoff_to_latest_revision(
    tmp_path,
):
    engine = _engine(tmp_path, "preflight-after-host-persistence")
    messages = [
        {
            "role": "user",
            "content": "api_key=sk-synthetic-repeated-preflight-ingest-000000",
        }
    ]
    engine._compression_attempt_generation = 42
    assert engine.should_compress_preflight(deepcopy(messages)) is True
    first_revision = engine._foreground_ingest_revision

    engine._ingest_cursor = 0
    engine._ingest_cursor_needs_reconcile = True
    assert engine.should_compress_preflight(deepcopy(messages)) is True

    assert engine._foreground_ingest_revision > first_revision
    prepared = engine.prepare_compression_operation(
        deepcopy(messages),
        session_id=engine.bound_session_id,
        attempt_generation=42,
    )
    assert prepared is not None


def test_intervening_foreground_ingest_invalidates_sanitation_claim(tmp_path):
    engine = _engine(tmp_path, "claim-foreground-ingest-revision")
    messages = [
        {
            "role": "user",
            "content": "api_key=sk-synthetic-foreground-ingest-revision-000000",
        }
    ]
    claim = _claim_sanitation(engine, messages, generation=42)
    engine.ingest(messages + [{"role": "assistant", "content": "intervening durable turn"}])

    result = engine.compress(deepcopy(messages), operation_claim=claim)

    assert not (isinstance(result, tuple) and result[1] is claim)


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


def test_ignored_live_auxiliary_start_preserves_foreground_claim(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path, "claim-live-auxiliary-start")
    engine.threshold_tokens = 90_000
    messages = [
        {
            "role": "user",
            "content": "api_key=sk-synthetic-live-auxiliary-claim-000000000",
        }
    ]
    assert engine.should_compress_preflight(deepcopy(messages)) is True
    claim = _claim_sanitation(engine, messages, generation=32)
    pending_claim = engine._pending_sanitation_claim
    monkeypatch.setattr(
        engine,
        "_is_live_auxiliary_child_session",
        lambda *_args, **_kwargs: True,
    )

    engine.on_session_start(
        "ignored-auxiliary-child",
        platform="synthetic",
        parent_session_id=engine.bound_session_id,
        context_length=100_000,
    )

    assert engine._pending_sanitation_claim is pending_claim
    sanitized, returned_claim = engine.compress(
        deepcopy(messages),
        operation_claim=claim,
    )
    assert returned_claim is claim
    assert engine.last_compression_status == "sanitized"
    assert "sk-synthetic-live-auxiliary-claim" not in _content_text(sanitized)


def test_auxiliary_preflight_does_not_invalidate_foreground_sanitation_claim(tmp_path):
    class HostAgentFrame:
        def __init__(self, session_id: str, parent_session_id: str, hermes_home: str):
            self.session_id = session_id
            self._parent_session_id = parent_session_id
            self._hermes_home = hermes_home
            self.enabled_toolsets = ["memory", "skills"]
            self.log_prefix = "[subagent-test] "
            self._subagent_id = session_id
            self._delegate_depth = 1

        def on_session_start(self, engine: LCMEngine) -> None:
            engine.on_session_start(
                self.session_id,
                hermes_home=self._hermes_home,
                platform="telegram",
                context_length=100_000,
            )

        def should_compress_preflight(self, engine: LCMEngine, messages):
            return engine.should_compress_preflight(messages)

    engine = _engine(tmp_path, "claim-auxiliary-preflight-shared-engine")
    engine.threshold_tokens = 90_000
    foreground_messages = [
        {
            "role": "user",
            "content": "api_key=sk-synthetic-shared-engine-claim-000000000000",
        }
    ]
    assert engine.should_compress_preflight(deepcopy(foreground_messages)) is True
    claim = _claim_sanitation(engine, foreground_messages, generation=72)

    child = HostAgentFrame(
        "background-review-session",
        engine.current_session_id,
        str(engine._hermes_home),
    )
    child.on_session_start(engine)
    assert child.should_compress_preflight(
        engine,
        [{"role": "user", "content": "shared-engine auxiliary payload"}],
    ) is False

    claimed_result = engine.compress(
        deepcopy(foreground_messages),
        current_tokens=count_messages_tokens(foreground_messages),
        operation_claim=claim,
    )
    assert isinstance(
        claimed_result,
        tuple,
    ), "auxiliary preflight invalidated a foreground sanitation claim"
    sanitized, returned_claim = claimed_result
    assert returned_claim is claim
    assert engine.last_compression_status == "sanitized"
    assert "sk-synthetic-shared-engine-claim" not in _content_text(sanitized)

    assert engine.should_compress_preflight(deepcopy(foreground_messages)) is True
    second_claim = _claim_sanitation(engine, foreground_messages, generation=73)
    assert engine.should_compress_preflight(
        [{"role": "user", "content": "foreground intervening preflight"}]
    ) is False
    assert isinstance(
        engine.compress(deepcopy(foreground_messages), operation_claim=second_claim),
        list,
    )


def test_auxiliary_prepare_preserves_foreground_sanitation_claim(
    tmp_path,
    monkeypatch,
):
    class HostAgentFrame:
        def __init__(self, session_id: str, parent_session_id: str, hermes_home: str):
            self.session_id = session_id
            self._parent_session_id = parent_session_id
            self._hermes_home = hermes_home
            self.enabled_toolsets = ["memory", "skills"]
            self.log_prefix = "[subagent-test] "
            self._subagent_id = session_id
            self._delegate_depth = 1

        def on_session_start(self, engine: LCMEngine) -> None:
            engine.on_session_start(
                self.session_id,
                hermes_home=self._hermes_home,
                platform="telegram",
                context_length=100_000,
            )

        def prepare(self, engine: LCMEngine, messages, *, generation: int):
            assert engine.should_compress_preflight(messages) is True
            return engine.prepare_compression_operation(
                messages,
                session_id=self.session_id,
                attempt_generation=generation,
            )

    engine = _engine(tmp_path, "claim-auxiliary-prepare-shared-engine")
    engine.threshold_tokens = 90_000
    foreground_messages = [
        {
            "role": "user",
            "content": "api_key=sk-synthetic-auxiliary-prepare-claim-0000000000000",
        }
    ]
    monkeypatch.setattr(
        lcm_engine,
        "summarize_with_escalation",
        Mock(side_effect=AssertionError("claimed sanitation must not summarize")),
    )

    assert engine.should_compress_preflight(deepcopy(foreground_messages)) is True
    claim = _claim_sanitation(engine, foreground_messages, generation=75)
    pending_claim = engine._pending_sanitation_claim

    child = HostAgentFrame(
        "background-review-session",
        engine.current_session_id,
        str(engine._hermes_home),
    )
    child.on_session_start(engine)
    engine.threshold_tokens = 1
    assert (
        child.prepare(
            engine,
            [{"role": "user", "content": "above-threshold auxiliary payload"}],
            generation=75,
        )
        is None
    )
    assert (
        engine._pending_sanitation_claim is pending_claim
    ), "auxiliary preparation consumed a foreground sanitation claim"

    engine.threshold_tokens = 90_000
    sanitized, returned_claim = engine.compress(
        deepcopy(foreground_messages),
        operation_claim=claim,
    )
    assert returned_claim is claim
    assert engine.last_compression_status == "sanitized"
    assert "sk-synthetic-auxiliary-prepare-claim" not in _content_text(sanitized)


def test_mismatched_session_prepare_preserves_claim_but_foreground_stale_mismatch_invalidates(
    tmp_path,
):
    engine = _engine(tmp_path, "claim-prepare-mismatch")
    engine.threshold_tokens = 90_000
    messages = [
        {
            "role": "user",
            "content": "api_key=sk-synthetic-prepare-mismatch-0000000000000",
        }
    ]
    assert engine.should_compress_preflight(deepcopy(messages)) is True
    claim = _claim_sanitation(engine, messages, generation=76)
    pending_claim = engine._pending_sanitation_claim

    assert (
        engine.prepare_compression_operation(
            deepcopy(messages),
            session_id="different-host-session",
            attempt_generation=76,
        )
        is None
    )
    assert engine._pending_sanitation_claim is pending_claim
    assert (
        engine.prepare_compression_operation(
            [{"role": "user", "content": "stale foreground payload"}],
            session_id=engine.bound_session_id,
            attempt_generation=76,
        )
        is None
    )
    assert engine._pending_sanitation_claim is None
    assert isinstance(
        engine.compress(deepcopy(messages), operation_claim=claim),
        list,
    )


def test_auxiliary_compress_bypass_preserves_foreground_sanitation_claim(
    tmp_path,
    monkeypatch,
):
    class HostAgentFrame:
        def __init__(self, session_id: str, parent_session_id: str, hermes_home: str):
            self.session_id = session_id
            self._parent_session_id = parent_session_id
            self._hermes_home = hermes_home
            self.enabled_toolsets = ["memory", "skills"]
            self.log_prefix = "[subagent-test] "
            self._subagent_id = session_id
            self._delegate_depth = 1

        def on_session_start(self, engine: LCMEngine) -> None:
            engine.on_session_start(
                self.session_id,
                hermes_home=self._hermes_home,
                platform="telegram",
                context_length=100_000,
            )

        def compress(self, engine: LCMEngine, messages, **kwargs):
            return engine.compress(messages, **kwargs)

    engine = _engine(tmp_path, "claim-auxiliary-compress-shared-engine", fresh_tail_count=1)
    engine.threshold_tokens = 90_000
    foreground_messages = [
        {"role": "user", "content": "old eligible backlog " * 20},
        {"role": "assistant", "content": "old eligible answer " * 20},
        {
            "role": "user",
            "content": "api_key=sk-synthetic-auxiliary-compress-claim-0000000000000",
        },
        {"role": "assistant", "content": "fresh follow-up"},
    ]
    monkeypatch.setattr(
        lcm_engine,
        "summarize_with_escalation",
        Mock(side_effect=AssertionError("claimed sanitation must not summarize")),
    )

    assert engine.should_compress_preflight(deepcopy(foreground_messages)) is True
    claim = _claim_sanitation(engine, foreground_messages, generation=74)
    pending_claim = engine._pending_sanitation_claim

    child = HostAgentFrame(
        "background-review-session",
        engine.current_session_id,
        str(engine._hermes_home),
    )
    child.on_session_start(engine)
    child.compress(
        engine,
        [{"role": "user", "content": "shared-engine auxiliary payload"}],
    )

    assert (
        engine._pending_sanitation_claim is pending_claim
    ), "auxiliary bypass compression consumed a foreground sanitation claim"
    claimed_result = engine.compress(
        deepcopy(foreground_messages),
        current_tokens=count_messages_tokens(foreground_messages),
        operation_claim=claim,
    )
    assert isinstance(
        claimed_result,
        tuple,
    ), "foreground sanitation claim was not echoed after auxiliary bypass compression"
    sanitized, returned_claim = claimed_result
    assert returned_claim is claim
    assert engine.last_compression_status == "sanitized"
    assert "sk-synthetic-auxiliary-compress-claim" not in _content_text(sanitized)


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


def test_claim_echo_requires_post_ingest_handoff_match(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path, "claim-echo-handoff-match")
    engine.threshold_tokens = 90_000
    messages = [
        {
            "role": "user",
            "content": "api_key=sk-synthetic-claim-echo-handoff-match-0000000000000",
        },
        {"role": "assistant", "content": "fresh answer"},
    ]
    assert engine.should_compress_preflight(deepcopy(messages)) is True
    claim = _claim_sanitation(engine, messages, generation=27)
    mismatched_replay = deepcopy(messages)
    mismatched_replay[0][
        "content"
    ] = "api_key=sk-synthetic-claim-echo-handoff-mutated-0000000000000"
    monkeypatch.setattr(
        engine,
        "_ingest_messages",
        Mock(return_value=mismatched_replay),
    )

    result = engine.compress(
        deepcopy(messages),
        current_tokens=count_messages_tokens(messages),
        operation_claim=claim,
    )

    assert isinstance(
        result,
        list,
    ), "claim echo must require the post-ingest replay handoff to match"
    assert engine.last_compression_status == "sanitized"


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


def test_stale_scaffold_removal_reports_reassembled_when_assembly_matches(tmp_path):
    engine = _engine(
        tmp_path,
        "stale-scaffold-reassembled-status",
        fresh_tail_count=1,
        sensitive_patterns_enabled=False,
    )
    messages = [
        {
            "role": "user",
            "content": (
                "[Recent Summary (d0, node 999999)]\n"
                "retired compressed details\n"
                "[Expand for details: retired]"
            ),
        },
        {"role": "user", "content": "fresh question"},
    ]

    result = engine.compress(deepcopy(messages), operation_claim=object())

    assert result == [{"role": "user", "content": "fresh question"}]
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


def test_cleanup_critical_partial_leaf_is_not_classified_sanitation_only(
    tmp_path,
    monkeypatch,
):
    engine = _engine(
        tmp_path,
        "cleanup-critical-partial-leaf",
        fresh_tail_count=1,
        leaf_chunk_tokens=50_000,
        threshold_full_sweep_enabled=True,
    )
    messages = [
        {"role": "user", "content": "tiny critical raw prefix"},
        {
            "role": "assistant",
            "content": "api_key=sk-synthetic-critical-partial-leaf-0000000000000",
        },
        {"role": "user", "content": "fresh"},
    ]
    rough = count_messages_tokens(messages)
    engine.threshold_tokens = max(1, rough - 1)
    engine._last_boundary_skip_time = time.time()
    monkeypatch.setattr(
        engine,
        "_critical_budget_pressure_reached",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(
        engine,
        "_has_ignored_backlog_outside_fresh_tail",
        lambda _messages: False,
    )
    monkeypatch.setattr(
        engine,
        "_should_run_deferred_maintenance",
        lambda *_args, **_kwargs: False,
    )
    summary_spy = Mock(return_value=("critical partial leaf summary", 1))
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summary_spy)

    assert engine.should_compress_preflight(deepcopy(messages)) is True
    assert (
        engine._preflight_cleanup_only is False
    ), "critical partial leaves under threshold sweep must not be sanitation-only"
    result = engine.compress(
        deepcopy(messages),
        current_tokens=rough,
    )

    assert isinstance(result, list)
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


def test_cooldown_allows_critical_partial_threshold_sweep_leaf(
    tmp_path,
    monkeypatch,
):
    engine = _engine(
        tmp_path,
        "cooldown-critical-partial-threshold-sweep",
        fresh_tail_count=1,
        leaf_chunk_tokens=50_000,
        threshold_full_sweep_enabled=True,
        sensitive_patterns_enabled=False,
    )
    messages = [
        {"role": "user", "content": "tiny critical raw prefix"},
        {"role": "assistant", "content": "tiny critical answer"},
        {"role": "user", "content": "fresh"},
    ]
    rough = count_messages_tokens(messages)
    engine.threshold_tokens = max(1, rough - 1)
    engine._last_boundary_skip_time = time.time()
    monkeypatch.setattr(
        engine,
        "_critical_budget_pressure_reached",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(
        engine,
        "_has_ignored_backlog_outside_fresh_tail",
        lambda _messages: False,
    )
    monkeypatch.setattr(
        engine,
        "_should_run_deferred_maintenance",
        lambda *_args, **_kwargs: False,
    )
    summary_spy = Mock(return_value=("critical partial sweep summary", 1))
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summary_spy)

    assert engine.should_compress_preflight(deepcopy(messages)) is True
    result = engine.compress(deepcopy(messages), current_tokens=rough)

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


@pytest.mark.parametrize("caller", ["host_claim", "direct"])
def test_cooldown_cleanup_handoff_above_threshold_stays_sanitation_only(
    tmp_path,
    monkeypatch,
    caller,
):
    engine = _engine(
        tmp_path,
        f"cooldown-cleanup-threshold-{caller}",
        fresh_tail_count=1,
        threshold_full_sweep_enabled=False,
    )
    engine._last_boundary_skip_time = time.time()
    messages = [
        {"role": "user", "content": "old eligible backlog " * 40},
        {"role": "assistant", "content": "old eligible answer"},
        {
            "role": "user",
            "content": "api_key=sk-synthetic-cooldown-threshold-" + ("x" * 3_000),
        },
    ]
    rough = count_messages_tokens(messages)
    engine.threshold_tokens = max(1, rough - 1)
    engine._compression_attempt_generation = 31
    summary_spy = Mock(
        side_effect=AssertionError(
            "cooldown-authorized sanitation must not summarize"
        )
    )
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summary_spy)

    assert engine.should_compress_preflight(deepcopy(messages)) is True
    assert engine._preflight_cleanup_only is True

    if caller == "host_claim":
        prepared = engine.prepare_compression_operation(
            deepcopy(messages),
            session_id=engine.bound_session_id,
            attempt_generation=31,
        )
        assert prepared is not None
        operation, claim = prepared
        assert operation == "sanitize"
        engine._last_boundary_skip_time = 0
        sanitized, returned_claim = engine.compress(
            deepcopy(messages),
            current_tokens=rough,
            operation_claim=claim,
        )
        assert returned_claim is claim
    else:
        engine._last_boundary_skip_time = 0
        sanitized = engine.compress(
            deepcopy(messages),
            current_tokens=rough,
        )

    assert engine.last_compression_status == "sanitized"
    assert engine._dag.get_session_node_count(engine.current_session_id) == 0
    assert "sk-synthetic-cooldown-threshold" not in _content_text(sanitized)
    summary_spy.assert_not_called()


def test_direct_compress_without_current_tokens_preserves_cleanup_pressure_for_deferred(
    tmp_path,
    monkeypatch,
):
    engine = _engine(
        tmp_path,
        "direct-no-current-preserve-pressure",
        fresh_tail_count=1,
        leaf_chunk_tokens=5_000,
        dynamic_leaf_chunk_enabled=False,
        threshold_full_sweep_enabled=False,
        critical_budget_pressure_ratio=0.8,
    )
    engine.context_length = 1_000
    engine.threshold_tokens = 90_000
    messages = [
        {"role": "user", "content": "old eligible request"},
        {"role": "assistant", "content": "old eligible answer"},
        {
            "role": "user",
            "content": "api_key=sk-synthetic-direct-no-current-" + ("x" * 4_000),
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
    monkeypatch.setattr(
        engine,
        "_has_ignored_backlog_outside_fresh_tail",
        lambda _messages: False,
    )
    maintenance_observed_tokens = []

    def deferred_maintenance_due(_messages, *, observed_tokens=None):
        maintenance_observed_tokens.append(observed_tokens)
        return True

    monkeypatch.setattr(
        engine,
        "_should_run_deferred_maintenance",
        deferred_maintenance_due,
    )
    summary_spy = Mock(return_value=("direct no-current summary", 1))
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summary_spy)

    assert rough >= critical_floor
    assert engine.should_compress_preflight(deepcopy(messages)) is True
    assert count_messages_tokens(replays[0]) < critical_floor
    assert engine._preflight_cleanup_only is False

    result = engine.compress(deepcopy(messages))

    assert isinstance(result, list)
    assert engine.last_compression_status == "compacted"
    assert maintenance_observed_tokens[-1] == rough
    summary_spy.assert_called()


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


def test_cleanup_handoff_identity_includes_normalized_timestamp(tmp_path):
    engine = _engine(tmp_path, "handoff-timestamp-identity")
    base = [{"role": "user", "content": "same content", "timestamp": 1750000000.0}]

    changed_epoch = deepcopy(base)
    changed_epoch[0]["timestamp"] = 1750000001.0
    same_instant_iso = deepcopy(base)
    same_instant_iso[0]["timestamp"] = "2025-06-15T15:06:40+00:00"
    same_instant_offset = deepcopy(base)
    same_instant_offset[0]["timestamp"] = "2025-06-15T11:06:40-04:00"
    numeric_string = deepcopy(base)
    numeric_string[0]["timestamp"] = "1750000000.0"
    missing_timestamp = [{"role": "user", "content": "same content"}]
    naive_rejected = deepcopy(base)
    naive_rejected[0]["timestamp"] = "2025-07-15T16:26:40"

    base_identity = engine._cleanup_handoff_message_identity(base)
    assert base_identity != engine._cleanup_handoff_message_identity(changed_epoch)
    assert base_identity != engine._cleanup_handoff_message_identity(missing_timestamp)
    assert base_identity != engine._cleanup_handoff_message_identity(naive_rejected)
    assert base_identity == engine._cleanup_handoff_message_identity(same_instant_iso)
    assert base_identity == engine._cleanup_handoff_message_identity(same_instant_offset)
    assert base_identity == engine._cleanup_handoff_message_identity(numeric_string)


def test_timestamp_change_between_preflight_and_prepare_consumes_handoff(tmp_path):
    engine = _engine(tmp_path, "handoff-timestamp-claim")
    engine.threshold_tokens = 90_000
    messages = [
        {
            "role": "user",
            "content": "api_key=sk-synthetic-handoff-timestamp-000000000000",
            "timestamp": "2025-06-15T15:06:40+00:00",
        },
        {"role": "assistant", "content": "fresh answer", "timestamp": 1750000000.5},
    ]
    assert engine.should_compress_preflight(deepcopy(messages)) is True
    stale_timestamp = deepcopy(messages)
    stale_timestamp[0]["timestamp"] = 1750000009.0

    assert (
        engine.prepare_compression_operation(
            stale_timestamp,
            session_id=engine.bound_session_id,
            attempt_generation=1,
        )
        is None
    ), "a changed message timestamp must consume the published handoff"
    assert engine._preflight_cleanup_handoff is None, (
        "the stale-timestamp prepare must consume the handoff atomically"
    )

    # A fresh preflight of the unchanged replay republishes a handoff that
    # the original generation can still claim.
    engine._compression_attempt_generation = 1
    assert engine.should_compress_preflight(deepcopy(messages)) is True
    prepared = engine.prepare_compression_operation(
        deepcopy(messages),
        session_id=engine.bound_session_id,
        attempt_generation=1,
    )
    assert prepared is not None
    operation, claim = prepared
    assert operation == "sanitize"


def test_load_externalized_payload_requires_string_content(tmp_path):
    engine = _engine(tmp_path, "sidecar-nonstring-content")
    try:
        storage = tmp_path / "sidecar-nonstring-content-externalized"
        storage.mkdir(parents=True, exist_ok=True)
        for name, content in (
            ("truthy-list.json", [1, 2]),
            ("dict.json", {"a": 1}),
            ("number.json", 123),
            ("empty-list.json", []),
        ):
            (storage / name).write_text(
                json.dumps(
                    {
                        "kind": "tool_result",
                        "tool_call_id": "call-nonstring",
                        "content": content,
                    }
                ),
                encoding="utf-8",
            )
            assert engine.load_externalized_payload_sidecar(name) is None, name

        (storage / "no-content.json").write_text(
            json.dumps({"kind": "tool_result"}),
            encoding="utf-8",
        )
        assert engine.load_externalized_payload_sidecar("no-content.json") is None

        # The summary fallback used by legacy readers must not raise on
        # non-string content either (e.g. metadata re-reads of malformed files).
        summary = _externalized_summary(Path("malformed.json"), {"content": [1, 2]})
        assert summary["content_chars"] is None
        assert summary["content_bytes"] is None

        (storage / "ok.json").write_text(
            json.dumps({"kind": "tool_result", "content": "durable text"}),
            encoding="utf-8",
        )
        loaded = engine.load_externalized_payload_sidecar("ok.json")
        assert loaded is not None
        assert loaded["content"] == "durable text"
        assert loaded["content_chars"] == len("durable text")
    finally:
        engine.shutdown()


def test_stale_auxiliary_session_end_preserves_reused_foreground_claim(tmp_path):
    class StaleAuxiliaryFrame:
        def __init__(self, session_id: str):
            self.session_id = session_id
            self._subagent_id = session_id
            self._delegate_depth = 1

        def end(self, engine: LCMEngine, messages) -> None:
            engine.on_session_end(self.session_id, messages)

    engine = _engine(tmp_path, "claim-stale-aux-reused-end")
    engine.threshold_tokens = 90_000
    messages = [
        {
            "role": "user",
            "content": "api_key=sk-synthetic-stale-aux-reuse-0000000000000",
        },
        {"role": "assistant", "content": "fresh answer"},
    ]
    assert engine.should_compress_preflight(deepcopy(messages)) is True
    claim = _claim_sanitation(engine, messages, generation=81)
    pending_claim = engine._pending_sanitation_claim

    stale_frame = StaleAuxiliaryFrame(engine.bound_session_id)
    stale_frame.end(engine, [{"role": "user", "content": "stale auxiliary tail"}])

    assert (
        engine._pending_sanitation_claim is pending_claim
    ), "a stale auxiliary end for a reused foreground id destroyed the fresh claim"
    sanitized, returned_claim = engine.compress(
        deepcopy(messages),
        operation_claim=claim,
    )
    assert returned_claim is claim
    assert engine.last_compression_status == "sanitized"
    assert "sk-synthetic-stale-aux-reuse" not in _content_text(sanitized)


def test_session_end_does_not_block_behind_unclaimed_summarization(
    tmp_path,
    monkeypatch,
):
    engine = _engine(
        tmp_path,
        "session-end-not-behind-summarization",
        fresh_tail_count=1,
        leaf_chunk_tokens=1,
        sensitive_patterns_enabled=False,
    )
    engine.threshold_tokens = 1
    messages = [
        {"role": "user", "content": "eligible backlog " * 40},
        {"role": "assistant", "content": "eligible answer"},
        {"role": "user", "content": "fresh question"},
    ]
    summarize_started = threading.Event()
    release_summarization = threading.Event()

    def gated_summarization(**_kwargs):
        summarize_started.set()
        assert release_summarization.wait(10)
        return ("slow summary", 1)

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", gated_summarization)

    results = {}
    compressor = threading.Thread(
        target=lambda: results.setdefault(
            "compress",
            engine.compress(deepcopy(messages), current_tokens=100_000),
        )
    )
    compressor.start()
    assert summarize_started.wait(5)

    # While unclaimed summarization runs, the claim lock must be free so a
    # concurrent bound session end can invalidate claims without waiting.
    lock_free = engine._sanitation_claim_lock.acquire(blocking=False)
    if lock_free:
        engine._sanitation_claim_lock.release()
    assert lock_free, "unclaimed compaction must not hold the claim lock through summarization"

    ender_done = threading.Event()
    ender = threading.Thread(
        target=lambda: (
            engine.on_session_end(engine.bound_session_id, [{"role": "user", "content": "end"}]),
            ender_done.set(),
        )
    )
    ender.start()
    ender_completed_during_summarization = ender_done.wait(2.0)

    release_summarization.set()
    compressor.join(10)
    ender.join(10)

    assert ender_completed_during_summarization, (
        "on_session_end must not wait behind unclaimed compaction summarization"
    )
    assert not compressor.is_alive()
    assert not ender.is_alive()
    assert isinstance(results["compress"], list)
    assert engine.last_compression_status == "compacted"


def test_claimed_sanitation_stays_serialized_with_bound_session_end(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path, "claimed-sanitation-serialized")
    engine.threshold_tokens = 90_000
    messages = [
        {
            "role": "user",
            "content": "api_key=sk-synthetic-claimed-serialize-000000000000",
        },
        {"role": "assistant", "content": "fresh answer"},
    ]
    claim = _claim_sanitation(engine, messages, generation=91)

    ingest_entered = threading.Event()
    release_ingest = threading.Event()
    invalidation_done = threading.Event()
    session_end_done = threading.Event()
    real_ingest = engine._ingest_messages
    real_invalidate = engine._invalidate_sanitation_operation
    results = {}

    def blocked_ingest(candidate_messages):
        ingest_entered.set()
        assert release_ingest.wait(10)
        return real_ingest(candidate_messages)

    def track_invalidation():
        real_invalidate()
        invalidation_done.set()

    monkeypatch.setattr(engine, "_ingest_messages", blocked_ingest)
    monkeypatch.setattr(engine, "_invalidate_sanitation_operation", track_invalidation)
    compressor = threading.Thread(
        target=lambda: results.setdefault(
            "compress",
            engine.compress(deepcopy(messages), operation_claim=claim),
        )
    )
    compressor.start()
    assert ingest_entered.wait(5)

    session_ender = threading.Thread(
        target=lambda: (
            engine.on_session_end(engine.bound_session_id, deepcopy(messages)),
            session_end_done.set(),
        )
    )
    session_ender.start()
    lifecycle_overtook_claim_execution = invalidation_done.wait(0.2)
    release_ingest.set()
    compressor.join(10)
    session_ender.join(10)

    assert not lifecycle_overtook_claim_execution, (
        "bound session end must not invalidate while claimed sanitation executes"
    )
    assert session_end_done.is_set()
    assert isinstance(results["compress"], tuple)
    assert results["compress"][1] is claim


def test_tool_call_ingest_is_serialized_with_claimed_compression(tmp_path, monkeypatch):
    engine = _engine(tmp_path, "tool-call-ingest-claim-race2")
    engine.threshold_tokens = 90_000
    messages = [
        {
            "role": "user",
            "content": "api_key=sk-synthetic-tool-call-ingest-000000000000",
        },
        {"role": "assistant", "content": "fresh answer"},
    ]
    later_messages = deepcopy(messages)
    later_messages.append({"role": "user", "content": "later tool-turn follow-up"})
    claim = _claim_sanitation(engine, messages, generation=35)

    ingest_entered = threading.Event()
    release_ingest = threading.Event()
    real_ingest = engine._ingest_messages
    gate_used = threading.Semaphore(0)
    results = {}

    def gated_ingest(candidate_messages):
        if gate_used.acquire(blocking=False):
            # Second (compressor-side) call passes through.
            return real_ingest(candidate_messages)
        ingest_entered.set()
        assert release_ingest.wait(10), "tool-call ingest was never released"
        return real_ingest(candidate_messages)

    monkeypatch.setattr(engine, "_ingest_messages", gated_ingest)

    tool_caller = threading.Thread(
        target=lambda: results.setdefault(
            "tool_response",
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "unrelated"},
                messages=deepcopy(later_messages),
            ),
        )
    )
    tool_caller.start()
    assert ingest_entered.wait(5)

    lock_free = engine._sanitation_claim_lock.acquire(blocking=False)
    if lock_free:
        engine._sanitation_claim_lock.release()
    assert not lock_free, "the blocked tool-call ingest must hold the claim lock"

    compressor = threading.Thread(
        target=lambda: results.setdefault(
            "compress",
            engine.compress(deepcopy(messages), operation_claim=claim),
        )
    )
    compressor.start()
    compressor_still_blocked = not compressor.join(0.5)
    assert compressor_still_blocked, (
        "claimed compression must not complete while a tool-call ingest holds the claim lock"
    )

    release_ingest.set()
    tool_caller.join(10)
    compressor.join(10)

    assert not tool_caller.is_alive()
    assert not compressor.is_alive()
    assert isinstance(results["tool_response"], str)
    assert isinstance(results["compress"], list), (
        "a stale sanitation claim must not be echoed after an intervening tool-call ingest"
    )
