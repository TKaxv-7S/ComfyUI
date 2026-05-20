"""Tests for the opaque per-prompt metadata mechanism on PromptServer."""

from unittest.mock import MagicMock

import pytest

from comfy_execution.jobs import extract_workflow_id


class TestExtractWorkflowId:

    def test_returns_id_when_present(self):
        assert extract_workflow_id({"extra_pnginfo": {"workflow": {"id": "wf-1"}}}) == "wf-1"

    def test_returns_none_when_missing(self):
        assert extract_workflow_id({}) is None
        assert extract_workflow_id({"extra_pnginfo": {}}) is None
        assert extract_workflow_id({"extra_pnginfo": {"workflow": {}}}) is None

    def test_returns_none_for_empty_or_wrong_type(self):
        assert extract_workflow_id({"extra_pnginfo": {"workflow": {"id": ""}}}) is None
        assert extract_workflow_id({"extra_pnginfo": {"workflow": {"id": 42}}}) is None
        assert extract_workflow_id({"extra_pnginfo": {"workflow": {"id": None}}}) is None

    def test_returns_none_for_non_dict_input(self):
        assert extract_workflow_id(None) is None
        assert extract_workflow_id("not a dict") is None
        assert extract_workflow_id({"extra_pnginfo": "not a dict"}) is None
        assert extract_workflow_id({"extra_pnginfo": {"workflow": "not a dict"}}) is None


class _FakeServer:
    """Minimal PromptServer stand-in mirroring send_sync verbatim."""

    def __init__(self):
        self.active_prompt_metadata = None
        self.captured = []
        self.loop = MagicMock()
        self.loop.call_soon_threadsafe.side_effect = (
            lambda fn, msg: self.captured.append(msg)
        )
        self.messages = MagicMock()
        self.messages.put_nowait = MagicMock()

    def send_sync(self, event, data, sid=None):
        meta = self.active_prompt_metadata
        if meta and isinstance(data, dict):
            data = {**meta, **data}
        self.loop.call_soon_threadsafe(
            self.messages.put_nowait, (event, data, sid)
        )


@pytest.fixture
def server():
    return _FakeServer()


class TestSendSyncMerge:
    def test_spreads_active_metadata_onto_dict_payload(self, server):
        server.active_prompt_metadata = {"workflow_id": "wf-1"}

        server.send_sync(
            "executing", {"node": "n1", "prompt_id": "p1"}, "client-1"
        )

        event, data, sid = server.captured[0]
        assert event == "executing"
        assert data == {
            "workflow_id": "wf-1",
            "node": "n1",
            "prompt_id": "p1",
        }
        assert sid == "client-1"

    def test_passthrough_when_no_active_metadata(self, server):
        server.active_prompt_metadata = None

        server.send_sync("executing", {"node": "n1", "prompt_id": "p1"})

        _, data, _ = server.captured[0]
        assert data == {"node": "n1", "prompt_id": "p1"}

    def test_passthrough_when_metadata_is_empty_dict(self, server):
        server.active_prompt_metadata = {}

        server.send_sync("executing", {"node": "n1", "prompt_id": "p1"})

        _, data, _ = server.captured[0]
        assert data == {"node": "n1", "prompt_id": "p1"}

    def test_event_payload_wins_on_key_conflict(self, server):
        server.active_prompt_metadata = {"workflow_id": "wf-1", "prompt_id": "from-meta"}

        server.send_sync("executing", {"node": "n1", "prompt_id": "from-frame"}, "c1")

        _, data, _ = server.captured[0]
        assert data["prompt_id"] == "from-frame"
        assert data["workflow_id"] == "wf-1"

    def test_non_dict_payload_passes_through_untouched(self, server):
        # BinaryEventTypes.TEXT byte frames must not be merged.
        server.active_prompt_metadata = {"workflow_id": "wf-1"}

        server.send_sync("text", b"\x00\x00\x00\x03foobar", "c1")

        _, data, _ = server.captured[0]
        assert data == b"\x00\x00\x00\x03foobar"

    def test_terminal_executing_frame_includes_metadata(self, server):
        # Slot is cleared after this send in main.py so the reset still carries metadata (#13684 race).
        server.active_prompt_metadata = {"workflow_id": "wf-1"}

        server.send_sync(
            "executing", {"node": None, "prompt_id": "p1"}, "client-1"
        )

        _, data, _ = server.captured[0]
        assert data == {
            "workflow_id": "wf-1",
            "node": None,
            "prompt_id": "p1",
        }

    def test_opaque_dict_supports_arbitrary_keys(self, server):
        server.active_prompt_metadata = {
            "workflow_id": "wf-1",
            "trace_id": "trace-123",
            "tenant": "acme",
        }

        server.send_sync("executing", {"node": "n1", "prompt_id": "p1"})

        _, data, _ = server.captured[0]
        assert data["workflow_id"] == "wf-1"
        assert data["trace_id"] == "trace-123"
        assert data["tenant"] == "acme"


class TestWorkerSerializationIsolatesMetadata:
    def test_two_prompts_sharing_prompt_id_get_correct_metadata(self, server):
        # Prompt A
        server.active_prompt_metadata = {"workflow_id": "wf-AAA"}
        server.send_sync("execution_start", {"prompt_id": "P-shared"})
        server.send_sync("executing", {"node": "n1", "prompt_id": "P-shared"})
        server.send_sync("executing", {"node": None, "prompt_id": "P-shared"})
        server.active_prompt_metadata = None

        # Prompt B — same prompt_id, different workflow
        server.active_prompt_metadata = {"workflow_id": "wf-BBB"}
        server.send_sync("execution_start", {"prompt_id": "P-shared"})
        server.send_sync("executing", {"node": "n2", "prompt_id": "P-shared"})
        server.send_sync("executing", {"node": None, "prompt_id": "P-shared"})
        server.active_prompt_metadata = None

        frames = [d for (_, d, _) in server.captured]
        a_frames = frames[:3]
        b_frames = frames[3:]

        assert all(f["workflow_id"] == "wf-AAA" for f in a_frames)
        assert all(f["workflow_id"] == "wf-BBB" for f in b_frames)
        assert all(f["prompt_id"] == "P-shared" for f in frames)
