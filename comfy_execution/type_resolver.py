"""Server-side type resolver for prompt graphs.

Resolves the concrete io_type of an output slot or input slot by walking the
prompt graph. Handles:

  * Static V1/V3 ``RETURN_TYPES`` (returned as-is).
  * V3 ``MatchType.Output`` (resolved by walking inputs that share the same
    ``template_id`` until a concrete type is found).
  * Cycles and unbounded recursion (terminates at ``AnyType`` with a one-shot
    warning).
  * Unknown / unresolvable / wildcard outputs (fall back to ``AnyType`` with a
    one-shot warning).

The resolver works against either a raw prompt dict
(``{node_id: {"class_type": str, "inputs": dict}}``) or a
``comfy_execution.graph.DynamicPrompt`` instance.

All resolved values are plain strings, so the resolver state is trivially
serializable across processes if needed.
"""

from __future__ import annotations

import logging
from typing import Any

from comfy_api.latest import io
from comfy_api.internal import _ComfyNodeInternal

# Sentinel for "type is unknown / wildcard". Matches AnyType.io_type ("*").
ANY_TYPE: str = io.AnyType.io_type

# Hard cap on resolver recursion depth. MatchType chains should never be
# anywhere near this deep; this is a belt-and-suspenders guard against malformed
# graphs and pathological cycles.
MAX_RESOLVE_DEPTH: int = 64


class TypeResolver:
    """Resolves concrete io_types for a prompt graph.

    Instantiate once per prompt (or per ``DynamicPrompt``) and reuse; results
    are cached. Call :py:meth:`invalidate` (or :py:meth:`invalidate_node`) when
    the underlying graph mutates (e.g. when an ephemeral node is added).
    """

    def __init__(self, prompt_source: Any):
        """Args:
            prompt_source: Either a ``DynamicPrompt`` (anything with
                ``get_node(node_id)`` / ``has_node(node_id)``) or a plain
                ``dict[node_id, {"class_type", "inputs"}]``.
        """
        self._source = prompt_source
        self._output_cache: dict[tuple[str, int], str] = {}
        self._is_output_list_cache: dict[tuple[str, int], bool] = {}
        self._warned: set[tuple[str, Any, str]] = set()

    # ---- prompt access ----------------------------------------------------
    def _has_node(self, node_id: str) -> bool:
        if hasattr(self._source, "has_node"):
            return self._source.has_node(node_id)
        return node_id in self._source

    def _get_node(self, node_id: str) -> dict[str, Any] | None:
        try:
            if hasattr(self._source, "get_node"):
                return self._source.get_node(node_id)
            return self._source[node_id]
        except Exception:
            return None

    @staticmethod
    def _get_class_def(class_type: str):
        # Local import to avoid a hard import-cycle between nodes.py and
        # comfy_execution at module-load time.
        import nodes
        return nodes.NODE_CLASS_MAPPINGS.get(class_type)

    # ---- cache management -------------------------------------------------
    def invalidate(self) -> None:
        """Clear all cached resolutions. Cheap; call after any graph mutation."""
        self._output_cache.clear()
        self._is_output_list_cache.clear()
        # Intentionally do NOT clear self._warned: those messages are already
        # logged and re-warning would just spam the log.

    def invalidate_node(self, node_id: str) -> None:
        """Clear cached entries for a single node (e.g. after node-level expand)."""
        for key in [k for k in self._output_cache if k[0] == node_id]:
            del self._output_cache[key]
        for key in [k for k in self._is_output_list_cache if k[0] == node_id]:
            del self._is_output_list_cache[key]

    # ---- output resolution -----------------------------------------------
    def resolve_output_type(self, node_id: str, slot_idx: int,
                            _stack: frozenset[tuple[str, int]] | None = None) -> str:
        """Return the resolved io_type string of ``node_id``'s output slot.

        Falls back to ``ANY_TYPE`` on cycle, depth-overflow, unknown class,
        out-of-range slot, missing node, or unresolved MatchType template.
        """
        cache_key = (node_id, slot_idx)
        if cache_key in self._output_cache:
            return self._output_cache[cache_key]

        if _stack is None:
            _stack = frozenset()
        if cache_key in _stack:
            self._warn(node_id, slot_idx, "cycle detected during type resolution; defaulting to AnyType")
            return ANY_TYPE
        if len(_stack) >= MAX_RESOLVE_DEPTH:
            self._warn(node_id, slot_idx, f"exceeded MAX_RESOLVE_DEPTH={MAX_RESOLVE_DEPTH}; defaulting to AnyType")
            return ANY_TYPE
        next_stack = _stack | {cache_key}

        if not self._has_node(node_id):
            return ANY_TYPE

        node = self._get_node(node_id)
        if node is None:
            return ANY_TYPE

        class_type = node.get("class_type")
        class_def = self._get_class_def(class_type) if class_type is not None else None
        if class_def is None:
            return ANY_TYPE

        try:
            return_types = class_def.RETURN_TYPES
        except Exception:
            return ANY_TYPE
        if return_types is None or slot_idx < 0 or slot_idx >= len(return_types):
            return ANY_TYPE

        declared = return_types[slot_idx]

        # V3 nodes may have MatchType outputs that need to be traced through
        # the schema. V1 nodes (and V3 nodes with plain outputs) just use the
        # declared RETURN_TYPES string.
        resolved = declared
        if isinstance(class_def, type) and issubclass(class_def, _ComfyNodeInternal):
            schema = getattr(class_def, "SCHEMA", None)
            if schema is None:
                # Trigger schema computation. RETURN_TYPES would have done this
                # already, but be defensive.
                try:
                    schema = class_def.GET_SCHEMA()
                except Exception:
                    schema = None
            if schema is not None and slot_idx < len(schema.outputs):
                out = schema.outputs[slot_idx]
                if isinstance(out, io.MatchType.Output):
                    resolved = self._resolve_match_template(
                        node_id, schema, out.template.template_id, next_stack
                    )

        # Treat the legacy wildcard literally as AnyType. We warn only when the
        # source node's *declared* type was already wildcard, so MatchType-style
        # "no upstream connected" cases (which warn elsewhere) don't double-warn.
        if isinstance(resolved, str) and resolved == ANY_TYPE and declared == ANY_TYPE:
            self._warn(
                node_id, slot_idx,
                f"node '{class_type}' output slot {slot_idx} is wildcard; defaulting to AnyType",
            )

        if not isinstance(resolved, str):
            # Non-string types (e.g., legacy combos passed as list) — bail to AnyType.
            self._warn(node_id, slot_idx,
                       f"node '{class_type}' output slot {slot_idx} has non-string return type {type(resolved).__name__}; defaulting to AnyType")
            resolved = ANY_TYPE

        self._output_cache[cache_key] = resolved
        return resolved

    def _resolve_match_template(self, node_id: str, schema, template_id: str,
                                stack: frozenset[tuple[str, int]]) -> str:
        """Resolve a MatchType.Output by inspecting the node's MatchType.Inputs
        with the same template_id.

        Strategy (per design decision): walk inputs in schema order, pick the
        FIRST concrete (non-AnyType) resolution. If none resolve, return
        AnyType with a one-shot warning.
        """
        node = self._get_node(node_id)
        inputs_dict = (node or {}).get("inputs", {}) or {}
        any_input_seen = False
        for inp in schema.inputs:
            if not isinstance(inp, io.MatchType.Input):
                continue
            if inp.template.template_id != template_id:
                continue
            any_input_seen = True
            val = inputs_dict.get(inp.id)
            if val is None:
                continue
            if isinstance(val, list) and len(val) == 2 and isinstance(val[0], str):
                src_node, src_slot = val[0], val[1]
                t = self.resolve_output_type(src_node, src_slot, stack)
                if t != ANY_TYPE:
                    return t
            # Literal value: a MatchType slot has no concrete declared type, so
            # we cannot infer anything useful here.
        if not any_input_seen:
            # Schema declared a template_id with no Input bearing it. This is a
            # node-author bug; warn once.
            self._warn(node_id, None,
                       f"MatchType output template '{template_id}' has no matching Input on the node; defaulting to AnyType")
        else:
            self._warn(node_id, None,
                       f"MatchType template '{template_id}' has no bound concrete upstream input; defaulting to AnyType")
        return ANY_TYPE

    def is_output_list(self, node_id: str, slot_idx: int) -> bool:
        """Whether the source slot is declared as a list output (``OUTPUT_IS_LIST[idx]``)."""
        cache_key = (node_id, slot_idx)
        if cache_key in self._is_output_list_cache:
            return self._is_output_list_cache[cache_key]
        result = False
        node = self._get_node(node_id)
        if node is not None:
            class_def = self._get_class_def(node.get("class_type"))
            if class_def is not None:
                lst = getattr(class_def, "OUTPUT_IS_LIST", None)
                if lst is not None and 0 <= slot_idx < len(lst):
                    result = bool(lst[slot_idx])
        self._is_output_list_cache[cache_key] = result
        return result

    # ---- input resolution ------------------------------------------------
    def resolve_input_type(self, node_id: str, input_id: str) -> str:
        """Resolve the io_type of the value currently bound to a node's input.

        * If the value is a link, return the resolved type of the source slot.
        * If the value is a literal, return the declared slot's effective
          io_type (peeling dynamic-input wrappers — e.g. an Autogrow-of-Image
          slot resolves to ``IMAGE``, not ``COMFY_AUTOGROW_V3``).
        * If the value is missing or the slot is unknown, return ``ANY_TYPE``.
        """
        node = self._get_node(node_id)
        if node is None:
            return ANY_TYPE
        inputs = node.get("inputs", {}) or {}
        if input_id not in inputs:
            return ANY_TYPE
        val = inputs[input_id]
        if isinstance(val, list) and len(val) == 2 and isinstance(val[0], str):
            return self.resolve_output_type(val[0], val[1])
        return self.get_declared_slot_io_type(node_id, input_id)

    def is_input_list(self, node_id: str, input_id: str) -> bool:
        """Whether the value bound to ``input_id`` originates from a list output."""
        node = self._get_node(node_id)
        if node is None:
            return False
        val = (node.get("inputs", {}) or {}).get(input_id)
        if isinstance(val, list) and len(val) == 2 and isinstance(val[0], str):
            return self.is_output_list(val[0], val[1])
        return False

    def get_declared_slot_io_type(self, node_id: str, input_id: str) -> str:
        """Return the effective declared io_type of a node's input slot.

        Peels dynamic-input wrappers so that the user-facing element type is
        returned:

        * Autogrow → wrapped template input's io_type
        * DynamicSlot → underlying slot's io_type
        * Anything else → the slot's own io_type
        * DynamicCombo / unsupported → ``ANY_TYPE`` (the combo key is itself
          dynamic, not a meaningful type for consumers)
        """
        node = self._get_node(node_id)
        if node is None:
            return ANY_TYPE
        class_def = self._get_class_def(node.get("class_type"))
        if class_def is None:
            return ANY_TYPE

        # Prefer V3 schema (carries dynamic-input wrapper info).
        if isinstance(class_def, type) and issubclass(class_def, _ComfyNodeInternal):
            schema = getattr(class_def, "SCHEMA", None)
            if schema is None:
                try:
                    class_def.GET_SCHEMA()
                    schema = getattr(class_def, "SCHEMA", None)
                except Exception:
                    schema = None
            if schema is not None:
                # First, try a top-level input id match.
                for inp in schema.inputs:
                    if inp.id == input_id:
                        return self._effective_io_type(inp)
                # Then a nested match (DynamicSlot / DynamicCombo prefix.child).
                if "." in input_id:
                    top, _, _ = input_id.partition(".")
                    for inp in schema.inputs:
                        if inp.id != top:
                            continue
                        for child in inp.get_all():
                            if child is inp:
                                continue
                            if child.id == input_id.split(".", 1)[1]:
                                return self._effective_io_type(child)
                # Fall through to V1 dict for hidden inputs etc.

        # V1 fallback: look at INPUT_TYPES() dict.
        try:
            inputs = class_def.INPUT_TYPES()
        except Exception:
            return ANY_TYPE
        for section in ("required", "optional"):
            section_d = inputs.get(section, {})
            if input_id in section_d:
                entry = section_d[input_id]
                if not entry:
                    return ANY_TYPE
                t = entry[0]
                if isinstance(t, str):
                    return t
                if isinstance(t, list):
                    # legacy combo declared as a list of options.
                    return io.Combo.io_type
                return ANY_TYPE
        return ANY_TYPE

    @staticmethod
    def _effective_io_type(inp) -> str:
        """Return the consumer-facing io_type of a (possibly dynamic) input."""
        # Autogrow wraps a template input — the element type is what matters.
        if isinstance(inp, io.Autogrow.Input):
            try:
                return inp.template.input.get_io_type()
            except Exception:
                return ANY_TYPE
        # DynamicSlot wraps an underlying slot input.
        if isinstance(inp, io.DynamicSlot.Input):
            try:
                return inp.slot.get_io_type()
            except Exception:
                return ANY_TYPE
        # DynamicCombo's "type" is a key value selector, not a connection type.
        if isinstance(inp, io.DynamicCombo.Input):
            return ANY_TYPE
        # Everything else: trust the input's declared io_type.
        try:
            return inp.get_io_type()
        except Exception:
            return ANY_TYPE

    # ---- bulk helpers ----------------------------------------------------
    def compute_live_input_types(self, node_id: str) -> dict[str, str]:
        """Build the ``{input_id: resolved_io_type}`` map for a node.

        Used by :py:func:`comfy_api.latest._io.get_finalized_class_inputs` so
        future dynamic-input expansion strategies (per-type DynamicType, etc.)
        can branch on what was actually connected.
        """
        node = self._get_node(node_id)
        if node is None:
            return {}
        out: dict[str, str] = {}
        for input_id in (node.get("inputs", {}) or {}).keys():
            out[input_id] = self.resolve_input_type(node_id, input_id)
        return out

    # ---- diagnostics -----------------------------------------------------
    def _warn(self, node_id: str, slot_idx: Any, msg: str) -> None:
        key = (node_id, slot_idx, msg)
        if key in self._warned:
            return
        self._warned.add(key)
        logging.warning("TypeResolver: node=%s slot=%s %s", node_id, slot_idx, msg)
