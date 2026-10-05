"""Store-agnostic metadata filter DSL.

Two input forms are accepted by the API:

* **Structured** — boolean groups of conditions, nestable::

    {"must": [{"field": "brand", "op": "eq", "value": "Panasonic"},
              {"field": "price", "op": "lte", "value": 120}],
     "should": [{"field": "tags", "op": "in", "value": ["quiet", "led"]}],
     "must_not": [{"field": "discontinued", "op": "eq", "value": true}]}

* **Shorthand** — a flat mapping ANDed together::

    {"brand": "Panasonic", "tags": ["quiet", "led"], "price": {"gte": 10, "lte": 120}}

Field names refer to user metadata (``brand`` → ``metadata.brand``). The system fields
``source_id``, ``chunk_index``, ``chunk_hash``, ``title`` and ``mime_type`` refer to the point
itself. Vector-store adapters translate a :class:`FilterSpec` into their native query language; the
in-memory store and tests evaluate it with :func:`matches`.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, model_validator

from rag_service.core.errors import InvalidFilterError
from rag_service.core.utils import get_path
from rag_service.domain.models import PayloadKeys

MAX_DEPTH = 5
MAX_CONDITIONS = 64


class FilterOp(StrEnum):
    EQ = "eq"
    NE = "ne"
    IN = "in"
    NIN = "nin"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    EXISTS = "exists"


RANGE_OPS = {FilterOp.GT, FilterOp.GTE, FilterOp.LT, FilterOp.LTE}
_SHORTHAND_OPS = {op.value for op in FilterOp}


class Condition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str = Field(..., min_length=1, max_length=256, examples=["brand"])
    op: FilterOp = FilterOp.EQ
    value: Any = None

    @model_validator(mode="after")
    def _check_value(self) -> Condition:
        if self.op in (FilterOp.IN, FilterOp.NIN):
            if not isinstance(self.value, list) or not self.value:
                raise ValueError(f"'{self.op}' requires a non-empty list value")
            if any(isinstance(v, dict | list) for v in self.value):
                raise ValueError(f"'{self.op}' values must be scalars")
        elif self.op in RANGE_OPS:
            if isinstance(self.value, bool) or not isinstance(self.value, int | float | str):
                raise ValueError(f"'{self.op}' requires a number or ISO-8601 datetime string")
        elif self.op == FilterOp.EXISTS:
            if self.value is None:
                self.value = True
            if not isinstance(self.value, bool):
                raise ValueError("'exists' requires a boolean value")
        elif isinstance(self.value, dict | list) or self.value is None:
            raise ValueError(f"'{self.op}' requires a scalar value")
        return self

    @property
    def path(self) -> str:
        return resolve_field(self.field)


def _item_kind(value: Any) -> str:
    """Discriminate group items explicitly: anything with a ``field`` key is a condition.

    Without this, a smart union could read ``{"field": ..., "op": ...}`` as a *shorthand group*
    (every key a field name), which is wrong and recursively expensive.
    """
    if isinstance(value, Mapping):
        return "condition" if "field" in value else "group"
    return "condition" if isinstance(value, Condition) else "group"


FilterItem = Annotated[
    Annotated[Condition, Tag("condition")] | Annotated["FilterSpec", Tag("group")],
    Discriminator(_item_kind),
]


class FilterSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    must: list[FilterItem] = Field(default_factory=list)
    should: list[FilterItem] = Field(default_factory=list)
    must_not: list[FilterItem] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _accept_shorthand(cls, data: Any) -> Any:
        if isinstance(data, Mapping) and data and not set(data) & {"must", "should", "must_not"}:
            return {"must": _shorthand_to_conditions(data)}
        return data

    @model_validator(mode="after")
    def _check_limits(self) -> FilterSpec:
        depth, count = _measure(self)
        if depth > MAX_DEPTH:
            raise ValueError(f"filter nesting deeper than {MAX_DEPTH}")
        if count > MAX_CONDITIONS:
            raise ValueError(f"filter has more than {MAX_CONDITIONS} conditions")
        return self

    def is_empty(self) -> bool:
        return not (self.must or self.should or self.must_not)

    @classmethod
    def all_of(cls, *conditions: Condition | FilterSpec | None) -> FilterSpec:
        return cls(must=[c for c in conditions if c is not None and not (isinstance(c, FilterSpec) and c.is_empty())])

    @classmethod
    def for_source(cls, source_id: str) -> FilterSpec:
        return cls(must=[Condition(field=PayloadKeys.SOURCE_ID, op=FilterOp.EQ, value=source_id)])


def _shorthand_to_conditions(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    conditions: list[dict[str, Any]] = []
    for key, value in data.items():
        if isinstance(value, list):
            conditions.append({"field": key, "op": "in", "value": value})
        elif isinstance(value, Mapping):
            unknown = set(value) - _SHORTHAND_OPS
            if unknown:
                raise ValueError(f"unknown operator(s) for '{key}': {sorted(unknown)}")
            conditions.extend({"field": key, "op": op, "value": v} for op, v in value.items())
        else:
            conditions.append({"field": key, "op": "eq", "value": value})
    return conditions


def _measure(spec: FilterSpec, depth: int = 1) -> tuple[int, int]:
    max_depth, count = depth, 0
    for item in (*spec.must, *spec.should, *spec.must_not):
        if isinstance(item, FilterSpec):
            d, c = _measure(item, depth + 1)
            max_depth, count = max(max_depth, d), count + c
        else:
            count += 1
    return max_depth, count


def resolve_field(name: str) -> str:
    """Map an API field name to its payload path."""
    if name in PayloadKeys.SYSTEM:
        return name
    if name.startswith(f"{PayloadKeys.METADATA}."):
        return name
    return f"{PayloadKeys.METADATA}.{name}"


def parse_filter(raw: Any) -> FilterSpec | None:
    if raw is None or isinstance(raw, FilterSpec):
        return raw
    try:
        return FilterSpec.model_validate(raw)
    except Exception as exc:
        raise InvalidFilterError(str(exc)) from exc


# ---------------------------------------------------------------------------- evaluation


def _compare(op: FilterOp, actual: Any, expected: Any) -> bool:
    if op == FilterOp.EXISTS:
        present = actual is not None and actual != []
        return present is bool(expected)
    values = actual if isinstance(actual, list) else [actual]
    if op == FilterOp.EQ:
        return expected in values
    if op == FilterOp.NE:
        return expected not in values
    if op == FilterOp.IN:
        return any(v in expected for v in values)
    if op == FilterOp.NIN:
        return not any(v in expected for v in values)
    for v in values:
        if v is None:
            continue
        try:
            if (
                (op == FilterOp.GT and v > expected)
                or (op == FilterOp.GTE and v >= expected)
                or (op == FilterOp.LT and v < expected)
                or (op == FilterOp.LTE and v <= expected)
            ):
                return True
        except TypeError:
            continue
    return False


def _item_matches(item: Condition | FilterSpec, payload: Mapping[str, Any]) -> bool:
    if isinstance(item, FilterSpec):
        return matches(item, payload)
    return _compare(item.op, get_path(payload, item.path), item.value)


def matches(spec: FilterSpec | None, payload: Mapping[str, Any]) -> bool:
    """Reference semantics used by the in-memory store (and mirrored by every adapter)."""
    if spec is None or spec.is_empty():
        return True
    if spec.must and not all(_item_matches(i, payload) for i in spec.must):
        return False
    if spec.should and not any(_item_matches(i, payload) for i in spec.should):
        return False
    return not (spec.must_not and any(_item_matches(i, payload) for i in spec.must_not))
