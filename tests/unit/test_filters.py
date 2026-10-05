from __future__ import annotations

import pytest

from rag_service.adapters.vectorstores.pgvector import PgFilterCompiler
from rag_service.core.errors import InvalidFilterError
from rag_service.domain.filters import FilterSpec, matches, parse_filter, resolve_field

PAYLOAD = {
    "source_id": "s1",
    "chunk_index": 3,
    "metadata": {
        "brand": "Delta",
        "cfm": 80,
        "tags": ["quiet", "led"],
        "released": "2024-05-01",
        "rating": 4.5,
    },
}


def test_shorthand_is_anded() -> None:
    spec = parse_filter({"brand": "Delta", "tags": ["led", "x"], "cfm": {"gte": 50, "lte": 100}})
    assert spec is not None
    assert len(spec.must) == 4
    assert matches(spec, PAYLOAD)
    assert not matches(parse_filter({"brand": "Panasonic"}), PAYLOAD)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({"must": [{"field": "brand", "op": "eq", "value": "Delta"}]}, True),
        ({"must": [{"field": "brand", "op": "ne", "value": "Delta"}]}, False),
        ({"must": [{"field": "tags", "op": "eq", "value": "quiet"}]}, True),  # array membership
        ({"must": [{"field": "tags", "op": "nin", "value": ["heater"]}]}, True),
        ({"must": [{"field": "cfm", "op": "gt", "value": 80}]}, False),
        ({"must": [{"field": "rating", "op": "eq", "value": 4.5}]}, True),
        ({"must": [{"field": "released", "op": "gte", "value": "2024-01-01"}]}, True),
        ({"must": [{"field": "missing", "op": "exists", "value": True}]}, False),
        ({"must": [{"field": "missing", "op": "exists", "value": False}]}, True),
        (
            {
                "should": [
                    {"field": "brand", "op": "eq", "value": "X"},
                    {"field": "cfm", "op": "eq", "value": 80},
                ]
            },
            True,
        ),
        ({"must_not": [{"field": "tags", "op": "in", "value": ["led"]}]}, False),
        (
            {
                "must": [
                    {"field": "source_id", "op": "eq", "value": "s1"},
                    {"field": "chunk_index", "op": "lte", "value": 3},
                ]
            },
            True,
        ),
        (
            {
                "must": [
                    {"should": [{"field": "brand", "op": "eq", "value": "Delta"}]},
                    {"field": "cfm", "op": "lt", "value": 100},
                ]
            },
            True,
        ),
    ],
)
def test_reference_semantics(raw: dict[str, object], expected: bool) -> None:
    assert matches(parse_filter(raw), PAYLOAD) is expected


@pytest.mark.parametrize(
    "raw",
    [
        {"must": [{"field": "a", "op": "in", "value": "not-a-list"}]},
        {"must": [{"field": "a", "op": "gt", "value": [1]}]},
        {"must": [{"field": "a", "op": "bogus", "value": 1}]},
        {"a": {"between": 1}},
    ],
)
def test_invalid_filters(raw: dict[str, object]) -> None:
    with pytest.raises(InvalidFilterError):
        parse_filter(raw)


def test_field_resolution() -> None:
    assert resolve_field("brand") == "metadata.brand"
    assert resolve_field("metadata.brand") == "metadata.brand"
    assert resolve_field("source_id") == "source_id"


def test_pg_compiler_parameterises_values() -> None:
    spec = parse_filter({"brand": "De'lta\"; DROP TABLE x;--", "source_id": "s1", "cfm": {"gte": 50}})
    sql, params = PgFilterCompiler().compile(spec)
    assert "DROP" not in sql
    assert sql.count("%s") == len(params)
    assert params[1] == ["s1"]  # column fast-path for source_id
    assert params[0] == '$."metadata"."brand" ? (@ == "De\'lta\\"; DROP TABLE x;--")'


def test_qdrant_translation_shapes() -> None:
    pytest.importorskip("qdrant_client")
    from rag_service.adapters.vectorstores.qdrant import QdrantFilterTranslator

    spec = FilterSpec.model_validate(
        {
            "must": [
                {"field": "brand", "op": "ne", "value": "X"},
                {"field": "rating", "op": "eq", "value": 4.5},
            ]
        }
    )
    qfilter = QdrantFilterTranslator().translate(spec)
    assert qfilter.must is not None and len(qfilter.must) == 2
    assert qfilter.must[1].range.gte == 4.5  # float equality becomes a closed range
