"""Unit tests verifying all registered Ghana Search MCP tools by name,
including input execution and all four required MCP quality annotations (readOnlyHint,
destructiveHint, idempotentHint, openWorldHint).
"""
import pytest
import json
import asyncio
from ghana_mcp.server import (
    mcp,
    ghana_search,
    ghana_news_search,
    ghana_academic_search,
    ghana_public_services_search,
    ghana_wikipedia_search,
    ghana_business_directory,
    ghana_business_search,
)


def test_ghana_search_by_name():
    """Verify primary national knowledge retrieval tool."""
    raw = ghana_search("Ghana Revenue Authority", top_k=2)
    assert raw is not None
    data = json.loads(raw)
    assert isinstance(data, list)


def test_ghana_news_search_by_name():
    """Verify live news search tool."""
    raw = ghana_news_search("Ghana inflation economy", top_k=2)
    assert raw is not None
    data = json.loads(raw)
    assert isinstance(data, list)


def test_ghana_academic_search_by_name():
    """Verify university academic thesis repository tool."""
    raw = ghana_academic_search("solar energy storage", top_k=2)
    assert raw is not None
    data = json.loads(raw)
    assert isinstance(data, list)


def test_ghana_public_services_search_by_name():
    """Verify government agencies and public services tool."""
    raw = ghana_public_services_search("passport office", top_k=2)
    assert raw is not None
    data = json.loads(raw)
    assert isinstance(data, list)


def test_ghana_wikipedia_search_by_name():
    """Verify Ghana-scoped Wikipedia tool."""
    raw = ghana_wikipedia_search("President of Ghana", top_k=2)
    assert raw is not None
    data = json.loads(raw)
    assert isinstance(data, list)


def test_ghana_business_directory_by_name():
    """Verify commercial business directory tool."""
    raw = ghana_business_directory("solar panels", top_k=2)
    assert raw is not None
    data = json.loads(raw)
    assert isinstance(data, list)


def test_ghana_business_search_by_name():
    """Verify commercial business search tool alias."""
    raw = ghana_business_search("pharmacy", top_k=2)
    assert raw is not None
    data = json.loads(raw)
    assert isinstance(data, list)


def test_all_tools_have_all_four_annotations():
    """Verify all 4 hints (readOnlyHint, destructiveHint, idempotentHint, openWorldHint)
    are explicitly defined as booleans on every registered tool.
    """
    async def _check():
        tools = await mcp.list_tools()
        assert len(tools) >= 5

        for tool in tools:
            assert tool.annotations is not None, f"Tool '{tool.name}' is missing annotations object."
            assert tool.annotations.read_only_hint is True, f"Tool '{tool.name}' must have read_only_hint=True."
            assert tool.annotations.destructive_hint is False, f"Tool '{tool.name}' must have destructive_hint=False."
            assert tool.annotations.idempotent_hint is True, f"Tool '{tool.name}' must have idempotent_hint=True."
            assert tool.annotations.open_world_hint is True, f"Tool '{tool.name}' must have open_world_hint=True."

    asyncio.run(_check())


if __name__ == "__main__":
    test_ghana_search_by_name()
    test_ghana_news_search_by_name()
    test_ghana_academic_search_by_name()
    test_ghana_public_services_search_by_name()
    test_ghana_wikipedia_search_by_name()
    test_ghana_business_directory_by_name()
    test_ghana_business_search_by_name()
    asyncio.run(test_all_tools_have_all_four_annotations())
    print("All tool tests and annotation verification passed successfully!")
