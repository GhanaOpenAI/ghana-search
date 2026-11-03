"""Tests for Ghana-MCP NLP extraction, providers, and server tools."""
import pytest
from ghana_mcp.nlp import extract_search_terms, get_singular_plural_variants
from ghana_mcp.providers.base import SearchResult
from ghana_mcp.providers.public_services import PublicServicesProvider
from ghana_mcp.server import _execute_search


def test_singular_plural_variants():
    """Verify singular and plural generation."""
    vars_tax = get_singular_plural_variants("tax")
    assert "tax" in vars_tax and "taxes" in vars_tax

    vars_battery = get_singular_plural_variants("battery")
    assert "battery" in vars_battery and "batteries" in vars_battery

    vars_minister = get_singular_plural_variants("ministers")
    assert "minister" in vars_minister and "ministers" in vars_minister


def test_nlp_noun_chunk_cleaning():
    """Verify cleaning of articles, adverbs, and pronouns while retaining adjectives like 'former'."""
    text = "who was the former vice president of ghana and his very new ministers?"
    res = extract_search_terms(text)

    # Noun chunks should preserve 'former vice president'
    assert "former vice president" in res["clean_chunks"]

    # Noun chunks should not contain 'the', 'very', 'who', etc.
    all_chunks_text = " ".join(res["clean_chunks"]).lower()
    assert "who" not in all_chunks_text
    assert "very" not in all_chunks_text
    assert "ghana" in all_chunks_text


def test_public_services_directory():
    """Verify live content search on verified Ghana public services portals."""
    provider = PublicServicesProvider()
    results = provider.search(["tax", "vat"], limit=5)
    assert len(results) > 0
    assert any("gra.gov.gh" in r.url or "Revenue" in r.source for r in results)
    assert any(r.url.startswith("http") for r in results)


def test_execute_search_end_to_end():
    """Verify end-to-end search query execution."""
    results = _execute_search("who is the current president of ghana", top_k=5, rerank=True)
    assert len(results) > 0
    assert all(isinstance(r, SearchResult) for r in results)
    assert any("president" in r.title.lower() or "ghana" in r.title.lower() for r in results)


def test_unified_search_all_sources():
    """Verify unified search includes entity definitions, landmark locations, and knowledge by default."""
    results = _execute_search("Ridge Hospital", top_k=5, rerank=True)
    assert len(results) > 0
    categories = {r.category for r in results}
    # Should include entity definition and/or location without needing specific category filter
    assert "named_entities" in categories or "location" in categories
    assert any("Ridge Hospital" in r.title for r in results)


if __name__ == "__main__":
    print("Testing singular/plural variants...")
    test_singular_plural_variants()
    print("Testing NLP noun chunk cleaning...")
    test_nlp_noun_chunk_cleaning()
    print("Testing Public Services provider...")
    test_public_services_directory()
    print("Testing End-to-End search with Gemini reranking...")
    test_execute_search_end_to_end()
    print("All unit tests passed successfully!")
