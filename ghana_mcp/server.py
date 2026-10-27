"""Ghana Search Server: Ghana's Information Retrieval Engine & Model Context Protocol (MCP) Server.
Developed by Ghana Open AI.
"""
import os
import sys
import json
import asyncio
from typing import List, Optional, Dict, Any, Union
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
import uvicorn
from fastapi import FastAPI, Query, Body, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, PlainTextResponse, FileResponse
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
import mcp.types as types

from ghana_mcp.providers.news import NewsProvider
from ghana_mcp.providers.wikipedia import WikipediaGhanaProvider
from ghana_mcp.providers.academic import AcademicProvider
from ghana_mcp.providers.public_services import PublicServicesProvider
from ghana_mcp.providers.ghalii import GhaLIIProvider
from ghana_mcp.providers.business import BusinessDirectoryProvider
from ghana_mcp.providers.price_finder import PriceFinderProvider
from ghana_mcp.providers.base import SearchResult
from ghana_mcp.reranker import reranker
from ghana_mcp.nlp import extract_search_terms
from ghana_mcp.analytics import analytics
from ghana_mcp.cache import search_cache
from ghana_mcp.hf_sync import hf_sync
from ghana_mcp.config import SERVER_HOST, SERVER_PORT, MAX_SEARCH_SECONDS, PROVIDER_TIMEOUT

# Initialize providers
news_provider = NewsProvider()
wiki_provider = WikipediaGhanaProvider()
academic_provider = AcademicProvider()
public_provider = PublicServicesProvider()
ghalii_provider = GhaLIIProvider()
business_provider = BusinessDirectoryProvider()
price_finder = PriceFinderProvider()

# Offline Mode Flag (air-gapped / local cache operation)
OFFLINE_MODE = (
    os.environ.get("GHANA_SEARCH_OFFLINE", "").lower() in ("1", "true", "yes")
    or "--offline" in sys.argv
)

# MCP Tool Annotations (All four hints explicitly declared for M8ven / OpenAI / Claude)
MCP_TOOL_ANNOTATIONS = types.ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=True,
)

# Initialize MCP Server
mcp = MCPServer(
    name="ghana-search",
    title="Ghana Search Engine",
    description="Search Ghanaian local news, university thesis archives, government agencies, and Ghana Wikipedia with 24-hour caching and hybrid reranking.",
    version="1.0.0",
)


def _execute_search(
    text: str,
    categories: Optional[List[str]] = None,
    top_k: int = 50,
    rerank: bool = True,
    client_type: str = "mcp_tool",
    user_agent: str = "",
) -> List[SearchResult]:
    """Execute search across verified Ghanaian websites with 24-hour caching and evaluation logging."""
    if not text or not text.strip():
        return []

    import time
    start_time = time.perf_counter()

    # 1. Extract clean noun phrases and entity nouns via spaCy
    nlp_data = extract_search_terms(text)
    primary_query = nlp_data.get("primary_query") or text.strip()
    clean_chunks = nlp_data.get("clean_chunks", [])
    lemma_chunks = nlp_data.get("lemma_chunks", [])
    nouns = nlp_data.get("nouns", [])

    cats = categories or ["news", "wikipedia", "academic", "public_services"]

    # 2. Check 24-hour cache first (instant response for repeated queries)
    cached_results = search_cache.get(text, cats, top_k)
    if cached_results is not None:
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        analytics.record_query_sync(
            query_text=text,
            clean_chunks=clean_chunks,
            num_results=len(cached_results),
            latency_ms=latency_ms,
            cached=True,
            client_type=client_type,
            user_agent=user_agent,
            categories=cats,
            sources={r.source: 1 for r in cached_results},
            top_results_summary=[
                {"title": r.title, "url": r.url, "source": r.source}
                for r in cached_results[:5]
            ],
        )
        return cached_results

    # Priority search terms: clean noun chunks first, followed by lemma chunks and individual nouns
    search_keywords = clean_chunks if clean_chunks else nouns
    if not search_keywords:
        search_keywords = [t for t in text.split() if len(t) > 1]

    # Offline Mode: Search exclusively across local SQLite cache database with 0 network calls
    if OFFLINE_MODE:
        offline_tokens = clean_chunks if clean_chunks else nouns
        offline_candidates = search_cache.search_offline(offline_tokens, cats, limit=top_k * 3)
        if rerank and offline_candidates:
            ranked = reranker.rerank(
                query=text,
                keywords=search_keywords,
                candidates=offline_candidates,
                top_k=min(top_k, len(offline_candidates)),
            )
        else:
            ranked = offline_candidates[:top_k]

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        analytics.record_query_sync(
            query_text=text,
            clean_chunks=clean_chunks,
            num_results=len(ranked),
            latency_ms=latency_ms,
            cached=False,
            client_type=client_type,
            user_agent=user_agent,
            categories=cats,
            sources={r.source: 1 for r in ranked},
            top_results_summary=[
                {"title": r.title, "url": r.url, "source": r.source}
                for r in ranked[:5]
            ],
        )
        return ranked

    candidates: List[SearchResult] = []
    per_provider_limit = min(max(int(top_k * 0.15), 6), 10)

    # 3. Parallel asynchronous dispatch across all selected providers
    tasks = []
    if "news" in cats:
        tasks.append(("news", lambda: news_provider.search(search_keywords, limit=per_provider_limit)))
    if "wikipedia" in cats:
        tasks.append(("wiki", lambda: wiki_provider.search(search_keywords, limit=min(per_provider_limit, 6))))
    if "academic" in cats:
        tasks.append(("academic", lambda: academic_provider.search(search_keywords, limit=per_provider_limit)))
    if "public_services" in cats:
        tasks.append(("public_services", lambda: public_provider.search(search_keywords, limit=per_provider_limit)))
        tasks.append(("ghalii", lambda: ghalii_provider.search(search_keywords, limit=min(per_provider_limit, 6))))
    if "business" in cats:
        tasks.append(("business", lambda: business_provider.search(search_keywords, limit=per_provider_limit)))

    from concurrent.futures import ThreadPoolExecutor, wait
    executor = ThreadPoolExecutor(max_workers=len(tasks) or 1)
    future_to_task = {executor.submit(fn): name for name, fn in tasks}
    # Wait strictly up to PROVIDER_TIMEOUT budget (3.2s so total response delivers under 4.0s)
    done, not_done = wait(future_to_task.keys(), timeout=PROVIDER_TIMEOUT)
    for future in done:
        try:
            res = future.result()
            if res:
                candidates.extend(res)
        except Exception:
            pass
    # Shutdown without blocking for straggler workers
    executor.shutdown(wait=False, cancel_futures=True)

    # Deduplicate candidates by canonical URL and hydrate from permanent URL content cache
    seen_urls = set()
    unique_candidates: List[SearchResult] = []
    sources_count: Dict[str, int] = {}
    candidate_urls = [c.url for c in candidates if c.url]

    # Tier 2: Batch check permanent URL content cache (0ms SQLite lookup)
    permanent_cached_docs = search_cache.get_urls_batch(candidate_urls)

    for c in candidates:
        norm_url = c.url.strip().rstrip("/")
        if norm_url and norm_url not in seen_urls:
            seen_urls.add(norm_url)
            # If this URL was permanently cached previously, hydrate full body content & metadata
            if c.url in permanent_cached_docs:
                cached_doc = permanent_cached_docs[c.url]
                if not c.content and cached_doc.content:
                    c.content = cached_doc.content
                if not c.published_date and cached_doc.published_date:
                    c.published_date = cached_doc.published_date
                if cached_doc.metadata.get("pdf_url") and not c.metadata.get("pdf_url"):
                    c.metadata["pdf_url"] = cached_doc.metadata["pdf_url"]
            unique_candidates.append(c)
            sources_count[c.source] = sources_count.get(c.source, 0) + 1
        elif not norm_url and c.title not in seen_urls:
            seen_urls.add(c.title)
            unique_candidates.append(c)
            sources_count[c.source] = sources_count.get(c.source, 0) + 1

    effective_k = min(top_k, len(unique_candidates)) if len(unique_candidates) > 0 else top_k

    # 4. 100% CPU Terminology Density & Recency Reranking
    if rerank and unique_candidates:
        ranked = reranker.rerank(
            query=text,
            keywords=search_keywords,
            candidates=unique_candidates,
            top_k=effective_k,
        )
    else:
        ranked = unique_candidates[:effective_k]

    # Tier 2: Permanently store all unique documents per URL (title, content, date saved forever)
    search_cache.set_urls_batch(unique_candidates)

    # Tier 1: Store ranked query results in 24-hour cache
    search_cache.set(text, cats, top_k, ranked)

    # Record evaluation dataset & analytics non-blockingly
    latency_ms = (time.perf_counter() - start_time) * 1000.0
    analytics.record_query_sync(
        query_text=text,
        clean_chunks=clean_chunks,
        num_results=len(ranked),
        latency_ms=latency_ms,
        cached=False,
        client_type=client_type,
        user_agent=user_agent,
        categories=cats,
        sources=sources_count,
        top_results_summary=[
            {"title": r.title, "url": r.url, "source": r.source}
            for r in ranked[:5]
        ],
    )

    return ranked


# --- MCP Tool Registrations ---


@mcp.tool(annotations=MCP_TOOL_ANNOTATIONS)
def ghana_search(
    text: str,
    top_k: int = 50,
    categories: Optional[List[str]] = None,
    rerank: bool = True,
) -> str:
    """Search Ghanaian local news, university thesis archives, government agencies, and Ghana Wikipedia.
    Takes free-form text input, extracts nouns using spaCy, queries local APIs, and ranks results by relevance.
    All searches strictly enforce a 4.0-second timeout budget across all providers.

    Args:
        text: Free-form text input or query from the LLM (e.g. "Who is the current president of Ghana?", "Research on solar energy from KNUST and UCC", "Ghana Revenue Authority tax requirements")
        top_k: Number of most relevant results to return (default: 50, maximum: total retrieved candidates)
        categories: Optional list of data sources to query. Defaults to knowledge repositories: ["news", "wikipedia", "academic", "public_services"]. Pass "business" explicitly to query Yellow Pages commercial listings.
        rerank: Whether to rerank candidates by terminology and recency (default: True)
    """
    results = _execute_search(text, categories, top_k, rerank)
    return json.dumps([r.to_dict() for r in results], indent=2)


@mcp.tool(annotations=MCP_TOOL_ANNOTATIONS)
def ghana_news_search(
    text: str,
    top_k: int = 50,
) -> str:
    """Search live Ghanaian online news outlets with verified public APIs (B&FT Online, 3News, Yen.com.gh, Citi Newsroom, GNA, Chronicle, GBN, Starr FM).
    Extracts nouns via spaCy and returns articles ordered by relevance.

    Args:
        text: News search query or phrase (e.g. "Ghana cedi exchange rate and Bank of Ghana inflation update")
        top_k: Number of relevant articles to return (default: 50)
    """
    results = _execute_search(text, categories=["news"], top_k=top_k, rerank=True)
    return json.dumps([r.to_dict() for r in results], indent=2)


@mcp.tool(annotations=MCP_TOOL_ANNOTATIONS)
def ghana_academic_search(
    text: str,
    top_k: int = 50,
) -> str:
    """Search Ghanaian university institutional repositories (KNUST, University of Ghana, UCC) for academic thesis abstracts.
    Extracts key research terms via spaCy and returns abstracts with official handle links.

    Args:
        text: Academic research topic (e.g. "renewable solar energy storage and battery technology")
        top_k: Number of thesis abstracts to return (default: 50)
    """
    results = _execute_search(text, categories=["academic"], top_k=top_k, rerank=True)
    return json.dumps([r.to_dict() for r in results], indent=2)


@mcp.tool(annotations=MCP_TOOL_ANNOTATIONS)
def ghana_public_services_search(
    text: str,
    top_k: int = 50,
) -> str:
    """Search 1,862 Ghanaian public sector institutions, ministries, departments, assemblies, and official service portals.

    Args:
        text: Government department or citizen service (e.g. "business registration, passport office, GRA tax filing")
        top_k: Number of public entities to return (default: 50)
    """
    results = _execute_search(text, categories=["public_services"], top_k=top_k, rerank=True)
    return json.dumps([r.to_dict() for r in results], indent=2)


@mcp.tool(annotations=MCP_TOOL_ANNOTATIONS)
def ghana_wikipedia_search(
    text: str,
    top_k: int = 50,
) -> str:
    """Search Wikipedia with location and context set to Ghana for background history, leaders, and landmarks.

    Args:
        text: Topic of interest (e.g. "Founding fathers of Ghana and 1992 constitution")
        top_k: Number of Wikipedia summaries to return (default: 50)
    """
    results = _execute_search(text, categories=["wikipedia"], top_k=top_k, rerank=True)
    return json.dumps([r.to_dict() for r in results], indent=2)


@mcp.tool(annotations=MCP_TOOL_ANNOTATIONS)
def ghana_business_directory(
    query: str,
    top_k: int = 15,
) -> str:
    """Search the Ghana Business Directory & Yellow Pages for companies, commercial enterprises, shops, and services across all 16 regions of Ghana.
    Works both ways:
    1. Product / Commodity / Service Search: Pass what you are looking to buy or hire (e.g. 'solar panels', 'fertilizer', 'tractor parts', 'pharmacies in Kumasi', 'car batteries') to get a list of Ghanaian businesses that sell them with physical locations and phone contacts.
    2. Business Name Lookup: Pass a company or provider name (e.g. 'Ernest Chemists', 'Solar 4 Ghana', 'Lawyer Gwira') to retrieve their profile, location, and contact numbers.

    Args:
        query: Product, commodity, service, or business name to search for
        top_k: Number of business listings to return (default: 15)
    """
    results = _execute_search(query, categories=["business"], top_k=top_k, rerank=True)
    return json.dumps([r.to_dict() for r in results], indent=2)


@mcp.tool(annotations=MCP_TOOL_ANNOTATIONS)
def ghana_business_search(text: str, top_k: int = 15) -> str:
    """Alias for ghana_business_directory."""
    return ghana_business_directory(query=text, top_k=top_k)


@mcp.tool(annotations=MCP_TOOL_ANNOTATIONS)
def ghana_price_finder(
    product: str,
    limit: int = 10,
) -> str:
    """Search Ghanaian marketplace classifieds to find active product listings and calculate real-time market price metrics in Ghana Cedis (GH₵).
    Calculates the average price, price range (min/max), and returns current active ad listings with locations and links.

    Args:
        product: Product or item name to price check (e.g. 'MacBook Pro', 'solar inverter', 'car tracking system', 'HP toner cartridge')
        limit: Number of classified listings to sample (default: 10)
    """
    res = price_finder.search_prices(product, limit=limit)
    return json.dumps(res, indent=2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown lifecycle cleanly."""
    await analytics.start()
    await hf_sync.start()
    yield
    await hf_sync.stop()
    await analytics.stop()


# --- FastAPI REST App & Web Interface ---

app = FastAPI(
    title="Ghana Search API",
    description="Ghana Search: National Information Retrieval Engine. Developed by Ghana Open AI.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/mcp",
    mcp.sse_app(
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
        host="0.0.0.0",
    ),
)


class SearchRequest(BaseModel):
    text: str = Field(..., description="Free-form query text input")
    categories: Optional[List[str]] = Field(
        default=None,
        description="Categories to search (defaults to knowledge repositories: news, wikipedia, academic, public_services. Pass 'business' explicitly to query Yellow Pages & commercial directory)",
    )
    top_k: int = Field(default=50, ge=1, le=500, description="Number of results to return (default: 50)")
    rerank: bool = Field(default=True, description="Enable hybrid semantic reranking")


@app.post("/api/search")
async def api_search_post(req: SearchRequest = Body(...)):
    """REST API endpoint taking raw text input (enforces strict 4.0s timeout budget)."""
    nlp_data = extract_search_terms(req.text)
    try:
        results = await asyncio.wait_for(
            asyncio.to_thread(
                _execute_search,
                text=req.text,
                categories=req.categories,
                top_k=req.top_k,
                rerank=req.rerank,
            ),
            timeout=MAX_SEARCH_SECONDS,
        )
    except asyncio.TimeoutError:
        results = []

    return {
        "status": "success",
        "input_text": req.text,
        "extracted_terms": {
            "clean_noun_chunks": nlp_data.get("clean_chunks", []),
            "lemma_chunks": nlp_data.get("lemma_chunks", []),
            "nouns": nlp_data.get("nouns", []),
            "primary_query": nlp_data.get("primary_query", ""),
        },
        "count": len(results),
        "requested_top_k": req.top_k,
        "results": [r.to_dict() for r in results],
    }


@app.get("/api/search")
async def api_search_get(
    text: Optional[str] = Query(None, description="Free-form text input"),
    query: Optional[str] = Query(None, description="Alias for text"),
    categories: Optional[str] = Query(None, description="Comma-separated categories"),
    top_k: int = Query(50, ge=1, le=500),
    rerank: bool = Query(True),
):
    """GET REST endpoint for quick URL queries (enforces strict 4.0s timeout budget)."""
    input_str = text or query or ""
    cat_list = [c.strip() for c in categories.split(",")] if categories else None
    try:
        results = await asyncio.wait_for(
            asyncio.to_thread(
                _execute_search,
                text=input_str,
                categories=cat_list,
                top_k=top_k,
                rerank=rerank,
            ),
            timeout=MAX_SEARCH_SECONDS,
        )
    except asyncio.TimeoutError:
        results = []

    nlp_data = extract_search_terms(input_str)
    return {
        "status": "success",
        "input_text": input_str,
        "extracted_terms": {
            "clean_noun_chunks": nlp_data.get("clean_chunks", []),
            "lemma_chunks": nlp_data.get("lemma_chunks", []),
            "nouns": nlp_data.get("nouns", []),
        },
        "count": len(results),
        "results": [r.to_dict() for r in results],
    }


class BusinessSearchRequest(BaseModel):
    query: str = Field(..., description="Product, commodity, service, or business name to search for")
    top_k: int = Field(default=15, ge=1, le=50, description="Number of business listings to return")


class PriceSearchRequest(BaseModel):
    product: str = Field(..., description="Product name or item to search prices for in Ghana")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum classifieds listings to sample")


@app.post("/api/business/search")
async def api_business_search(req: BusinessSearchRequest = Body(...)):
    """Search Ghana Business Directory & Yellow Pages for companies or product suppliers (enforces strict 4.0s timeout)."""
    try:
        results = await asyncio.wait_for(
            asyncio.to_thread(
                _execute_search,
                text=req.query,
                categories=["business"],
                top_k=req.top_k,
                rerank=True,
            ),
            timeout=MAX_SEARCH_SECONDS,
        )
    except asyncio.TimeoutError:
        results = []

    return {
        "status": "success",
        "query": req.query,
        "count": len(results),
        "results": [r.to_dict() for r in results],
    }


@app.post("/api/prices/search")
async def api_prices_search(req: PriceSearchRequest = Body(...)):
    """Search active classifieds listings in Ghana and compute real-time price metrics in Ghana Cedis (enforces strict 4.0s timeout)."""
    try:
        data = await asyncio.wait_for(
            asyncio.to_thread(
                price_finder.search_prices,
                product_query=req.product,
                limit=req.limit,
            ),
            timeout=MAX_SEARCH_SECONDS,
        )
    except asyncio.TimeoutError:
        data = {
            "query": req.product,
            "sample_size": 0,
            "average_price_ghs": None,
            "min_price_ghs": None,
            "max_price_ghs": None,
            "currency": "GHS (Ghana Cedi)",
            "listings": [],
        }

    return {
        "status": "success",
        "data": data,
    }


# --- Standard OpenAI-Compatible Tool Specifications for vLLM & SGLang ---

GHANA_SEARCH_OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "ghana_search",
            "description": (
                "Search verified Ghanaian national digital repositories including government ministries, "
                "departments and agencies (GRA, FDA, GSA, Lands Commission), district assemblies (261 MMDAs), "
                "court judgments and legislation (GhaLII), parliamentary bills, university thesis archives "
                "(KNUST, University of Ghana, UCC), and national news platforms. Returns factual titles, source agency, "
                "publication dates, excerpts, and direct links."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": (
                            "Natural language question or policy query regarding Ghana "
                            "(e.g., 'Ghana Revenue Authority VAT requirements', 'KNUST solar battery storage research', "
                            "'Supreme Court election petition judgments', 'Cocoa Marketing Board regulations')."
                        ),
                    },
                    "categories": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["news", "academic", "public_services", "wikipedia"],
                        },
                        "description": "Optional category filter. Defaults to all verified knowledge sources.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of relevant documents to return (default: 10, max: 50).",
                        "default": 10,
                    },
                },
                "required": ["text"],
            },
        },
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ghana_business_directory",
            "description": (
                "Search the Ghana Business Directory & Yellow Pages for commercial enterprises, companies, shops, and services. "
                "Works both ways: (1) Pass a product, commodity, or service (e.g. 'solar panels', 'fertilizer', 'tractor parts', "
                "'pharmacies in Kumasi') to get verified businesses that sell or supply them along with physical locations/addresses "
                "and phone numbers; (2) Pass a specific business or company name (e.g. 'Ernest Chemists', 'Solar 4 Ghana') to "
                "retrieve branch locations, telephone numbers, and profile details."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Product, commodity, service, or business name to search for (e.g. 'solar panels Accra', 'fertilizer', 'Ernest Chemists').",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of business listings to return (default: 10, max: 30).",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        },
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ghana_price_finder",
            "description": (
                "Search active Ghanaian marketplace classifieds to find product listings and calculate real-time market price "
                "metrics in Ghana Cedis (GH₵). Pass a product or item (e.g. 'MacBook Pro', 'solar inverter', 'car tracking system', "
                "'HP toner cartridge') to get active classifieds ads, average price (GH₵), price range (min/max), sample size, "
                "and direct listing links."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "Product or item name to price check in Ghana (e.g. 'MacBook Pro', 'solar inverter', 'Yamaha generator').",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of classified listings to sample (default: 10, max: 30).",
                        "default": 10,
                    },
                },
                "required": ["product"],
            },
        },
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    },
]


class ToolExecutionRequest(BaseModel):
    name: Optional[str] = Field("ghana_search", description="Tool name")
    arguments: Optional[Union[Dict[str, Any], str]] = Field(
        None,
        description="Tool arguments (either a JSON string as emitted by vLLM/SGLang tool callers, or a parsed dictionary)",
    )
    text: Optional[str] = Field(None, description="Direct search query")
    categories: Optional[List[str]] = Field(None, description="Direct category filter")
    top_k: Optional[int] = Field(10, description="Direct top_k count")


@app.get("/api/tools")
@app.get("/v1/tools")
async def api_tools():
    """Return standard OpenAI-compatible tool specifications for vLLM, SGLang, and agent runtimes."""
    return {
        "tools": GHANA_SEARCH_OPENAI_TOOLS,
        "format": "openai_compatible",
        "supported_frameworks": ["vllm", "sglang", "ollama", "litellm", "langchain"],
    }


@app.post("/api/tools/execute")
@app.post("/v1/tools/execute")
async def api_tools_execute(req: ToolExecutionRequest = Body(...)):
    """Execute a tool call emitted by vLLM, SGLang, or any OpenAI-compatible function calling model.
    Accepts tool arguments either as a parsed dictionary or as a raw JSON string (as returned in
    tool_call.function.arguments). Supports 'ghana_search', 'ghana_business_directory', and 'ghana_price_finder'.
    """
    args = {}
    if req.arguments:
        if isinstance(req.arguments, str):
            try:
                args = json.loads(req.arguments)
            except Exception:
                args = {"text": req.arguments, "query": req.arguments, "product": req.arguments}
        elif isinstance(req.arguments, dict):
            args = req.arguments

    tool_name = req.name or "ghana_search"

    # --- 1. Price Finder Tool Execution ---
    if tool_name == "ghana_price_finder":
        product_query = args.get("product") or args.get("query") or args.get("text") or req.text or ""
        limit = int(args.get("limit") or 10)
        data = await asyncio.to_thread(price_finder.search_prices, product_query=product_query, limit=limit)
        listings = data.get("listings", [])
        avg = data.get("average_price_ghs")
        min_p = data.get("min_price_ghs")
        max_p = data.get("max_price_ghs")

        lines = [f"Ghana Marketplace Price Summary for '{product_query}':"]
        if avg is not None:
            lines.append(f"• Average Market Price: GH₵{avg:,.2f}")
            lines.append(f"• Price Range: GH₵{min_p:,.2f} – GH₵{max_p:,.2f}")
            lines.append(f"• Sampled Classifieds: {len(listings)}\n")
        else:
            lines.append("• No exact active classifieds listings found for this product.\n")

        if listings:
            lines.append("Active Classifieds Listings:")
            for idx, item in enumerate(listings, 1):
                lines.append(f"[{idx}] {item['title']} — GH₵{item['price_ghs']:,.2f} (Location: {item.get('location', 'Ghana')})\nURL: {item['url']}")

        llm_context_output = "\n".join(lines)
        return {
            "status": "success",
            "tool_name": tool_name,
            "product": product_query,
            "data": data,
            "output": llm_context_output,
            "content": llm_context_output,
        }

    # --- 2. Business Directory Tool Execution ---
    if tool_name in ("ghana_business_directory", "ghana_business_search"):
        biz_query = args.get("query") or args.get("text") or req.text or ""
        top_k = int(args.get("top_k") or req.top_k or 15)
        results = await asyncio.to_thread(_execute_search, text=biz_query, categories=["business"], top_k=top_k, rerank=True)
        lines = [f"Found {len(results)} verified Ghanaian business listings for '{biz_query}':\n"]
        for idx, r in enumerate(results, 1):
            lines.append(f"[{idx}] {r.title}\n{r.snippet}\nDirectory Profile: {r.url}")

        llm_context_output = "\n\n".join(lines) if results else f"No verified Ghanaian businesses found matching '{biz_query}'."
        return {
            "status": "success",
            "tool_name": tool_name,
            "query": biz_query,
            "count": len(results),
            "output": llm_context_output,
            "content": llm_context_output,
            "results": [r.to_dict() for r in results],
        }

    # --- 3. National Knowledge Search (Default) ---
    query_text = args.get("text") or req.text or ""
    if not query_text.strip():
        return {
            "status": "error",
            "message": "Missing required 'text' parameter in tool arguments",
            "output": "No search query provided.",
            "content": "No search query provided.",
            "results": [],
        }

    raw_top_k = args.get("top_k") or req.top_k or 10
    try:
        top_k = max(1, min(int(raw_top_k), 50))
    except (ValueError, TypeError):
        top_k = 10

    categories = args.get("categories") or req.categories

    try:
        results = await asyncio.wait_for(
            asyncio.to_thread(
                _execute_search,
                text=query_text,
                categories=categories,
                top_k=top_k,
                rerank=True,
                client_type="inference_tool_caller",
            ),
            timeout=MAX_SEARCH_SECONDS,
        )
    except asyncio.TimeoutError:
        results = []

    formatted_chunks = []
    for idx, r in enumerate(results, 1):
        snippet = r.content if r.content and len(r.content) > len(r.snippet) else r.snippet
        if snippet and len(snippet) > 800:
            snippet = snippet[:800] + "..."
        date_str = f" | Date: {r.published_date}" if r.published_date else ""
        pdf_str = f" | PDF: {r.metadata.get('pdf_url')}" if r.metadata.get("pdf_url") else ""
        formatted_chunks.append(
            f"[{idx}] {r.title}\n"
            f"Source: {r.source}{date_str}{pdf_str}\n"
            f"URL: {r.url}\n"
            f"Content: {snippet}"
        )

    llm_context_output = (
        f"Retrieved {len(results)} verified Ghanaian records for query '{query_text}':\n\n"
        + "\n\n".join(formatted_chunks)
        if formatted_chunks
        else f"No verified Ghanaian records found for query '{query_text}'."
    )

    return {
        "status": "success",
        "tool_name": tool_name,
        "query": query_text,
        "count": len(results),
        "output": llm_context_output,
        "content": llm_context_output,
        "results": [r.to_dict() for r in results],
    }


@app.get("/api/analytics/stats")
async def api_analytics_stats():
    """Return live usage statistics, query latency, cache hit rate, and trending topics."""
    return analytics.get_stats()


@app.get("/api/analytics/dataset")
async def api_analytics_dataset(
    format: str = Query("json", description="Export format: 'json' or 'jsonl'"),
    limit: int = Query(500, ge=1, le=5000),
):
    """Export logged evaluation dataset for offline testing and model improvement."""
    records = analytics.export_dataset(limit=limit)
    if format == "jsonl":
        lines = [json.dumps(r, ensure_ascii=False) for r in records]
        return PlainTextResponse(
            "\n".join(lines),
            media_type="application/x-ndjson",
            headers={"Content-Disposition": "attachment; filename=query_evaluation_dataset.jsonl"},
        )
    return {
        "status": "success",
        "count": len(records),
        "records": records,
    }


@app.post("/api/analytics/backup-hf")
async def api_backup_hf():
    """Trigger an immediate background backup push of the permanent URL cache to Hugging Face."""
    result = await asyncio.to_thread(hf_sync.push_to_huggingface)
    return result


@app.get("/api/health")
async def api_health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "ghana-search",
        "version": "1.0.0",
        "nlp_engine": "spaCy (en_core_web_sm)",
        "reranker": "100% CPU Title-Only & Recent-First Ranker (Zero GPU / 0MB VRAM)",
        "caching": "Two-Tier (24h Query Cache + Permanent URL Content Cache)",
        "offline_mode": OFFLINE_MODE,
        "offline_records": search_cache.get_url_cache_stats()["total_permanent_urls"],
        "workers": int(os.environ.get("WEB_CONCURRENCY", min(4, max(1, os.cpu_count() or 1)))),
        "cpu_cores": os.cpu_count(),
        "default_max_wait_seconds": 4.0,
        "evaluation_dataset": "Enabled (JSONL export)",
        "hf_backup_repo": "ghanaopenai/ghana-search-corpus",
        "organization": "Ghana Open AI",
        "endpoints": {
            "mcp_sse": "/mcp/sse",
            "rest_search": "/api/search",
            "rest_business_search": "/api/business/search",
            "rest_price_search": "/api/prices/search",
            "tools_schema": "/api/tools",
            "tools_execute": "/api/tools/execute",
            "analytics_stats": "/api/analytics/stats",
            "analytics_dataset": "/api/analytics/dataset",
            "backup_hf": "/api/analytics/backup-hf",
            "web_ui": "/",
        },
    }


static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", response_class=HTMLResponse)
    @app.head("/")
    @app.get("/demo", response_class=HTMLResponse)
    @app.head("/demo")
    async def serve_index():
        index_path = os.path.join(static_dir, "index.html")
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        return HTMLResponse("<h1>Ghana Search Server</h1>")

    @app.get("/favicon.ico")
    async def serve_favicon():
        favicon_path = os.path.join(static_dir, "images", "favicon.ico")
        if os.path.exists(favicon_path):
            return FileResponse(favicon_path, media_type="image/x-icon")
        return Response(status_code=404)


def run():
    """CLI Entrypoint to start the server with automatic multi-core worker scaling."""
    if "--stdio" in sys.argv:
        asyncio.run(mcp.run_stdio_async())
    else:
        port = int(os.environ.get("PORT", SERVER_PORT))
        host = os.environ.get("HOST", SERVER_HOST)
        # Default to 4 workers on multi-core environments, or scale to available CPU cores
        default_workers = min(4, max(1, os.cpu_count() or 1))
        workers = int(os.environ.get("WEB_CONCURRENCY", default_workers))
        print(f"🚀 Starting Ghana Search Engine on http://{host}:{port} with {workers} worker processes")
        print(f"   MCP SSE Endpoint: http://{host}:{port}/mcp/sse")
        print(f"   REST Search API:  http://{host}:{port}/api/search")
        print(f"   Interactive UI:   http://{host}:{port}/")
        uvicorn.run("ghana_mcp.server:app", host=host, port=port, workers=workers)


if __name__ == "__main__":
    run()
