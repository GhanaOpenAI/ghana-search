"""Ghanaian Entity Articles Provider.
Indexes 506,415 comprehensive encyclopedia articles across 253,290 Ghanaian named entities
from ghanaopenai/ghana-entity-articles (generated with Gemini 3.8 Live API).
Provides in-depth, factual Wikipedia-style articles with instant sub-millisecond retrieval.
100% offline ready via pre-indexed compressed SQLite database.
"""
import os
import sqlite3
import zlib
import urllib.request
import logging
from typing import List, Optional

from ghana_mcp.providers.base import BaseProvider, SearchResult
from ghana_mcp.nlp import extract_search_terms

logger = logging.getLogger("ghana_search.entity_articles")

DEFAULT_ARTICLES_DB = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "entity_articles.db"
)
HF_ARTICLES_DB_URL = "https://huggingface.co/datasets/ghanaopenai/ghana-entity-articles/resolve/main/entity_articles.db"


def ensure_entity_articles_db(db_path: str = DEFAULT_ARTICLES_DB) -> bool:
    """Download entity_articles.db from Hugging Face if not already present."""
    if os.path.exists(db_path) and os.path.getsize(db_path) > 100 * 1024 * 1024:
        return True

    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    tmp_path = db_path + ".tmp"
    logger.info(f"Downloading entity_articles.db from {HF_ARTICLES_DB_URL}...")
    print(f"Downloading Ghana entity articles database (~438MB) from {HF_ARTICLES_DB_URL}...")

    req = urllib.request.Request(
        HF_ARTICLES_DB_URL,
        headers={"User-Agent": "ghana-search/1.0 (Ghana Open AI; https://search.ghanaopenai.org)"},
    )

    try:
        with urllib.request.urlopen(req, timeout=180.0) as resp, open(tmp_path, "wb") as f:
            total_size = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1MB chunks

            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0 and downloaded % (25 * 1024 * 1024) < chunk_size:
                    pct = (downloaded / total_size) * 100
                    print(f"Downloaded {downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB ({pct:.1f}%)...")

        os.replace(tmp_path, db_path)
        print(f"Successfully downloaded entity_articles.db ({os.path.getsize(db_path)/(1024*1024):.1f} MB)")
        return True
    except Exception as e:
        logger.warning(f"Could not download entity_articles.db: {e}")
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        return False


class EntityArticlesProvider(BaseProvider):
    name = "ghana_entity_articles"
    category = "entity_articles"

    def __init__(self, db_path: str = DEFAULT_ARTICLES_DB):
        self.db_path = db_path

    def search(self, keywords: List[str], limit: int = 2) -> List[SearchResult]:
        if not os.path.exists(self.db_path):
            return []

        raw_query = " ".join(keywords).strip()
        if not raw_query:
            return []

        nlp_data = extract_search_terms(raw_query)
        chunks = [c.lower().strip() for c in nlp_data.get("clean_chunks", [])]
        nouns = [n.lower().strip() for n in nlp_data.get("nouns", [])]
        terms = [raw_query.lower().strip()] + chunks + nouns

        seen_terms = set()
        uniq_terms = [t for t in terms if len(t) >= 2 and not (t in seen_terms or seen_terms.add(t))]
        if not uniq_terms:
            return []

        results: List[SearchResult] = []
        found_titles = set()

        try:
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, timeout=1.0)
            conn.row_factory = sqlite3.Row

            placeholders = ",".join("?" for _ in uniq_terms)
            sql = f"""
                SELECT entity, article_id, title, article_blob
                FROM entity_articles
                WHERE norm_entity IN ({placeholders})
                ORDER BY CASE WHEN norm_entity = ? THEN 0 ELSE 1 END, article_id ASC
                LIMIT {limit}
            """
            for row in conn.execute(sql, uniq_terms + [uniq_terms[0]]).fetchall():
                t = row["title"] or f"{row['entity']} - Article {row['article_id']}"
                if t not in found_titles:
                    found_titles.add(t)
                    article_text = zlib.decompress(row["article_blob"]).decode("utf-8")
                    snippet = article_text[:350].strip() + "..."
                    results.append(
                        SearchResult(
                            title=t,
                            url="",
                            snippet=snippet,
                            source="Ghana Entity Encyclopedia",
                            category=self.category,
                            content=article_text,
                            published_date="",
                            score=420.0,
                            metadata={
                                "type": "entity_article",
                                "entity": row["entity"],
                                "article_id": row["article_id"],
                            },
                        )
                    )
            conn.close()
        except Exception as e:
            logger.warning(f"Error querying entity_articles.db: {e}")

        return results
