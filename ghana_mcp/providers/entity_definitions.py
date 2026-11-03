"""Ghanaian Named Entity Definitions Provider.
Indexes 104,000+ curated Ghanaian named entities (political leaders, institutions,
traditions, landmarks, towns, festivals, and cultural concepts) from Ghana NLP
to instantly ground searches with factual Ghanaian definitions both offline and online.
"""
import os
import sqlite3
from typing import List, Dict, Any, Optional

from ghana_mcp.providers.base import BaseProvider, SearchResult
from ghana_mcp.nlp import extract_search_terms

DEFAULT_ENTITIES_DB = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "entity_definitions.db")


class EntityDefinitionsProvider(BaseProvider):
    name = "ghana_entities"
    category = "named_entities"

    def __init__(self, db_path: str = DEFAULT_ENTITIES_DB):
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

        results: List[SearchResult] = []
        found_phrases = set()

        try:
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, timeout=2.0)
            conn.row_factory = sqlite3.Row

            # 1. Exact match on full query, clean noun chunks, or key nouns
            placeholders = ",".join("?" for _ in uniq_terms)
            sql1 = f"""
                SELECT phrase, count, description
                FROM entity_definitions
                WHERE norm_phrase IN ({placeholders})
                ORDER BY count DESC
                LIMIT {limit}
            """
            for row in conn.execute(sql1, uniq_terms).fetchall():
                p = row["phrase"]
                if p not in found_phrases:
                    found_phrases.add(p)
                    results.append(
                        SearchResult(
                            title=f"{p} (Ghanaian Entity Definition)",
                            url="",
                            snippet=row["description"],
                            source="Ghana Named Entities",
                            category=self.category,
                            content=row["description"],
                            published_date="",
                            score=350.0,
                            metadata={
                                "type": "entity_definition",
                                "phrase": p,
                                "frequency": row["count"],
                            },
                        )
                    )

            # 2. Substring match if exact match didn't yield enough definitions
            if len(results) < limit and nouns:
                for n in nouns[:2]:
                    if len(n) <= 3 and n.lower() not in ("ecg", "gra", "ndc", "npp", "gfa", "nia"):
                        continue
                    sql2 = """
                        SELECT phrase, count, description
                        FROM entity_definitions
                        WHERE norm_phrase LIKE ?
                        ORDER BY count DESC
                        LIMIT ?
                    """
                    for row in conn.execute(sql2, (f"%{n}%", limit - len(results))).fetchall():
                        p = row["phrase"]
                        if p not in found_phrases:
                            found_phrases.add(p)
                            results.append(
                                SearchResult(
                                    title=f"{p} (Ghanaian Entity Definition)",
                                    url="",
                                    snippet=row["description"],
                                    source="Ghana Named Entities",
                                    category=self.category,
                                    content=row["description"],
                                    published_date="",
                                    score=320.0,
                                    metadata={
                                        "type": "entity_definition",
                                        "phrase": p,
                                        "frequency": row["count"],
                                    },
                                )
                            )
                    if len(results) >= limit:
                        break

            conn.close()
        except Exception:
            pass

        return results
