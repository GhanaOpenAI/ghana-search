"""Two-Tier Caching Engine for Ghana Search:
1. Query-Level 24-Hour Cache: Caches ranked search result sets per query/keywords with a 24-hour TTL.
2. Permanent URL Content Cache: Permanently caches full document titles, clean content bodies,
   dates, and metadata per unique URL so content is never re-fetched unnecessarily.
"""
import os
import json
import time
import hashlib
import sqlite3
from typing import List, Dict, Any, Optional

from ghana_mcp.providers.base import SearchResult

CACHE_DIR = os.path.expanduser("~/.cache/ghana_mcp")
DB_PATH = os.environ.get("GHANA_SEARCH_DB", os.path.join(CACHE_DIR, "search_cache.db"))
DEFAULT_QUERY_TTL = 86400  # 24 hours in seconds


class QueryCache:
    def __init__(self, db_path: str = DB_PATH, query_ttl: int = DEFAULT_QUERY_TTL):
        self.db_path = db_path
        self.query_ttl = query_ttl
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self._max_memory_items = 1000

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            # 1. Query-level cache (24-Hour TTL)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS search_cache (
                    cache_key TEXT PRIMARY KEY,
                    normalized_query TEXT,
                    categories TEXT,
                    top_k INTEGER,
                    results_json TEXT,
                    created_at REAL,
                    expires_at REAL,
                    hit_count INTEGER DEFAULT 0
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_expires ON search_cache(expires_at);")

            # 2. Permanent URL Content Cache (No expiration: stores full content permanently)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS url_content_cache (
                    url TEXT PRIMARY KEY,
                    title TEXT,
                    snippet TEXT,
                    content TEXT,
                    source TEXT,
                    category TEXT,
                    published_date TEXT,
                    metadata_json TEXT,
                    first_seen REAL,
                    last_accessed REAL,
                    access_count INTEGER DEFAULT 1
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_url_cache_source ON url_content_cache(source);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_url_cache_category ON url_content_cache(category);")

        # Automatically seed bundled offline knowledge packs into url_content_cache on startup
        self._seed_bundled_offline_packs()

    def _seed_bundled_offline_packs(self):
        """Automatically seed all bundled offline knowledge packs into url_content_cache on startup."""
        seed_files = []
        # Check ghana_mcp/data/
        pkg_data_dir = os.path.join(os.path.dirname(__file__), "data")
        if os.path.isdir(pkg_data_dir):
            for fname in os.listdir(pkg_data_dir):
                if fname.endswith(".jsonl"):
                    seed_files.append(os.path.join(pkg_data_dir, fname))

        # Check repository data/ directory
        repo_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        if os.path.isdir(repo_data_dir):
            for fname in os.listdir(repo_data_dir):
                if fname.endswith(".jsonl"):
                    fpath = os.path.join(repo_data_dir, fname)
                    if fpath not in seed_files:
                        seed_files.append(fpath)

        if not seed_files:
            return

        now = time.time()
        items_to_insert = []
        seen_urls = set()

        for sfile in seed_files:
            try:
                with open(sfile, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        doc = json.loads(line)
                        raw_url = doc.get("url", "").strip()
                        norm_url = raw_url.rstrip("/")
                        if not norm_url or norm_url in seen_urls:
                            continue
                        seen_urls.add(norm_url)
                        meta_json = json.dumps(doc.get("metadata") or {})
                        items_to_insert.append((
                            raw_url,
                            doc.get("title") or "",
                            doc.get("snippet") or "",
                            doc.get("content") or doc.get("snippet") or "",
                            doc.get("source") or "",
                            doc.get("category") or "",
                            doc.get("published_date") or "",
                            meta_json,
                            now,
                            now,
                        ))
            except Exception:
                continue

        if items_to_insert:
            try:
                with self._get_connection() as conn:
                    conn.executemany(
                        """
                        INSERT INTO url_content_cache (
                            url, title, snippet, content, source, category,
                            published_date, metadata_json, first_seen, last_accessed, access_count
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                        ON CONFLICT(url) DO UPDATE SET
                            content = CASE WHEN length(excluded.content) > length(url_content_cache.content) THEN excluded.content ELSE url_content_cache.content END
                        """,
                        items_to_insert,
                    )
            except Exception:
                pass

    # --- Tier 1: Query-Level 24-Hour Cache ---

    def _make_key(self, text: str, categories: Optional[List[str]], top_k: int) -> str:
        norm_tokens = sorted(list(set(text.lower().strip().split())))
        cats = sorted(categories) if categories else ["all"]
        key_raw = f"v3|{' '.join(norm_tokens)}|{','.join(cats)}|{top_k}"
        return hashlib.sha256(key_raw.encode("utf-8")).hexdigest()

    def get(
        self, text: str, categories: Optional[List[str]], top_k: int
    ) -> Optional[List[SearchResult]]:
        """Retrieve 24-hour cached query results if available."""
        cache_key = self._make_key(text, categories, top_k)
        now = time.time()

        # Check in-memory LRU cache
        if cache_key in self._memory_cache:
            entry = self._memory_cache[cache_key]
            if entry["expires_at"] > now:
                entry["hit_count"] += 1
                return self._deserialize_results(entry["results"], age=now - entry["created_at"])
            else:
                del self._memory_cache[cache_key]

        # Check SQLite WAL cache
        try:
            with self._get_connection() as conn:
                row = conn.execute(
                    """
                    SELECT results_json, created_at, expires_at, hit_count
                    FROM search_cache
                    WHERE cache_key = ? AND expires_at > ?
                    """,
                    (cache_key, now),
                ).fetchone()

                if row:
                    conn.execute(
                        "UPDATE search_cache SET hit_count = hit_count + 1 WHERE cache_key = ?",
                        (cache_key,),
                    )
                    results_data = json.loads(row["results_json"])
                    created_at = row["created_at"]

                    if len(self._memory_cache) >= self._max_memory_items:
                        self._memory_cache.clear()
                    self._memory_cache[cache_key] = {
                        "results": results_data,
                        "created_at": created_at,
                        "expires_at": row["expires_at"],
                        "hit_count": row["hit_count"] + 1,
                    }

                    return self._deserialize_results(results_data, age=now - created_at)
        except Exception:
            pass

        return None

    def set(
        self,
        text: str,
        categories: Optional[List[str]],
        top_k: int,
        results: List[SearchResult],
    ):
        """Store query results for 24 hours."""
        if not results:
            return

        cache_key = self._make_key(text, categories, top_k)
        now = time.time()
        expires_at = now + self.query_ttl
        results_data = [r.to_dict() for r in results]
        results_json = json.dumps(results_data)

        if len(self._memory_cache) >= self._max_memory_items:
            self._memory_cache.clear()
        self._memory_cache[cache_key] = {
            "results": results_data,
            "created_at": now,
            "expires_at": expires_at,
            "hit_count": 0,
        }

        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO search_cache (
                        cache_key, normalized_query, categories, top_k,
                        results_json, created_at, expires_at, hit_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 0)
                    ON CONFLICT(cache_key) DO UPDATE SET
                        results_json = excluded.results_json,
                        created_at = excluded.created_at,
                        expires_at = excluded.expires_at
                    """,
                    (
                        cache_key,
                        text.lower().strip(),
                        json.dumps(categories or []),
                        top_k,
                        results_json,
                        now,
                        expires_at,
                    ),
                )
        except Exception:
            pass

    # --- Tier 2: Permanent URL Content Cache ---

    def get_url(self, url: str) -> Optional[SearchResult]:
        """Retrieve a permanently cached document by its URL."""
        if not url:
            return None
        norm_url = url.strip().rstrip("/")
        try:
            with self._get_connection() as conn:
                row = conn.execute(
                    """
                    SELECT url, title, snippet, content, source, category,
                           published_date, metadata_json, access_count
                    FROM url_content_cache WHERE url = ?
                    """,
                    (norm_url,),
                ).fetchone()

                if row:
                    conn.execute(
                        """
                        UPDATE url_content_cache
                        SET access_count = access_count + 1, last_accessed = ?
                        WHERE url = ?
                        """,
                        (time.time(), norm_url),
                    )
                    meta = json.loads(row["metadata_json"] or "{}")
                    meta["url_cached_permanent"] = True
                    return SearchResult(
                        title=row["title"],
                        url=row["url"],
                        snippet=row["snippet"],
                        source=row["source"],
                        category=row["category"],
                        content=row["content"],
                        published_date=row["published_date"],
                        metadata=meta,
                    )
        except Exception:
            pass
        return None

    def get_urls_batch(self, urls: List[str]) -> Dict[str, SearchResult]:
        """Retrieve permanently cached documents for a list of URLs in a single query."""
        if not urls:
            return {}

        clean_map = {u.strip().rstrip("/"): u for u in urls if u}
        keys = list(clean_map.keys())
        if not keys:
            return {}

        found = {}
        try:
            with self._get_connection() as conn:
                placeholders = ",".join("?" for _ in keys)
                rows = conn.execute(
                    f"""
                    SELECT url, title, snippet, content, source, category,
                           published_date, metadata_json, access_count
                    FROM url_content_cache WHERE url IN ({placeholders})
                    """,
                    keys,
                ).fetchall()

                now = time.time()
                for row in rows:
                    orig_url = clean_map.get(row["url"], row["url"])
                    meta = json.loads(row["metadata_json"] or "{}")
                    meta["url_cached_permanent"] = True
                    found[orig_url] = SearchResult(
                        title=row["title"],
                        url=row["url"],
                        snippet=row["snippet"],
                        source=row["source"],
                        category=row["category"],
                        content=row["content"],
                        published_date=row["published_date"],
                        metadata=meta,
                    )

                # Batch update access timestamps
                if found:
                    conn.execute(
                        f"""
                        UPDATE url_content_cache
                        SET access_count = access_count + 1, last_accessed = ?
                        WHERE url IN ({placeholders})
                        """,
                        [now] + keys,
                    )
        except Exception:
            pass

        return found

    def set_urls_batch(self, results: List[SearchResult]):
        """Permanently store document content, titles, and dates per URL."""
        if not results:
            return

        now = time.time()
        items_to_save = []
        for r in results:
            if not r.url:
                continue
            norm_url = r.url.strip().rstrip("/")
            meta_json = json.dumps(r.metadata or {})
            body_content = r.content or r.snippet or ""
            items_to_save.append((
                norm_url,
                r.title or "",
                r.snippet or "",
                body_content,
                r.source or "",
                r.category or "",
                r.published_date or "",
                meta_json,
                now,
                now,
            ))

        try:
            with self._get_connection() as conn:
                conn.executemany(
                    """
                    INSERT INTO url_content_cache (
                        url, title, snippet, content, source, category,
                        published_date, metadata_json, first_seen, last_accessed, access_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                    ON CONFLICT(url) DO UPDATE SET
                        title = excluded.title,
                        snippet = excluded.snippet,
                        content = CASE WHEN length(excluded.content) > length(url_content_cache.content) THEN excluded.content ELSE url_content_cache.content END,
                        source = excluded.source,
                        published_date = CASE WHEN excluded.published_date != '' THEN excluded.published_date ELSE url_content_cache.published_date END,
                        metadata_json = excluded.metadata_json,
                        last_accessed = excluded.last_accessed,
                        access_count = url_content_cache.access_count + 1
                    """,
                    items_to_save,
                )
        except Exception:
            pass

    def get_url_cache_stats(self) -> Dict[str, Any]:
        """Return statistics on permanently cached URLs."""
        try:
            with self._get_connection() as conn:
                total_urls = conn.execute("SELECT COUNT(*) FROM url_content_cache").fetchone()[0]
                sources = dict(conn.execute("SELECT source, COUNT(*) FROM url_content_cache GROUP BY source").fetchall())
                return {"total_permanent_urls": total_urls, "sources_stored": sources}
        except Exception:
            return {"total_permanent_urls": 0, "sources_stored": {}}

    def _deserialize_results(self, items: List[Dict[str, Any]], age: float) -> List[SearchResult]:
        results = []
        for it in items:
            meta = dict(it.get("metadata", {}))
            meta["cached"] = True
            meta["cache_age_seconds"] = round(age, 1)
            results.append(
                SearchResult(
                    title=it["title"],
                    url=it["url"],
                    snippet=it["snippet"],
                    source=it["source"],
                    category=it["category"],
                    content=it.get("content"),
                    published_date=it.get("published_date"),
                    score=it.get("score"),
                    metadata=meta,
                )
            )
        return results

    def search_offline(
        self, query_tokens: List[str], categories: Optional[List[str]] = None, limit: int = 50
    ) -> List[SearchResult]:
        """Search permanently cached documents completely offline via title token matches."""
        if not query_tokens:
            return []

        # Split chunks into individual word tokens for flexible title matching
        words = set()
        for tok in query_tokens:
            for w in tok.strip().split():
                cleaned = w.lower().strip()
                if len(cleaned) >= 2:
                    words.add(cleaned)

        if not words:
            return []

        conditions = []
        params: List[Any] = []
        for w in list(words)[:12]:
            conditions.append("title LIKE ?")
            params.append(f"%{w}%")

        where_clause = " OR ".join(conditions)
        cat_clause = ""
        if categories:
            cat_placeholders = ",".join("?" for _ in categories)
            cat_clause = f" AND category IN ({cat_placeholders})"
            params.extend(categories)

        sql = f"""
            SELECT url, title, snippet, content, source, category, published_date, metadata_json
            FROM url_content_cache
            WHERE ({where_clause}) {cat_clause}
            ORDER BY access_count DESC
            LIMIT {limit * 3}
        """

        results: List[SearchResult] = []
        try:
            with self._get_connection() as conn:
                rows = conn.execute(sql, params).fetchall()
                for row in rows:
                    meta = json.loads(row["metadata_json"] or "{}")
                    meta["offline_mode"] = True
                    results.append(
                        SearchResult(
                            title=row["title"],
                            url=row["url"],
                            snippet=row["snippet"],
                            source=row["source"],
                            category=row["category"],
                            content=row["content"],
                            published_date=row["published_date"],
                            metadata=meta,
                        )
                    )
        except Exception:
            pass

        return results

    def clear_expired(self):
        """Remove query cache entries older than 24 hours (preserves permanent URL content cache)."""
        now = time.time()
        try:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM search_cache WHERE expires_at <= ?", (now,))
        except Exception:
            pass


# Global cache instance
search_cache = QueryCache()
