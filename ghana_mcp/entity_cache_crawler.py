"""Ghanaian Named Entities Autonomous Cache Crawler & Hugging Face Sync Daemon.
Crawls Ghanaian named entities from 'ghananlpcommunity/ghana-named-entities' in order of
frequency, queries Ghana Search engine with a 5.0s max timeout, stores full content behind
matching URLs into SQLite, and periodically pushes batch updates to Hugging Face:
'ghanaopenai/ghana-search-corpus'.

Designed to run continuously on the H200 in the background with automatic checkpointing,
graceful resumption, and error tolerance.
"""
import os
import sys
import json
import time
import signal
import sqlite3
import argparse
from typing import Dict, Any, List, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from huggingface_hub import HfApi
except ImportError:
    HfApi = None

from ghana_mcp.providers.base import SearchResult
from ghana_mcp.cache import QueryCache, search_cache
from ghana_mcp.server import _execute_search

DEFAULT_DATASET = "ghananlpcommunity/ghana-named-entities"
DEFAULT_HF_REPO = "ghanaopenai/ghana-search-corpus"
DEFAULT_CHECKPOINT_FILE = "data/crawler_checkpoint.json"
DEFAULT_CORPUS_JSONL = "data/ghana_search_corpus.jsonl"
DEFAULT_OFFLINE_SEED = "ghana_mcp/data/offline_seed.jsonl"


class EntityCacheCrawler:
    def __init__(
        self,
        dataset_name: str = DEFAULT_DATASET,
        hf_repo_id: str = DEFAULT_HF_REPO,
        batch_size: int = 50,
        max_wait_seconds: float = 5.0,
        workers: int = 3,
        limit: Optional[int] = None,
        checkpoint_file: str = DEFAULT_CHECKPOINT_FILE,
        push_to_hf: bool = True,
    ):
        self.dataset_name = dataset_name
        self.hf_repo_id = hf_repo_id
        self.batch_size = batch_size
        self.max_wait = max_wait_seconds
        self.workers = workers
        self.limit = limit
        self.checkpoint_file = checkpoint_file
        self.push_to_hf = push_to_hf
        self.running = True

        os.makedirs(os.path.dirname(os.path.abspath(checkpoint_file)), exist_ok=True)
        self._init_checkpoint_db()

    def _get_hf_token(self) -> Optional[str]:
        env_tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if env_tok:
            return env_tok.strip()
        tok_file = os.path.expanduser("~/.cache/huggingface/token")
        if os.path.exists(tok_file):
            try:
                with open(tok_file, "r", encoding="utf-8") as f:
                    return f.read().strip()
            except Exception:
                pass
        return None

    def _init_checkpoint_db(self):
        """Initialize SQLite table to track processed entities across runs."""
        with search_cache._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS crawler_processed (
                    phrase TEXT PRIMARY KEY,
                    count INTEGER,
                    urls_found INTEGER,
                    processed_at REAL
                );
            """)

    def _load_seen_phrases(self) -> Set[str]:
        with search_cache._get_connection() as conn:
            rows = conn.execute("SELECT phrase FROM crawler_processed").fetchall()
            return {r[0].lower().strip() for r in rows}

    def _record_processed(self, phrase: str, count: int, urls_found: int):
        with search_cache._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO crawler_processed (phrase, count, urls_found, processed_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(phrase) DO UPDATE SET
                    urls_found = excluded.urls_found,
                    processed_at = excluded.processed_at
                """,
                (phrase.strip(), count, urls_found, time.time()),
            )

    def _save_checkpoint_stats(self, processed_total: int, last_phrase: str):
        stats = search_cache.get_url_cache_stats()
        checkpoint = {
            "processed_entities": processed_total,
            "last_processed_phrase": last_phrase,
            "total_permanent_urls": stats.get("total_permanent_urls", 0),
            "sources_breakdown": stats.get("sources_stored", {}),
            "updated_at": time.time(),
            "updated_at_iso": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        }
        with open(self.checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2)

    def export_and_push_to_hf(self, batch_idx: int) -> bool:
        """Export all permanently cached documents to JSONL and push to Hugging Face dataset."""
        stats = search_cache.get_url_cache_stats()
        total_urls = stats.get("total_permanent_urls", 0)
        if total_urls == 0:
            print("[*] No documents in cache yet, skipping HF push.")
            return False

        print(f"\n[*] Exporting {total_urls} permanently cached documents to JSONL...")
        os.makedirs(os.path.dirname(os.path.abspath(DEFAULT_CORPUS_JSONL)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(DEFAULT_OFFLINE_SEED)), exist_ok=True)

        rows = []
        with search_cache._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT url, title, snippet, content, source, category, published_date, metadata_json
                FROM url_content_cache
                ORDER BY access_count DESC, first_seen DESC
                """
            ).fetchall()

        # Write to both the root corpus JSONL and the bundled offline seed JSONL
        for target_path in (DEFAULT_CORPUS_JSONL, DEFAULT_OFFLINE_SEED):
            try:
                with open(target_path, "w", encoding="utf-8") as f:
                    for r in rows:
                        record = {
                            "url": r[0],
                            "title": r[1],
                            "snippet": r[2],
                            "content": r[3],
                            "source": r[4],
                            "category": r[5],
                            "published_date": r[6],
                            "metadata": json.loads(r[7] or "{}"),
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[-] Error writing {target_path}: {e}")

        file_size_kb = os.path.getsize(DEFAULT_CORPUS_JSONL) / 1024
        print(f"[+] Exported {len(rows)} documents ({file_size_kb:.1f} KB)")

        if not self.push_to_hf:
            return True

        if HfApi is None:
            print("[-] huggingface_hub is not installed, skipping remote upload.")
            return False

        token = self._get_hf_token()
        if not token:
            print("[-] No Hugging Face token found, skipping remote upload.")
            return False

        try:
            print(f"[*] Pushing batch {batch_idx} to Hugging Face dataset '{self.hf_repo_id}'...")
            api = HfApi(token=token)
            api.create_repo(repo_id=self.hf_repo_id, repo_type="dataset", exist_ok=True)

            api.upload_file(
                path_or_fileobj=DEFAULT_CORPUS_JSONL,
                path_in_repo="ghana_search_corpus.jsonl",
                repo_id=self.hf_repo_id,
                repo_type="dataset",
                commit_message=f"Sync offline corpus: {total_urls} documents (Batch {batch_idx})",
            )
            print(f"[+] Successfully pushed {total_urls} documents to {self.hf_repo_id}!")
            return True
        except Exception as e:
            print(f"[-] Hugging Face upload failed: {e}")
            return False

    def run(self):
        """Execute the continuous crawler."""
        print("=" * 70)
        print("GHANA SEARCH: NAMED ENTITY OFFLINE CACHE CRAWLER")
        print("=" * 70)
        print(f"Dataset:            {self.dataset_name}")
        print(f"Target HF Repo:     {self.hf_repo_id}")
        print(f"Max Wait Per Query: {self.max_wait}s")
        print(f"Push Batch Size:    Every {self.batch_size} entities")
        print(f"Concurrency:        {self.workers} worker threads")
        print("=" * 70)

        # 1. Load seen phrases from previous runs
        seen_phrases = self._load_seen_phrases()
        print(f"[*] Resuming checkpoint: {len(seen_phrases)} entities already processed.")

        # 2. Stream dataset from Hugging Face
        print(f"[*] Streaming entities from {self.dataset_name}...")
        try:
            from datasets import load_dataset
            ds = load_dataset(self.dataset_name, split="train", streaming=True)
        except Exception as e:
            print(f"[-] Failed to load dataset: {e}")
            return

        batch_queue: List[Dict[str, Any]] = []
        processed_total = len(seen_phrases)
        batch_idx = 0
        start_time = time.time()

        def process_entity(entity: Dict[str, Any]) -> int:
            phrase = entity.get("phrase", "").strip()
            count = entity.get("count", 0)
            if not phrase or len(phrase) <= 1:
                return 0

            try:
                # Query Ghana Search across news, academic, public_services, wikipedia
                results = _execute_search(
                    text=phrase,
                    categories=["news", "academic", "public_services", "wikipedia"],
                    top_k=5,
                    rerank=True,
                    max_wait_seconds=self.max_wait,
                    client_type="entity_crawler",
                )
                self._record_processed(phrase, count, len(results))
                return len(results)
            except Exception:
                self._record_processed(phrase, count, 0)
                return 0

        # Main streaming iteration loop
        try:
            for row in ds:
                if not self.running:
                    print("\n[!] Gracefully stopping crawler loop...")
                    break

                phrase = (row.get("phrase") or "").strip()
                if not phrase or phrase.lower() in seen_phrases or len(phrase) <= 1:
                    continue

                seen_phrases.add(phrase.lower())
                batch_queue.append(row)

                if len(batch_queue) >= self.batch_size:
                    batch_idx += 1
                    b_t0 = time.time()

                    # Process batch with worker threads
                    with ThreadPoolExecutor(max_workers=self.workers) as executor:
                        future_to_entity = {
                            executor.submit(process_entity, ent): ent for ent in batch_queue
                        }
                        for future in as_completed(future_to_entity):
                            future.result()

                    processed_total += len(batch_queue)
                    last_phrase = batch_queue[-1].get("phrase", "")
                    batch_queue.clear()

                    # Save checkpoint state
                    self._save_checkpoint_stats(processed_total, last_phrase)

                    stats = search_cache.get_url_cache_stats()
                    total_urls = stats.get("total_permanent_urls", 0)
                    elapsed = time.time() - start_time
                    qps = processed_total / max(elapsed, 0.1)

                    print(
                        f"[Batch {batch_idx:04d}] Processed: {processed_total:,} entities | "
                        f"Unique Cached URLs: {total_urls:,} | "
                        f"QPS: {qps:.1f} | Last Entity: '{last_phrase}'"
                    )

                    # Auto-push to Hugging Face
                    self.export_and_push_to_hf(batch_idx)

                if self.limit and processed_total >= self.limit:
                    print(f"[+] Reached requested entity limit ({self.limit}).")
                    break

        except KeyboardInterrupt:
            print("\n[!] KeyboardInterrupt received. Saving checkpoint and exiting...")
        except Exception as e:
            print(f"\n[-] Crawler loop error: {e}")

        # Process any remaining entities in queue
        if batch_queue:
            batch_idx += 1
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                for ent in batch_queue:
                    executor.submit(process_entity, ent)
            processed_total += len(batch_queue)
            self._save_checkpoint_stats(processed_total, batch_queue[-1].get("phrase", ""))
            self.export_and_push_to_hf(batch_idx)

        print("\n" + "=" * 70)
        print("CRAWLER SESSION TERMINATED")
        stats = search_cache.get_url_cache_stats()
        print(f"Total Processed Entities: {processed_total:,}")
        print(f"Total Unique URLs Cached: {stats.get('total_permanent_urls', 0):,}")
        print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Ghana Search: Named Entity Offline Cache Crawler & Hugging Face Sync."
    )
    parser.add_argument("--batch-size", type=int, default=50, help="Entities per push batch (default: 50)")
    parser.add_argument("--max-wait", type=float, default=5.0, help="Max wait seconds per query (default: 5.0)")
    parser.add_argument("--workers", type=int, default=3, help="Concurrent worker threads (default: 3)")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on total entities to process")
    parser.add_argument("--no-hf-push", action="store_true", help="Disable pushing to Hugging Face")
    parser.add_argument("--status", action="store_true", help="Print crawler checkpoint status and exit")

    args = parser.parse_args()

    if args.status:
        if os.path.exists(DEFAULT_CHECKPOINT_FILE):
            with open(DEFAULT_CHECKPOINT_FILE, "r") as f:
                print(f.read())
        else:
            print("No checkpoint file found at", DEFAULT_CHECKPOINT_FILE)
        return

    crawler = EntityCacheCrawler(
        batch_size=args.batch_size,
        max_wait_seconds=args.max_wait,
        workers=args.workers,
        limit=args.limit,
        push_to_hf=not args.no_hf_push,
    )

    def signal_handler(sig, frame):
        print("\n[!] Received stop signal, shutting down gracefully...")
        crawler.running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    crawler.run()


if __name__ == "__main__":
    main()
