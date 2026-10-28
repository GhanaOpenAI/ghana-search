"""Title-Only & Recent-First Ranking Engine for Ghana Search.
100% CPU-based: Zero GPU or VRAM required.
Strict Ranking Policy:
  1. Title match ONLY (Mandatory): Candidates MUST match query keywords or phrases in their title.
     Any document with zero title match is discarded (no content/body searching).
  2. Foreign nation filter: Articles about other countries (e.g. Kenya, Nigeria) without Ghana in the title are rejected.
  3. Publication recency: Fresher and more current matching articles receive a substantial recency boost.
"""
import re
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Set

from ghana_mcp.providers.base import SearchResult
from ghana_mcp.nlp import token_matches_title, has_word, is_foreign_title, FOREIGN_COUNTRIES, GHANA_CITIES


def parse_date_timestamp(date_str: Optional[str]) -> float:
    """Parse various Ghanaian date formats into an epoch timestamp."""
    if not date_str:
        return 0.0

    ds = str(date_str).strip()

    # 1. Relative dates: "2 hours ago", "3 days ago"
    rel_match = re.search(r"(\d+)\s+(day|hour|minute|min|week|month|year)s?\s+ago", ds, re.I)
    if rel_match:
        val = int(rel_match.group(1))
        unit = rel_match.group(2).lower()
        if "day" in unit:
            delta = timedelta(days=val)
        elif "hour" in unit:
            delta = timedelta(hours=val)
        elif "min" in unit:
            delta = timedelta(minutes=val)
        elif "week" in unit:
            delta = timedelta(weeks=val)
        elif "month" in unit:
            delta = timedelta(days=val * 30)
        elif "year" in unit:
            delta = timedelta(days=val * 365)
        else:
            delta = timedelta(days=1)
        return (datetime.now(timezone.utc) - delta).timestamp()

    # 2. ISO format: "2026-10-23T10:50:47.712Z" or "2026-10-22"
    try:
        clean_iso = ds.replace("Z", "+00:00")
        return datetime.fromisoformat(clean_iso).timestamp()
    except Exception:
        pass

    # 3. String formats
    formats = [
        "%m/%d/%Y %I:%M:%S %p",
        "%m/%d/%Y",
        "%d %B %Y",
        "%d %b %Y",
        "%Y-%m-%d",
        "%Y-%m",
        "%Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(ds, fmt).replace(tzinfo=timezone.utc).timestamp()
        except Exception:
            continue

    # 4. 4-digit year fallback
    year_match = re.search(r"\b(19\d{2}|20\d{2})\b", ds)
    if year_match:
        try:
            return datetime(int(year_match.group(1)), 1, 1, tzinfo=timezone.utc).timestamp()
        except Exception:
            pass

    return 0.0


def compute_recency_bonus(date_epoch: float, now_epoch: float) -> float:
    """Calculate a recency bonus: newer/fresher articles receive a substantial boost."""
    if not date_epoch or date_epoch <= 0:
        return 0.0

    age_days = max(0.0, (now_epoch - date_epoch) / 86400.0)

    # Current week (0 to 7 days old): up to +40.0 pts
    if age_days <= 7:
        return round(40.0 - (age_days * 1.5), 2)
    # Current month (7 to 30 days old): +25.0 to +30.0 pts
    elif age_days <= 30:
        return round(30.0 - ((age_days - 7) * 0.2), 2)
    # Current year (30 to 365 days old): +15.0 to +25.0 pts
    elif age_days <= 365:
        return round(25.0 - ((age_days - 30) * 0.03), 2)
    # Within 3 years: +5.0 to +15.0 pts
    elif age_days <= 1095:
        return round(max(5.0, 15.0 - ((age_days - 365) * 0.01)), 2)

    return 0.0


class TitleFirstReranker:
    """Title-first and recent-first ranking engine:
    Prioritizes title matches heavily, boosting recent publications so fresh
    matching records appear first.
    Zero GPU or VRAM required.
    """

    STOPWORDS = {
        "who", "was", "is", "are", "were", "the", "a", "an", "of", "and",
        "in", "on", "at", "to", "for", "with", "about", "from", "by", "how",
        "what", "when", "where", "why", "can", "you", "tell", "me"
    }

    def rerank(
        self,
        query: str,
        keywords: List[str],
        candidates: List[SearchResult],
        top_k: int = 50,
    ) -> List[SearchResult]:
        if not candidates:
            return []

        query_lower = query.lower()

        # Extract meaningful query tokens
        raw_tokens = re.findall(r"\b[a-zA-Z0-9_\-]{2,}\b", query_lower)
        query_tokens = [t for t in raw_tokens if t not in self.STOPWORDS]
        if not query_tokens:
            query_tokens = raw_tokens

        # Extract full noun chunks and 2-word sub-phrases (e.g. "vice president" from "former vice president")
        chunks = [k.lower().strip() for k in keywords if len(k.strip()) > 1]
        sub_phrases: Set[str] = set()
        for c in chunks:
            words = c.split()
            if len(words) >= 2:
                sub_phrases.add(c)
                # Add adjacent bi-grams
                for i in range(len(words) - 1):
                    sub_phrases.add(f"{words[i]} {words[i+1]}")

        non_city_tokens = [t for t in query_tokens if t.lower() not in GHANA_CITIES]
        now_ts = datetime.now(timezone.utc).timestamp()
        scored_candidates = []

        for item in candidates:
            title = item.title or ""
            title_lower = title.lower()

            # 1. Filter out articles referencing foreign nations without Ghana context
            if is_foreign_title(title):
                continue

            cat_desc = (item.metadata.get("business_category") or "").lower()
            search_target = f"{title_lower} {cat_desc}" if item.category == "business" else title_lower

            # 2. Strict Core Subject Matching: If query has core non-city tokens,
            # document MUST match the core product/subject, never only the city modifier!
            if non_city_tokens:
                matches_core = any(token_matches_title(t, search_target) for t in non_city_tokens)
                if not matches_core:
                    continue

            # 3. Title Matching (No body/content searching)
            title_phrase_matches = sum(1 for p in sub_phrases if has_word(p, title_lower))
            title_token_matches = sum(1 for t in query_tokens if token_matches_title(t, search_target))

            # Mandatory match: Discard any document with zero overlap
            if title_phrase_matches == 0 and title_token_matches == 0:
                continue

            title_coverage = (
                title_token_matches / len(query_tokens)
                if query_tokens
                else 0.0
            )

            title_score = (
                (title_phrase_matches * 60.0)
                + (title_token_matches * 25.0)
                + (title_coverage * 40.0)
            )

            # 4. Recency bonus (matching items published recently rank higher)
            ts = parse_date_timestamp(item.published_date)
            r_bonus = compute_recency_bonus(ts, now_ts)

            # Strictly title-score + recency bonus (Zero points for body/snippet text)
            composite_score = round(title_score + r_bonus, 2)

            item.score = composite_score
            item.metadata["ranking_engine"] = "cpu_title_only_recent_first"
            item.metadata["title_score"] = round(title_score, 1)
            item.metadata["recency_bonus"] = round(r_bonus, 1)

            scored_candidates.append((composite_score, item))

        # Sort strictly by descending composite score (title match + recent first)
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored_candidates[:top_k]]


# Global Title-First & Recent-First Reranker
reranker = TitleFirstReranker()
