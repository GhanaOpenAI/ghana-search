"""Yellow Pages & Business Directory search provider for Ghanaian commercial enterprises."""
import re
import urllib.request
import urllib.parse
import ssl
from typing import List, Optional, Tuple, Set
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup

from ghana_mcp.providers.base import BaseProvider, SearchResult
from ghana_mcp.config import USER_AGENT, DEFAULT_TIMEOUT
from ghana_mcp.nlp import GHANA_CITIES, token_matches_title, has_word

SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

GENERIC_COMMERCIAL_TERMS = {
    "company", "companies", "business", "businesses", "enterprise",
    "enterprises", "service", "services", "shop", "shops", "store",
    "stores", "dealer", "dealers", "supplier", "suppliers", "ltd", "limited",
    "in", "at", "for", "the", "and", "of", "to"
}


def normalize_business_token(w: str) -> str:
    """Normalize plural and casing for business category matching."""
    lower = w.lower().strip()
    if lower.endswith("ies") and len(lower) > 4:
        return lower[:-3] + "y"
    if lower.endswith("s") and len(lower) > 3 and not lower.endswith("ss"):
        return lower[:-1]
    return lower


def is_relevant_business(name: str, cat_desc: str, core_tokens: List[str]) -> bool:
    """Ensure business is strictly relevant to the product/service searched.
    Matches either the business name or the official business category description.
    Prevents unrelated businesses (e.g. golf clubs, car parts shops) from matching
    a query just because they share a city name.
    """
    if not core_tokens:
        return True

    target_text = f"{name} {cat_desc}".lower()
    for t in core_tokens:
        if token_matches_title(t, target_text):
            return True
        norm_t = normalize_business_token(t)
        if has_word(norm_t, target_text):
            return True

    return False


class BusinessDirectoryProvider(BaseProvider):
    name = "ghana_business"
    category = "business"

    def __init__(self, timeout: float = DEFAULT_TIMEOUT):
        self.timeout = timeout
        self.headers = {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

    def _parse_query_params(self, raw_query: str) -> Tuple[List[str], str, Optional[str]]:
        """Extract core search tokens, clean query string, and optional city/location."""
        q_lower = raw_query.lower()
        found_city = None
        for city in GHANA_CITIES:
            if re.search(rf"\b{re.escape(city)}\b", q_lower):
                found_city = city
                break

        core_tokens = [
            normalize_business_token(t)
            for t in re.findall(r"\b[a-zA-Z0-9_\-]{2,}\b", raw_query)
            if t.lower() not in GENERIC_COMMERCIAL_TERMS and t.lower() != found_city
        ]
        q_str = " ".join(core_tokens) if core_tokens else raw_query
        return core_tokens, q_str, found_city

    def _fetch_listings(
        self,
        query_str: str,
        city: Optional[str],
        core_tokens: List[str],
        limit: int,
    ) -> List[SearchResult]:
        params = {"q": query_str}
        if city:
            params["where"] = city

        url = f"https://www.ghanabusinessweb.com/search?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers=self.headers)
        results: List[SearchResult] = []
        seen_urls: Set[str] = set()

        try:
            with urllib.request.urlopen(req, timeout=self.timeout, context=SSL_CTX) as resp:
                html = resp.read().decode("utf-8", errors="ignore")

            soup = BeautifulSoup(html, "html.parser")

            for art in soup.find_all("article"):
                h = art.find(["h2", "h3", "h4"])
                if not h or not h.find("a"):
                    continue

                name = h.get_text(strip=True)
                link = h.find("a").get("href", "")
                if not name or link in seen_urls:
                    continue

                # Extract text lines inside article card (skip single-letter avatar initials)
                texts = [
                    t.strip()
                    for t in art.stripped_strings
                    if len(t.strip()) > 1 and t.strip() != name
                ]

                # Phone number extraction
                phone_match = re.search(
                    r"(\+?233[\d\s\-]{8,}|\b0\d{2}[\s\-]?\d{3}[\s\-]?\d{4}\b)",
                    " ".join(texts),
                )
                phone = phone_match.group(0).strip() if phone_match else None

                cat_desc = texts[0] if texts else "Commercial Services"

                # STRICT RELEVANCE: Business name or category MUST match the core product/service tokens!
                # Discards unrelated businesses (e.g. Golf Club, Printing Press for a Pharmacy query)
                if not is_relevant_business(name, cat_desc, core_tokens):
                    continue

                seen_urls.add(link)

                loc = city.title() if city else "Ghana"
                for t in texts[1:]:
                    if t != phone and "open" not in t.lower() and "closed" not in t.lower():
                        loc = t
                        break

                snippet_parts = [cat_desc]
                if loc and loc != "Ghana":
                    snippet_parts.append(f"Location: {loc}")
                if phone:
                    snippet_parts.append(f"Tel: {phone}")

                snippet = " • ".join(snippet_parts)
                full_content = (
                    f"{name} is a verified Ghanaian commercial enterprise ({cat_desc}). "
                    f"Location/address: {loc}. Contact telephone: {phone or 'Listed on directory portal'}."
                )

                results.append(
                    SearchResult(
                        title=name,
                        url=link,
                        snippet=snippet,
                        source="Ghana Yellow Pages (GhanaBusinessWeb)",
                        category=self.category,
                        content=full_content,
                        metadata={
                            "type": "business_directory / yellow_pages",
                            "phone": phone,
                            "location": loc,
                            "business_category": cat_desc,
                        },
                    )
                )

                if len(results) >= limit:
                    break
        except Exception:
            pass

        return results

    def search(self, keywords: List[str], limit: int = 5) -> List[SearchResult]:
        raw_query = " ".join(keywords).strip()
        if not raw_query:
            return []

        core_tokens, q_term, city = self._parse_query_params(raw_query)

        # If a specific city was requested, search concurrently:
        # 1. Targeted city search
        # 2. National core product/service search (to provide verified alternatives if city results are sparse)
        if city:
            with ThreadPoolExecutor(max_workers=2) as ex:
                f_city = ex.submit(self._fetch_listings, q_term, city, core_tokens, limit)
                f_nat = ex.submit(self._fetch_listings, q_term, None, core_tokens, limit)
                res_city = f_city.result()
                res_nat = f_nat.result()

            seen: Set[str] = set()
            combined: List[SearchResult] = []
            for r in res_city:
                if r.url not in seen:
                    seen.add(r.url)
                    combined.append(r)

            for r in res_nat:
                if len(combined) >= limit:
                    break
                if r.url not in seen:
                    seen.add(r.url)
                    combined.append(r)

            return combined[:limit]
        else:
            return self._fetch_listings(q_term, None, core_tokens, limit)
