"""Classifieds ads product search and price aggregation provider for Ghana.
Queries Tonaton.com's web API (Jiji platform) to extract live classified ads, calculate
average asking prices, price ranges (min/max), and provide direct listing URLs in Ghana Cedis (GH₵).

Tonaton sits behind an AWS WAF captcha. Requests only succeed when TONATON_COOKIE holds a
session that has passed verification; otherwise the result carries an explanatory "error".
"""
import json
import urllib.request
import urllib.parse
import urllib.error
import ssl
from collections import Counter
from typing import Dict, Any, List, Optional, Tuple

from ghana_mcp.config import DEFAULT_TIMEOUT, TONATON_COOKIE

SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

TONATON_API = "https://tonaton.com/api_web/v1/listing"


class PriceFinderProvider:
    name = "ghana_price_finder"

    def __init__(self, timeout: float = DEFAULT_TIMEOUT):
        self.timeout = timeout
        self.headers = {
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:157.0) Gecko/20100101 Firefox/157.0",
            "Accept": "application/json",
            "Referer": "https://tonaton.com/",
        }
        if TONATON_COOKIE:
            self.headers["Cookie"] = TONATON_COOKIE

    def _fetch_tonaton_api(self, product_query: str, limit: int) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Fetch live classified ads from Tonaton.com's listing API, keeping only the dominant category."""
        listings: List[Dict[str, Any]] = []
        url = f"{TONATON_API}?{urllib.parse.urlencode({'query': product_query})}"

        try:
            req = urllib.request.Request(url, headers=self.headers)
            with urllib.request.urlopen(req, timeout=min(self.timeout, 3.2), context=SSL_CTX) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 405 or e.headers.get("x-amzn-waf-action"):
                return [], "Tonaton.com blocked the request (AWS WAF human verification required)."
            return [], f"Tonaton.com API returned HTTP {e.code}."
        except Exception as e:
            return [], f"Tonaton.com API request failed: {e}"

        adverts = data.get("adverts_list", {}).get("adverts", [])

        # Keep the category that dominates the results so accessories (chargers, cases)
        # don't drag the price stats of the main product down.
        top_cats = data.get("search_categories") or []
        main_cat = top_cats[0].get("text") if top_cats else None
        if not main_cat and adverts:
            main_cat = Counter(ad.get("category_name") for ad in adverts).most_common(1)[0][0]
        in_cat = [ad for ad in adverts if ad.get("category_name") == main_cat]
        if in_cat:
            adverts = in_cat

        for ad in adverts:
            title = ad.get("title") or ""
            price_val = (ad.get("price_obj") or {}).get("value") or 0
            guid = ad.get("guid") or ""
            slug = ad.get("slug") or ""
            if not (title and price_val > 0 and guid and slug):
                continue

            listings.append({
                "title": title,
                "price_ghs": float(price_val),
                "source": "Tonaton.com",
                "category": ad.get("category_name") or "",
                "location": ad.get("region_item_text") or ad.get("region_name") or "Ghana",
                "url": f"https://tonaton.com/a-{guid}-{slug}.html",
            })
            if len(listings) >= limit:
                break

        return listings, None

    def search_prices(self, product_query: str, limit: int = 15) -> Dict[str, Any]:
        """Search active Tonaton.com classifieds listings in Ghana and calculate price metrics."""
        clean_query = product_query.strip()
        result: Dict[str, Any] = {
            "query": clean_query,
            "sample_size": 0,
            "average_price_ghs": None,
            "min_price_ghs": None,
            "max_price_ghs": None,
            "currency": "GHS (Ghana Cedi)",
            "primary_source": "Tonaton.com",
            "listings": [],
        }
        if not clean_query:
            return result

        listings, err_msg = self._fetch_tonaton_api(clean_query, limit=limit)
        if err_msg:
            result["error"] = err_msg

        prices = [it["price_ghs"] for it in listings]
        if prices:
            result.update({
                "sample_size": len(listings),
                "average_price_ghs": round(sum(prices) / len(prices), 2),
                "min_price_ghs": min(prices),
                "max_price_ghs": max(prices),
                "listings": listings,
            })
        return result
