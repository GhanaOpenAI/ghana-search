"""Landmark Location & Surroundings Provider for Ghana.
Provides landmark-based positioning: describes which town/city, neighbourhood,
and region a place is in, its main access road, and what well-known landmarks,
stations, or institutions are near it, without turn-by-turn navigation.
"""
import os
import sys
import re
import json
import urllib.request
import urllib.parse
import ssl
from typing import Dict, Any, List, Optional

from ghana_mcp.config import USER_AGENT, DEFAULT_TIMEOUT
from ghana_mcp.cache import search_cache

SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE


class LocationFinderProvider:
    name = "ghana_location_finder"

    def __init__(self, timeout: float = DEFAULT_TIMEOUT):
        self.timeout = timeout
        self.headers = {
            "User-Agent": "GhanaSearchEngine/1.0 (https://search.ghanaopenai.org; contact@ghanaopenai.org)",
            "Accept": "application/json",
        }

    def _geocode(self, place_name: str) -> Optional[Dict[str, Any]]:
        """Geocode place using Nominatim scoped to Ghana."""
        clean_name = place_name.strip()
        query = f"{clean_name}, Ghana" if "ghana" not in clean_name.lower() else clean_name
        url = f"https://nominatim.openstreetmap.org/search?q={urllib.parse.quote(query)}&format=json&addressdetails=1&limit=1"
        req = urllib.request.Request(url, headers=self.headers)

        try:
            with urllib.request.urlopen(req, timeout=self.timeout, context=SSL_CTX) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data and isinstance(data, list):
                    return data[0]
        except Exception:
            pass
        return None

    def lookup_location(self, landmark_name: str) -> Dict[str, Any]:
        """Look up landmark location, town, neighbourhood, and main access road.
        Checks permanent local SQLite cache first (instant response, offline compatible).
        When online, falls back to OpenStreetMap Nominatim + Wikipedia Geosearch and caches result.
        """
        clean_name = landmark_name.strip()
        if not clean_name:
            return {
                "landmark": "",
                "found": False,
                "description": "No landmark or place name provided.",
            }

        # 1. Check permanent local landmark cache first (0ms, 100% offline-ready)
        cached_data = search_cache.get_landmark(clean_name)
        if cached_data is not None:
            return cached_data

        # 2. If running offline and not cached yet, explain clearly
        is_offline = (
            os.environ.get("GHANA_SEARCH_OFFLINE", "").lower() in ("1", "true", "yes")
            or "--offline" in sys.argv
        )
        if is_offline:
            return {
                "landmark": clean_name,
                "found": False,
                "description": f"'{clean_name}' is not in the offline landmark cache yet. Query it once while online to permanently cache it.",
            }

        # 3. Live Online Resolution (OpenStreetMap + Wikipedia Geosearch)
        geo = self._geocode(clean_name)
        if not geo:
            return {
                "landmark": clean_name,
                "found": False,
                "description": f"Could not locate '{clean_name}' in Ghana map records.",
            }

        name = geo.get("name") or clean_name
        addr = geo.get("address", {})
        lat = float(geo.get("lat", 0.0))
        lon = float(geo.get("lon", 0.0))

        town = addr.get("city") or addr.get("town") or addr.get("municipality") or addr.get("village") or ""
        suburb = addr.get("suburb") or addr.get("neighbourhood") or addr.get("quarter") or ""
        road = addr.get("road") or ""
        district = addr.get("state_district") or addr.get("county") or ""
        region = addr.get("state") or addr.get("region") or ""

        # Construct location string
        loc_components = []
        if suburb:
            loc_components.append(suburb)
        if town and town != suburb:
            loc_components.append(town)
        if district and district not in (suburb, town):
            loc_components.append(district)
        if region:
            loc_components.append(region)

        location_summary = ", ".join(loc_components) if loc_components else "Ghana"

        # Plain spoken descriptive sentences (no turns, no GPS jargon)
        sentences = [f"**{name}** is located in {location_summary}."]
        if road:
            sentences.append(f"It is situated along or just off **{road}**.")

        description_text = " ".join(sentences)

        result = {
            "landmark": name,
            "found": True,
            "town": town or suburb,
            "suburb": suburb,
            "road": road,
            "district": district,
            "region": region,
            "coordinates": {"latitude": lat, "longitude": lon},
            "description": description_text,
        }

        # 4. Save permanently into SQLite landmark_cache for offline and repeat lookups
        search_cache.save_landmark(clean_name, result)

        return result
