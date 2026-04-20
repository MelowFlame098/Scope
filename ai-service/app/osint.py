from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

import requests


class OSINTModule:
    def __init__(self, db):
        self.db = db
        self.collection = db["osint_events"]
        self.collection.create_index([("source", 1), ("source_id", 1)], unique=True)
        self.collection.create_index([("published_at", -1)])

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    def _compact(self, s: str) -> str:
        return " ".join((s or "").replace("\n", " ").split())

    def _truncate(self, s: str, max_len: int) -> str:
        s = self._compact(s)
        if max_len <= 0:
            return ""
        if len(s) <= max_len:
            return s
        return s[: max_len - 1] + "…"

    def _request_json(self, method: str, url: str, *, params: Optional[Dict[str, Any]] = None, data: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None, timeout_s: int = 20) -> Dict[str, Any]:
        hdrs = {"User-Agent": "ScopeOSINT/1.0"}
        if headers:
            hdrs.update(headers)
        if method.lower() == "post":
            resp = requests.post(url, data=data, params=params, headers=hdrs, timeout=timeout_s)
        else:
            resp = requests.get(url, params=params, headers=hdrs, timeout=timeout_s)
        resp.raise_for_status()
        return resp.json()

    def _hash_id(self, source: str, payload: Dict[str, Any]) -> str:
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return f"{source}:{hashlib.sha256(raw).hexdigest()[:24]}"

    def _point(self, lon: Optional[float], lat: Optional[float]) -> Optional[Dict[str, Any]]:
        if lon is None or lat is None:
            return None
        return {"type": "Point", "coordinates": [float(lon), float(lat)]}

    def _upsert_many(self, events: List[Dict[str, Any]]) -> int:
        if not events:
            return 0
        now = self._now()
        upserted = 0
        for e in events:
            if not e.get("source") or not e.get("source_id"):
                continue
            e["updated_at"] = now
            self.collection.update_one(
                {"source": e["source"], "source_id": e["source_id"]},
                {"$set": e, "$setOnInsert": {"created_at": now}},
                upsert=True,
            )
            upserted += 1
        return upserted

    def fetch_usgs_earthquakes(self) -> int:
        data = self._request_json("get", "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_hour.geojson")
        out: List[Dict[str, Any]] = []
        for f in data.get("features", []):
            props = f.get("properties") or {}
            geom = f.get("geometry") or {}
            coords = geom.get("coordinates") or []
            lon = coords[0] if len(coords) > 0 else None
            lat = coords[1] if len(coords) > 1 else None
            ts_ms = props.get("time")
            published_at = self._now()
            if isinstance(ts_ms, (int, float)):
                published_at = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            mag = props.get("mag")
            place = props.get("place") or ""
            title = props.get("title") or f"M{mag} {place}".strip()
            summary = place
            out.append(
                {
                    "source": "USGS",
                    "source_id": f"usgs:{f.get('id') or self._hash_id('usgs', f)}",
                    "kind": "earthquake",
                    "title": self._truncate(str(title), 160),
                    "summary": self._truncate(str(summary), 320),
                    "url": props.get("url") or "",
                    "region": place,
                    "category": "Earthquake",
                    "published_at": published_at,
                    "geo": self._point(lon, lat),
                    "tags": ["earthquake", "usgs"],
                    "metrics": {"magnitude": mag, "tsunami": props.get("tsunami"), "depth_km": coords[2] if len(coords) > 2 else None},
                }
            )
        return self._upsert_many(out)

    def fetch_nasa_eonet(self) -> int:
        data = self._request_json("get", "https://eonet.gsfc.nasa.gov/api/v3/events", params={"status": "open", "limit": 50})
        out: List[Dict[str, Any]] = []
        for ev in data.get("events", []):
            geometries = ev.get("geometry") or []
            published_at = self._now()
            lon = None
            lat = None
            if geometries:
                g0 = geometries[-1]
                date_str = g0.get("date")
                if isinstance(date_str, str) and date_str:
                    try:
                        published_at = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                    except Exception:
                        published_at = self._now()
                coords = g0.get("coordinates")
                if isinstance(coords, list) and len(coords) == 2 and all(isinstance(x, (int, float)) for x in coords):
                    lon, lat = coords[0], coords[1]
            categories = ev.get("categories") or []
            cat = categories[0].get("title") if categories and isinstance(categories[0], dict) else ""
            title = ev.get("title") or ""
            out.append(
                {
                    "source": "NASA EONET",
                    "source_id": f"eonet:{ev.get('id') or self._hash_id('eonet', ev)}",
                    "kind": "disaster",
                    "title": self._truncate(str(title), 160),
                    "summary": self._truncate(str(title), 320),
                    "url": ev.get("link") or "",
                    "region": "Global",
                    "category": cat or "Disaster",
                    "published_at": published_at if isinstance(published_at, datetime) else self._now(),
                    "geo": self._point(lon, lat),
                    "tags": ["disaster", "eonet"],
                }
            )
        return self._upsert_many(out)

    def fetch_gdelt(self, query: str = "conflict OR earthquake OR wildfire OR flood OR cyber") -> int:
        data = self._request_json(
            "get",
            "https://api.gdeltproject.org/api/v2/doc/doc",
            params={
                "query": query,
                "mode": "ArtList",
                "format": "json",
                "maxrecords": 50,
                "sort": "HybridRel",
            },
        )
        out: List[Dict[str, Any]] = []
        for a in data.get("articles", []):
            seendate = a.get("seendate") or ""
            published_at = self._now()
            if isinstance(seendate, str) and len(seendate) >= 14:
                try:
                    published_at = datetime.strptime(seendate[:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
                except Exception:
                    published_at = self._now()
            title = a.get("title") or ""
            url = a.get("url") or ""
            source_country = a.get("sourceCountry") or ""
            out.append(
                {
                    "source": "GDELT",
                    "source_id": self._hash_id("gdelt", {"url": url, "seendate": seendate}),
                    "kind": "gdelt",
                    "title": self._truncate(str(title), 160),
                    "summary": self._truncate(str(a.get("snippet") or ""), 320),
                    "url": url,
                    "region": source_country or "Global",
                    "category": "Global Events",
                    "published_at": published_at,
                    "tags": ["gdelt", "events"],
                }
            )
        return self._upsert_many(out)

    def fetch_urlhaus_recent(self, limit: int = 50) -> int:
        data = self._request_json("post", "https://urlhaus-api.abuse.ch/v1/urls/recent/", data={"limit": str(limit)})
        out: List[Dict[str, Any]] = []
        for u in data.get("urls", []) or []:
            date_added = u.get("date_added") or ""
            published_at = self._now()
            if isinstance(date_added, str) and date_added:
                try:
                    published_at = datetime.fromisoformat(date_added.replace("Z", "+00:00"))
                except Exception:
                    published_at = self._now()
            threat = u.get("threat") or "malicious"
            url = u.get("url") or ""
            title = f"URLhaus: {threat}"
            tags = u.get("tags") or []
            if isinstance(tags, str):
                tags = [tags]
            out.append(
                {
                    "source": "URLhaus",
                    "source_id": f"urlhaus:{u.get('url_id') or self._hash_id('urlhaus', {'url': url, 'date_added': date_added})}",
                    "kind": "cyber",
                    "title": self._truncate(title, 160),
                    "summary": self._truncate(url, 320),
                    "url": url,
                    "region": "Global",
                    "category": threat,
                    "published_at": published_at,
                    "tags": ["cyber", "urlhaus"] + [str(t) for t in tags[:8]],
                }
            )
        return self._upsert_many(out)

    def fetch_coingecko_prices(self, ids: Optional[List[str]] = None) -> int:
        if ids is None:
            ids = ["bitcoin", "ethereum", "solana", "cardano", "ripple"]
        data = self._request_json(
            "get",
            "https://api.coingecko.com/api/v3/simple/price",
            params={
                "ids": ",".join(ids),
                "vs_currencies": "usd",
                "include_24hr_change": "true",
            },
        )
        now = self._now()
        out: List[Dict[str, Any]] = []
        for coin_id, v in (data or {}).items():
            if not isinstance(v, dict):
                continue
            price = v.get("usd")
            change = v.get("usd_24h_change")
            title = f"{coin_id.upper()} ${price}"
            if change is not None:
                title = f"{coin_id.upper()} ${price} ({change:+.2f}% 24h)"
            out.append(
                {
                    "source": "CoinGecko",
                    "source_id": f"coingecko:{coin_id}:{now.strftime('%Y%m%d%H%M')}",
                    "kind": "crypto",
                    "title": self._truncate(title, 160),
                    "summary": self._truncate(f"USD price for {coin_id}: {price} (24h change: {change})", 320),
                    "url": f"https://www.coingecko.com/en/coins/{coin_id}",
                    "region": "Global",
                    "category": "Crypto",
                    "published_at": now,
                    "tags": ["crypto", "coingecko", coin_id],
                    "metrics": {"usd": price, "usd_24h_change": change},
                }
            )
        return self._upsert_many(out)

    def fetch_opensky_states(self, lamin: Optional[float] = None, lomin: Optional[float] = None, lamax: Optional[float] = None, lomax: Optional[float] = None, limit: int = 200) -> int:
        params: Dict[str, Any] = {}
        if lamin is not None and lomin is not None and lamax is not None and lomax is not None:
            params.update({"lamin": lamin, "lomin": lomin, "lamax": lamax, "lomax": lomax})
        data = self._request_json("get", "https://opensky-network.org/api/states/all", params=params)
        states = data.get("states") or []
        ts = data.get("time")
        published_at = self._now()
        if isinstance(ts, (int, float)):
            try:
                published_at = datetime.fromtimestamp(ts, tz=timezone.utc)
            except Exception:
                published_at = self._now()
        out: List[Dict[str, Any]] = []
        for s in states[: limit]:
            if not isinstance(s, list) or len(s) < 8:
                continue
            icao24 = s[0] or ""
            callsign = (s[1] or "").strip()
            origin = s[2] or ""
            lon = s[5]
            lat = s[6]
            velocity = s[9] if len(s) > 9 else None
            heading = s[10] if len(s) > 10 else None
            if not isinstance(lon, (int, float)) or not isinstance(lat, (int, float)):
                continue
            title = f"{callsign or icao24} over {origin}" if origin else (callsign or icao24)
            out.append(
                {
                    "source": "OpenSky",
                    "source_id": f"opensky:{icao24}",
                    "kind": "aviation",
                    "title": self._truncate(title, 160),
                    "summary": self._truncate(f"{callsign or icao24} {origin}", 320),
                    "url": "https://opensky-network.org",
                    "region": origin or "Global",
                    "category": "Aircraft",
                    "published_at": published_at,
                    "geo": self._point(lon, lat),
                    "tags": ["aviation", "opensky"],
                    "metrics": {"velocity": velocity, "heading": heading},
                }
            )
        return self._upsert_many(out)
