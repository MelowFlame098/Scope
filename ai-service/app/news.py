import time
from datetime import datetime
from finvizfinance.quote import finvizfinance
from pymongo import MongoClient
from textblob import TextBlob
import pandas as pd
import hashlib
import random
import re
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional, Tuple
import requests
import xml.etree.ElementTree as ET

# Stocks to track for news (default list)
TICKERS = ["AAPL", "GOOGL", "TSLA", "MSFT", "AMZN", "NVDA", "AMD"]

RSS_FEEDS: List[Tuple[str, str]] = [
    ("Google News Top", "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en"),
    ("Google World", "https://news.google.com/rss/headlines/section/topic/WORLD?hl=en-US&gl=US&ceid=US:en"),
    ("Google Business", "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-US&gl=US&ceid=US:en"),
    ("Google Technology", "https://news.google.com/rss/headlines/section/topic/TECHNOLOGY?hl=en-US&gl=US&ceid=US:en"),
    ("Google Science", "https://news.google.com/rss/headlines/section/topic/SCIENCE?hl=en-US&gl=US&ceid=US:en"),
    ("BBC World", "https://feeds.bbci.co.uk/news/world/rss.xml"),
    ("BBC Business", "https://feeds.bbci.co.uk/news/business/rss.xml"),
    ("Al Jazeera", "https://www.aljazeera.com/xml/rss/all.xml"),
]

HIGH_IMPACT_KEYWORDS: List[Tuple[str, int]] = [
    ("breaking", 6),
    ("attack", 6),
    ("strike", 5),
    ("missile", 6),
    ("drone", 4),
    ("explosion", 6),
    ("killed", 6),
    ("hostage", 6),
    ("sanction", 7),
    ("tariff", 6),
    ("embargo", 7),
    ("ceasefire", 5),
    ("invasion", 8),
    ("mobilization", 6),
    ("coup", 7),
    ("election", 4),
    ("nuclear", 7),
    ("iran", 6),
    ("israel", 5),
    ("gaza", 6),
    ("hormuz", 7),
    ("red sea", 7),
    ("yemen", 5),
    ("ukraine", 7),
    ("russia", 6),
    ("china", 6),
    ("taiwan", 6),
    ("south china sea", 6),
    ("opec", 6),
    ("oil", 4),
    ("gas", 4),
    ("lng", 5),
    ("copper", 5),
    ("rare earth", 6),
    ("inflation", 6),
    ("cpi", 6),
    ("gdp", 5),
    ("rate hike", 7),
    ("rate cut", 7),
    ("fed", 6),
    ("ecb", 5),
    ("boj", 5),
    ("default", 7),
    ("bankruptcy", 8),
    ("recall", 4),
    ("earnings", 5),
    ("guidance", 5),
    ("sec", 5),
    ("doj", 5),
    ("antitrust", 6),
    ("merger", 5),
    ("acquisition", 5),
    ("ipo", 4),
    ("ai", 4),
    ("chip", 4),
    ("semiconductor", 5),
]

class NewsModule:
    def __init__(self, db):
        self.collection = db['news']
        self.collection.create_index("title", unique=True)
        print("Initialized News Module")

    def _clean(self, s: str) -> str:
        s = (s or "").strip()
        s = re.sub(r"\s+", " ", s)
        return s

    def analyze_sentiment(self, text):
        analysis = TextBlob(text)
        return analysis.sentiment.polarity

    def generate_tags(self, text, ticker):
        tags = [ticker] if ticker else []
        t = (text or "").lower()
        if "tech" in t or "semiconductor" in t or "ai" in t or "chip" in t:
            tags.append("Technology")
        if "earnings" in t or "revenue" in t or "profit" in t or "guidance" in t:
            tags.append("Finance")
        if "crypto" in t or "bitcoin" in t or "ethereum" in t:
            tags.append("Crypto")
        if "fed" in t or "rate" in t or "inflation" in t or "cpi" in t or "gdp" in t:
            tags.append("Macro")
        if any(k in t for k in ["iran", "israel", "gaza", "ukraine", "russia", "china", "taiwan", "sanction", "tariff", "embargo", "nuclear"]):
            tags.append("Geopolitics")
        if any(k in t for k in ["oil", "opec", "gas", "lng", "copper", "uranium", "rare earth"]):
            tags.append("Natural Resources")
        if len(tags) == 0:
            tags.append("General")
        deduped = []
        seen = set()
        for tag in tags:
            if tag and tag not in seen:
                seen.add(tag)
                deduped.append(tag)
        return deduped

    def _score_headline(self, title: str, source_name: str) -> int:
        t = self._clean(title).lower()
        score = 0
        for kw, w in HIGH_IMPACT_KEYWORDS:
            if kw in t:
                score += w
        if source_name.lower().startswith("google"):
            score += 2
        if "live" in t:
            score += 2
        if "developing" in t:
            score += 3
        if len(t) >= 90:
            score += 1
        if len(t) < 35:
            score -= 2
        return score

    def _fingerprint(self, title: str, url: str, source: str) -> str:
        raw = f"{source}||{title}||{url}".encode("utf-8", errors="ignore")
        return hashlib.sha1(raw).hexdigest()

    def _parse_rss(self, xml_text: str) -> List[Dict[str, Any]]:
        root = ET.fromstring(xml_text)
        out: List[Dict[str, Any]] = []
        for item in root.findall(".//item"):
            title = item.findtext("title") or ""
            link = item.findtext("link") or ""
            pub = item.findtext("pubDate") or ""
            out.append({"title": title, "link": link, "pubDate": pub})
        return out

    def fetch_rss_headlines(self) -> int:
        inserted = 0
        headers = {
            "User-Agent": f"ScopeMonitor/1.0 (+https://localhost) {random.randint(1000,9999)}"
        }

        for source_name, url in RSS_FEEDS:
            try:
                res = requests.get(url, headers=headers, timeout=15)
                if res.status_code != 200:
                    continue
                items = self._parse_rss(res.text)
                if not items:
                    continue

                for it in items:
                    raw_title = self._clean(it.get("title") or "")
                    link = self._clean(it.get("link") or "")
                    if not raw_title or not link:
                        continue

                    score = self._score_headline(raw_title, source_name)
                    if score < 8:
                        continue

                    pub_dt = None
                    pub_raw = self._clean(it.get("pubDate") or "")
                    if pub_raw:
                        try:
                            pub_dt = parsedate_to_datetime(pub_raw)
                        except Exception:
                            pub_dt = None
                    if pub_dt is None:
                        pub_dt = datetime.now()

                    title = f"[{source_name}] {raw_title}"
                    sentiment_score = self.analyze_sentiment(raw_title)
                    tags = self.generate_tags(raw_title, "")

                    article = {
                        "title": title,
                        "content": f"{raw_title}. Read more at {link}",
                        "source": source_name,
                        "url": link,
                        "timestamp": pub_dt,
                        "sentiment": sentiment_score,
                        "tags": tags,
                        "relevance": score,
                        "fingerprint": self._fingerprint(raw_title, link, source_name),
                    }

                    try:
                        result = self.collection.update_one(
                            {"title": title},
                            {"$setOnInsert": article},
                            upsert=True
                        )
                        if result.upserted_id:
                            inserted += 1
                    except Exception:
                        continue
            except Exception:
                continue

        return inserted

    def fetch_news_for_ticker(self, ticker):
        print(f"[{datetime.now()}] Fetching news for {ticker} using Finviz...")
        try:
            stock = finvizfinance(ticker)
            news_df = stock.ticker_news()
            
            if news_df is None or news_df.empty:
                print(f"No news found for {ticker}")
                return

            news_records = news_df.to_dict('records')
            
            for item in news_records:
                title = item.get('Title')
                link = item.get('Link')
                dt_raw = item.get('Date')
                
                title = self._clean(title or "")
                link = self._clean(link or "")
                if not title or not link:
                    continue

                ts = datetime.now()
                try:
                    if isinstance(dt_raw, datetime):
                        ts = dt_raw
                    elif isinstance(dt_raw, pd.Timestamp):
                        ts = dt_raw.to_pydatetime()
                    elif isinstance(dt_raw, str):
                        ds = self._clean(dt_raw)
                        for fmt in ["%b-%d-%y %I:%M%p", "%b-%d-%y"]:
                            try:
                                ts = datetime.strptime(ds, fmt)
                                break
                            except Exception:
                                pass
                except Exception:
                    ts = datetime.now()

                sentiment_score = self.analyze_sentiment(title)
                tags = self.generate_tags(title, ticker)
                relevance = self._score_headline(title, "Finviz Aggregated")
                
                article = {
                    "title": title,
                    "content": f"{title}. Read more at {link}",
                    "source": "Finviz Aggregated",
                    "url": link,
                    "timestamp": ts,
                    "sentiment": sentiment_score,
                    "tags": tags,
                    "related_ticker": ticker,
                    "relevance": relevance,
                }

                try:
                    result = self.collection.update_one(
                        {"title": title},
                        {"$setOnInsert": article},
                        upsert=True
                    )
                    if result.upserted_id:
                        print(f"News: Inserted new article for {ticker}: {title[:30]}... | Sentiment: {sentiment_score:.2f}")
                except Exception as e:
                    print(f"Error inserting article: {e}")

        except Exception as e:
            print(f"Error fetching news for {ticker}: {e}")

    def fetch_all_news(self):
        print(f"[{datetime.now()}] Starting news fetch cycle...")
        try:
            n = self.fetch_rss_headlines()
            if n:
                print(f"News: inserted {n} breaking headlines (RSS)")
        except Exception as e:
            print(f"Error fetching RSS headlines: {e}")
        for ticker in TICKERS:
            self.fetch_news_for_ticker(ticker)
            time.sleep(2)
