import time
from datetime import datetime
from finvizfinance.quote import finvizfinance
from pymongo import MongoClient
from textblob import TextBlob
import pandas as pd

# Stocks to track for news (default list)
TICKERS = ["AAPL", "GOOGL", "TSLA", "MSFT", "AMZN", "NVDA", "AMD"]

class NewsModule:
    def __init__(self, db):
        self.collection = db['news']
        self.collection.create_index("title", unique=True)
        print("Initialized News Module")

    def analyze_sentiment(self, text):
        analysis = TextBlob(text)
        return analysis.sentiment.polarity

    def generate_tags(self, text, ticker):
        tags = [ticker]
        if "Tech" in text or "Semiconductor" in text or "AI" in text:
            tags.append("Technology")
        if "Earnings" in text or "Revenue" in text:
            tags.append("Finance")
        if "Crypto" in text or "Bitcoin" in text:
            tags.append("Crypto")
        if "Fed" in text or "rate" in text or "inflation" in text:
            tags.append("Macro")
        if len(tags) == 1: 
            tags.append("General")
        return tags

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
                
                if not title:
                    continue

                sentiment_score = self.analyze_sentiment(title)
                tags = self.generate_tags(title, ticker)
                
                article = {
                    "title": title,
                    "content": f"{title}. Read more at {link}",
                    "source": "Finviz Aggregated",
                    "url": link,
                    "timestamp": datetime.now(), 
                    "sentiment": sentiment_score,
                    "tags": tags,
                    "related_ticker": ticker
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
        for ticker in TICKERS:
            self.fetch_news_for_ticker(ticker)
            time.sleep(2)
