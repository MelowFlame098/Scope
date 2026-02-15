import time
import schedule
from pymongo import MongoClient
from app.screener import ScreenerModule
from app.insider import InsiderModule
from app.sector import SectorModule
from app.news import NewsModule
from app.fundamentals import run_fundamentals_batch

import os

# Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://user:password@localhost:27017")
DB_NAME = "scope_mongo"

class AIService:
    def __init__(self):
        try:
            self.client = MongoClient(MONGO_URI)
            self.db = self.client[DB_NAME]
            print(f"Connected to MongoDB: {DB_NAME}")
            
            # Initialize Modules
            self.screener = ScreenerModule(self.db)
            self.insider = InsiderModule(self.db)
            self.sector = SectorModule(self.db)
            self.news = NewsModule(self.db)
            # Expanded ticker list to match chart/screener/movers/ETFs
            self.fundamentals_tickers = [
                # Tech / Blue Chips
                "AAPL", "GOOGL", "TSLA", "MSFT", "AMZN", 
                # NASDAQ Gainers
                "MARA", "RIVN", "DKNG", "LCID", "PLUG", "SOFI",
                # NASDAQ Losers
                "PTON", "COIN", "ZM", "ROKU", "DOCU", "SNOW",
                # ETFs / Indices
                "QQQ", "ICLN", "VEGN", "SOXX", "DTCR", "VGT", "XRT"
            ]
            
        except Exception as e:
            print(f"Failed to connect to MongoDB: {e}")

    def run_all_tasks(self):
        print("Running all scheduled tasks...")
        try:
            self.screener.run_screen()
        except Exception as e:
            print(f"Error running screener: {e}")
            
        try:
            self.insider.fetch_insider_trades()
        except Exception as e:
            print(f"Error running insider: {e}")
            
        try:
            self.sector.fetch_sector_performance()
        except Exception as e:
            print(f"Error running sector: {e}")
            
        try:
            self.news.fetch_all_news()
        except Exception as e:
            print(f"Error running news: {e}")
        
        # Dynamic Ticker Expansion: Get tickers from Screener Results in DB
        try:
            screener_tickers = self.db.screener_results.distinct("Ticker")
            all_tickers = list(set(self.fundamentals_tickers + screener_tickers))
            print(f"Fundamentals: Processing {len(all_tickers)} tickers (including {len(screener_tickers)} from screener)...")
            
            run_fundamentals_batch(self.db, all_tickers)
        except Exception as e:
            print(f"Error running fundamentals: {e}")

    def run_fundamentals_dynamic(self):
        print(f"[{datetime.now()}] Running Dynamic Fundamentals Batch...")
        try:
            screener_tickers = self.db.screener_results.distinct("Ticker")
            all_tickers = list(set(self.fundamentals_tickers + screener_tickers))
            print(f"Fundamentals: Processing {len(all_tickers)} tickers (including {len(screener_tickers)} from screener)...")
            run_fundamentals_batch(self.db, all_tickers)
        except Exception as e:
            print(f"Error running dynamic fundamentals: {e}")

    def run(self):
        print("AI Service Started (Modular). Scheduling tasks...", flush=True)
        
        # Run immediately once
        self.run_all_tasks()
        
        # Schedule tasks
        # News every 15 mins
        schedule.every(15).minutes.do(self.news.fetch_all_news)
        
        # Screener every 1 hour
        schedule.every(1).hours.do(self.screener.run_screen)
        
        # Insider every 4 hours
        schedule.every(4).hours.do(self.insider.fetch_insider_trades)
        
        # Sector every 4 hours
        schedule.every(4).hours.do(self.sector.fetch_sector_performance)
        
        # Fundamentals every 4 hours (Dynamic)
        schedule.every(4).hours.do(self.run_fundamentals_dynamic)
        
        while True:
            schedule.run_pending()
            time.sleep(1)

if __name__ == "__main__":
    service = AIService()
    service.run()
