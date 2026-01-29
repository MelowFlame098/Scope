import time
import schedule
from pymongo import MongoClient
from app.screener import ScreenerModule
from app.insider import InsiderModule
from app.sector import SectorModule
from app.news import NewsModule

# Configuration
MONGO_URI = "mongodb://user:password@localhost:27017"
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
            
        except Exception as e:
            print(f"Failed to connect to MongoDB: {e}")

    def run_all_tasks(self):
        print("Running all scheduled tasks...")
        self.screener.run_screen()
        self.insider.fetch_insider_trades()
        self.sector.fetch_sector_performance()
        self.news.fetch_all_news()

    def run(self):
        print("AI Service Started (Modular). Scheduling tasks...")
        
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
        
        while True:
            schedule.run_pending()
            time.sleep(1)

if __name__ == "__main__":
    service = AIService()
    service.run()
