import time
from datetime import datetime
from finvizfinance.group.overview import Overview
from pymongo import MongoClient

class SectorModule:
    def __init__(self, db):
        self.collection = db['sector_performance']
        print("Initialized Sector Module")

    def fetch_sector_performance(self):
        print(f"[{datetime.now()}] Fetching Sector Performance...")
        try:
            # Group by Sector
            fg = Overview() 
            # Default view is Sector, but let's be explicit if needed or just use default
            # finvizfinance.group.overview.Overview defaults to group='Sector'
            
            df = fg.screener_view(group='Sector')
            
            if df is None or df.empty:
                print("No sector data found.")
                return

            records = df.to_dict('records')
            batch_time = datetime.now()
            
            for record in records:
                record['fetched_at'] = batch_time
                # Upsert based on Sector Name
                self.collection.update_one(
                    {"Name": record['Name']}, # 'Name' is usually the Sector name in the DF
                    {"$set": record},
                    upsert=True
                )
            
            print(f"Sector: Updated {len(records)} sectors.")
            
        except Exception as e:
            print(f"Error in Sector Module: {e}")
