import time
from datetime import datetime
from finvizfinance.screener.overview import Overview
from pymongo import MongoClient

class ScreenerModule:
    def __init__(self, db):
        self.collection = db['screener_results']
        # self.collection.create_index("Ticker", unique=True) # Ticker should be unique per run, but we might keep history
        print("Initialized Screener Module")

    def run_screen(self):
        print(f"[{datetime.now()}] Running Stock Screener...")
        try:
            # Example: Top gainers with high volume
            # Filters can be customized. For now, let's just get the top stocks by default or specific signal
            foverview = Overview()
            
            # Setting filters: e.g., 'Signal': 'Top Gainers'
            filters_dict = {'Signal': 'Top Gainers'}
            foverview.set_filter(filters_dict=filters_dict)
            
            df = foverview.screener_view()
            if df is None or df.empty:
                print("No stocks found matching filters.")
                return

            records = df.to_dict('records')
            
            # Timestamp for this batch
            batch_time = datetime.now()
            
            for record in records:
                record['fetched_at'] = batch_time
                record['strategy'] = 'Top Gainers'
                
                # Update or Insert based on Ticker and Strategy
                self.collection.update_one(
                    {"Ticker": record['Ticker'], "strategy": "Top Gainers"},
                    {"$set": record},
                    upsert=True
                )
                
            print(f"Screener: Updated {len(records)} stocks.")
            
        except Exception as e:
            print(f"Error in Screener Module: {e}")
