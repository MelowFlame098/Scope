import time
from datetime import datetime
from finvizfinance.insider import Insider
from pymongo import MongoClient

class InsiderModule:
    def __init__(self, db):
        self.collection = db['insider_trades']
        # self.collection.create_index("link", unique=True) # Assuming link or combination of fields is unique
        print("Initialized Insider Module")

    def fetch_insider_trades(self):
        print(f"[{datetime.now()}] Fetching Insider Trades...")
        try:
            minsider = Insider(option='top owner trade') # options: latest, top week, top owner trade
            df = minsider.get_insider()
            
            if df is None or df.empty:
                print("No insider trades found.")
                return

            records = df.to_dict('records')
            
            count = 0
            for record in records:
                # Create a unique ID or use provided fields if available. 
                # Finviz insider table usually has Date, Ticker, Owner, Relationship, Transaction, Cost, #Shares, Value, Total Shares, SEC Form 4
                
                # We can construct a unique query based on Ticker, Owner, Date, and Value
                query = {
                    "Ticker": record.get('Ticker'),
                    "Owner": record.get('Owner'),
                    "Date": record.get('Date'),
                    "Value ($)": record.get('Value ($)')
                }
                
                record['fetched_at'] = datetime.now()
                
                result = self.collection.update_one(
                    query,
                    {"$setOnInsert": record},
                    upsert=True
                )
                
                if result.upserted_id:
                    count += 1
            
            print(f"Insider: Inserted {count} new trades.")
            
        except Exception as e:
            print(f"Error in Insider Module: {e}")
