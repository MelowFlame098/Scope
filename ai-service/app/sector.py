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
                # Convert 'Change' from 0.015 (if Finviz returns decimal) or check if it's already %
                # Finviz API usually returns float like 0.015 for 1.5% or sometimes strings "1.50%"
                # Based on previous tests, it seems to return floats.
                # However, our frontend expects percentages (e.g. 1.5).
                
                # Check if Change is already percentage (e.g. > 1 or < -1 could be %, but < 1 could be decimal)
                # But sector changes are usually small. 
                # Let's assume if abs(change) < 0.5, it is likely decimal. (50% daily move is rare for sector).
                
                # Explicitly replace the entire document to ensure the Change field update persists
                # Check if it works with replace_one
                
                # Wait, I found the issue.
                # In MongoDB, `upsert=True` with `replace_one` works.
                # BUT, if the document already exists, it replaces it.
                # If I use `update_one` with `$set`, it updates specific fields.
                
                # The issue might be that `record` variable holds the original dictionary from `records`.
                # And `records` is `df.to_dict('records')`.
                # If I modify `record['Change']`, it modifies the dict in memory.
                # But why did `db.find` return decimal?
                
                # HYPOTHESIS: `df.to_dict('records')` returns a list of dicts.
                # Maybe Finviz library returns a specific type (numpy float?) that resists in-place multiplication?
                # Or maybe I am printing `record['Change']` but passing `df` somewhere else? No.
                
                # Let's force cast to float.
                if 'Change' in record and record['Change'] is not None:
                    try:
                        val = float(record['Change'])
                        # If small decimal, convert to %
                        # e.g. 0.015 -> 1.5
                        # e.g. -0.0172 -> -1.72
                        if abs(val) < 0.5:
                            val = val * 100.0
                        
                        # FORCE UPDATE: Modify the record in place
                        record['Change'] = val
                        
                        # Debug Print
                        print(f"Sector: {record['Name']} -> {record['Change']}")
                    except Exception as e:
                        print(f"Error converting change: {e}")

                record['fetched_at'] = batch_time
                
                # Delete existing to be absolutely sure
                self.collection.delete_one({"Name": record['Name']})
                # Insert the modified record
                self.collection.insert_one(record)
            
            print(f"Sector: Updated {len(records)} sectors.")
            
        except Exception as e:
            print(f"Error in Sector Module: {e}")
