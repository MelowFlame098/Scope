import time
from datetime import datetime
from finvizfinance.screener.overview import Overview
from finvizfinance.screener.valuation import Valuation
from finvizfinance.screener.financial import Financial
import pandas as pd
from pymongo import MongoClient

class ScreenerModule:
    def __init__(self, db):
        self.collection = db['screener_results']
        # self.collection.create_index("Ticker", unique=True) # Ticker should be unique per run, but we might keep history
        print("Initialized Screener Module")

    def run_screen(self):
        print(f"[{datetime.now()}] Running Stock Screener...")
        try:
            # Filters can be customized. For now, let's just get the top stocks by default or specific signal
            filters_dict = {'Signal': 'Top Gainers'}
            
            # Initialize Screeners
            foverview = Overview()
            fvaluation = Valuation()
            ffinancial = Financial()
            
            foverview.set_filter(filters_dict=filters_dict)
            fvaluation.set_filter(filters_dict=filters_dict)
            ffinancial.set_filter(filters_dict=filters_dict)
            
            print("Fetching Overview...")
            df_overview = foverview.screener_view()
            print("Fetching Valuation...")
            df_valuation = fvaluation.screener_view()
            print("Fetching Financial...")
            df_financial = ffinancial.screener_view()
            
            if df_overview is None or df_overview.empty:
                print("No stocks found matching filters.")
                return

            # Merge DataFrames
            # Overview has: Ticker, Company, Sector, Industry, Country, Market Cap, P/E, Price, Change, Volume
            # Valuation has: Ticker, P/B, EPS (ttm), Dividend Yield
            # Financial has: Ticker, ROE, Profit Margin, Debt/Eq, Sales
            
            # Select only needed columns to avoid collisions (except Ticker)
            val_cols = ['Ticker', 'P/B', 'EPS (ttm)', 'Dividend Yield']
            fin_cols = ['Ticker', 'ROE', 'Profit Margin', 'Debt/Eq', 'Sales']
            
            # Ensure columns exist (in case finviz changes them or data is missing)
            existing_val_cols = [c for c in val_cols if c in df_valuation.columns]
            existing_fin_cols = [c for c in fin_cols if c in df_financial.columns]
            
            df = df_overview
            
            if not df_valuation.empty:
                df = pd.merge(df, df_valuation[existing_val_cols], on='Ticker', how='left')
                
            if not df_financial.empty:
                df = pd.merge(df, df_financial[existing_fin_cols], on='Ticker', how='left')

            # Rename columns to match MongoDB/Go expectations if necessary
            # Go struct tags: "Total Debt/Eq" -> json "debt"
            # Finviz likely returns "Debt/Eq". I'll rename it to "Total Debt/Eq" to match my Go struct mapping
            # or update Go struct mapping.
            # My Go struct has `bson:"Total Debt/Eq"`.
            # If dataframe has `Debt/Eq`, I should rename it.
            
            if 'Debt/Eq' in df.columns:
                df.rename(columns={'Debt/Eq': 'Total Debt/Eq'}, inplace=True)

            # Ensure all data is string format to match Go backend structs
            # This prevents BSON decoding errors if Go expects string but gets float
            df = df.astype(str)

            records = df.to_dict('records')
            
            # Timestamp for this batch
            batch_time = datetime.now()
            
            for record in records:
                record['fetched_at'] = batch_time
                record['strategy'] = 'Top Gainers'
                
                # Convert NaNs to None/String if necessary (PyMongo handles NaNs but JSON might prefer nulls)
                # For simplicity, let's keep as is, Go driver handles BSON
                
                # Update or Insert based on Ticker and Strategy
                self.collection.update_one(
                    {"Ticker": record['Ticker'], "strategy": "Top Gainers"},
                    {"$set": record},
                    upsert=True
                )
                
            print(f"Screener: Updated {len(records)} stocks with comprehensive data.")
            
        except Exception as e:
            print(f"Error in Screener Module: {e}")
