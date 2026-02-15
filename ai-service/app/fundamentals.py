import time
from datetime import datetime

import pandas as pd
import yfinance as yf
from pymongo import MongoClient


class FundamentalsModule:
    def __init__(self, db):
        self.collection = db["fundamentals"]
        print("Initialized Fundamentals Module")

    def fetch_fundamentals_for_symbol(self, ticker):
        print(f"[{datetime.now()}] Fetching fundamentals for {ticker} via finvizfinance and yfinance...", flush=True)
        try:
            # --- 1. Finviz Fundamentals (Part 1 - Raw Fundamentals) ---
            # Using finvizfinance.quote to get extensive fundamental table
            from finvizfinance.quote import finvizfinance
            
            fund_data = {}
            try:
                # Add a small delay to avoid rate limiting
                import time
                time.sleep(1)
                
                stock_finviz = finvizfinance(ticker)
                fund_data = stock_finviz.ticker_fundament()
                print(f"Finviz data fetched for {ticker}: {len(fund_data)} fields. Sample Keys: {list(fund_data.keys())[:10]}", flush=True)
            except Exception as e:
                print(f"Finviz fetch failed for {ticker}: {e}", flush=True)
                # Continue with yfinance even if finviz fails
            
            # Map Finviz keys to our schema
            # We will store the raw Finviz dictionary directly under 'finviz_raw' 
            # and also map key metrics to top-level fields for easier access.
            
            # Helper to parse Finviz strings (e.g. "1.50B", "2.5%", "100.00")
            def parse_finviz_val(val_str):
                if val_str is None:
                    return None
                if isinstance(val_str, (int, float)):
                    return float(val_str)
                if not isinstance(val_str, str):
                    return None
                
                val_str = val_str.strip()
                if val_str == '-':
                    return None

                # Remove commas
                val_str = val_str.replace(',', '')
                
                # Remove %
                is_pct = False
                if val_str.endswith('%'):
                    val_str = val_str[:-1]
                    is_pct = True
                
                # Handle B/M/K suffixes
                multiplier = 1.0
                if val_str.endswith('B'):
                    multiplier = 1_000_000_000
                    val_str = val_str[:-1]
                elif val_str.endswith('M'):
                    multiplier = 1_000_000
                    val_str = val_str[:-1]
                elif val_str.endswith('K'):
                    multiplier = 1_000
                    val_str = val_str[:-1]
                
                try:
                    val = float(val_str)
                    if is_pct:
                        val = val / 100.0 # Store 2.5% as 0.025
                    else:
                        val = val * multiplier
                    return val
                except:
                    return None

            # --- 2. yfinance Quarterly Data (for time-series/growth) ---
            stock_yf = yf.Ticker(ticker)
            q_fin = stock_yf.quarterly_financials
            q_bs = stock_yf.quarterly_balance_sheet
            q_cf = stock_yf.quarterly_cashflow
            
            # Annual Data
            a_fin = stock_yf.financials
            a_bs = stock_yf.balance_sheet
            a_cf = stock_yf.cashflow

            now_dt = datetime.utcnow()
            
            # We will process quarters as before, but ENRICH them with the latest Finviz snapshot data
            # Since Finviz data is "current" (snapshot), we attach it to the "current" timeframe record 
            # and potentially the latest quarter record.
            
            # Sort columns descending by date
            sorted_dates = sorted(q_fin.columns, reverse=True) if q_fin is not None and not q_fin.empty else []
            
            # --- CALCULATED METRICS (Part 2) Implementation ---
            # We need values to calculate ratios. We'll use the latest TTM or Quarterly data from yfinance 
            # combined with market price/cap from Finviz.
            
            market_cap = parse_finviz_val(fund_data.get('Market Cap'))
            price = parse_finviz_val(fund_data.get('Price'))
            
            # Helper to safely get latest value from yfinance DF
            def get_latest(df, row):
                if df is not None and not df.empty and row in df.index:
                    return df.loc[row].iloc[0] # Latest quarter
                return 0.0

            # Calculate metrics if not in Finviz
            # Example: FCF Yield = FCF / Market Cap
            fcf = get_latest(q_cf, "Free Cash Flow")
            fcf_yield = (fcf * 4) / market_cap if market_cap and fcf else None # Annualized FCF estimate
            
            # Enterprise Value (Finviz usually has it, but let's calculate/check)
            # EV = Market Cap + Total Debt - Cash
            total_debt = get_latest(q_bs, "Total Debt")
            cash = get_latest(q_bs, "Cash And Cash Equivalents")
            enterprise_value = market_cap + total_debt - cash if market_cap else None
            
            # Create a comprehensive metrics dictionary
            # Ensure keys match frontend exactly (lowercase, snake_case)
            comprehensive_metrics = {
                # --- 1. Valuation Metrics ---
                "market_cap": parse_finviz_val(fund_data.get('Market Cap')),
                "enterprise_value": parse_finviz_val(fund_data.get('Enterprise Value')), # Check exact key
                "pe_ratio": parse_finviz_val(fund_data.get('P/E')),
                "forward_pe": parse_finviz_val(fund_data.get('Forward P/E')),
                "peg_ratio": parse_finviz_val(fund_data.get('PEG')),
                "ps_ratio": parse_finviz_val(fund_data.get('P/S')),
                "pb_ratio": parse_finviz_val(fund_data.get('P/B')),
                "price_to_cash": parse_finviz_val(fund_data.get('P/C')), # Missing
                "price_to_fcf": parse_finviz_val(fund_data.get('P/FCF')), # Missing
                "ev_ebitda": parse_finviz_val(fund_data.get('EV/EBITDA')), # Missing (EV / EBITDA)
                # EV/Revenue is calculated usually or in finviz as EV/Sales sometimes? Let's check finviz keys.
                # If not present, calculate: EV / Sales
                "ev_revenue": parse_finviz_val(fund_data.get('EV/Sales')) if fund_data.get('EV/Sales') else (
                    (enterprise_value / parse_finviz_val(fund_data.get('Sales'))) if enterprise_value and parse_finviz_val(fund_data.get('Sales')) else None
                ), 

                # --- 2. Profitability Metrics ---
                "gross_margin": parse_finviz_val(fund_data.get('Gross Margin')),
                "operating_margin": parse_finviz_val(fund_data.get('Oper. Margin')),
                "profit_margin": parse_finviz_val(fund_data.get('Profit Margin')),
                "roa": parse_finviz_val(fund_data.get('ROA')),
                "roe": parse_finviz_val(fund_data.get('ROE')),
                "roi": parse_finviz_val(fund_data.get('ROIC')), # Finviz uses ROIC usually

                # --- 3. Growth Metrics ---
                "eps_growth_past_5y": parse_finviz_val(fund_data.get('EPS past 3/5Y').split()[1]) if fund_data.get('EPS past 3/5Y') else None, # "6.89% 17.91%"
                "eps_growth_next_5y": parse_finviz_val(fund_data.get('EPS next 5Y')),
                "sales_growth_past_5y": parse_finviz_val(fund_data.get('Sales past 3/5Y').split()[1]) if fund_data.get('Sales past 3/5Y') else None,
                "eps_growth_this_year": parse_finviz_val(fund_data.get('EPS this Y')),
                "eps_growth_next_year": parse_finviz_val(fund_data.get('EPS next Y Percentage')), # Key is 'EPS next Y Percentage' or 'EPS next Y' (EPS next Y is value, Percentage is growth)
                "eps_growth_qtr_over_qtr": parse_finviz_val(fund_data.get('EPS Q/Q')),
                "sales_growth_qtr_over_qtr": parse_finviz_val(fund_data.get('Sales Q/Q')),
                
                # --- 4. Financial Health & Liquidity ---
                "current_ratio": parse_finviz_val(fund_data.get('Current Ratio')),
                "quick_ratio": parse_finviz_val(fund_data.get('Quick Ratio')),
                "debt_to_equity": parse_finviz_val(fund_data.get('Debt/Eq')),
                "lt_debt_to_equity": parse_finviz_val(fund_data.get('LT Debt/Eq')),
                "total_debt": total_debt,
                "total_cash": cash,
                "book_value_per_share": parse_finviz_val(fund_data.get('Book/sh')),

                # --- 5. Cash Flow Metrics ---
                "operating_cash_flow": get_latest(q_cf, "Operating Cash Flow"),
                "free_cash_flow": fcf,
                "cash_per_share": parse_finviz_val(fund_data.get('Cash/sh')),

                # --- 6. Earnings & Analyst Data ---
                "eps_ttm": parse_finviz_val(fund_data.get('EPS (ttm)')),
                "eps_next_q": parse_finviz_val(fund_data.get('EPS next Q')),
                "eps_next_y": parse_finviz_val(fund_data.get('EPS next Y')),
                "eps_surprise": parse_finviz_val(fund_data.get('EPS/Sales Surpr.').split()[0]) if fund_data.get('EPS/Sales Surpr.') else None, # "6.24% 3.88%"
                "analyst_recom": parse_finviz_val(fund_data.get('Recom')),
                "target_price": parse_finviz_val(fund_data.get('Target Price')),
                "earnings_date": fund_data.get('Earnings'),

                # --- 7. Dividends ---
                "dividend_yield": parse_finviz_val(fund_data.get('Dividend Est.').split('(')[1].replace(')', '')) if '(' in (fund_data.get('Dividend Est.') or '') else parse_finviz_val(fund_data.get('Dividend TTM').split('(')[1].replace(')', '')) if '(' in (fund_data.get('Dividend TTM') or '') else None,
                "payout_ratio": parse_finviz_val(fund_data.get('Payout')),
                "dividend_growth": parse_finviz_val(fund_data.get('Dividend Gr. 3/5Y').split()[1]) if fund_data.get('Dividend Gr. 3/5Y') else None,
                "ex_dividend_date": fund_data.get('Dividend Ex-Date'),

                # --- 8. Ownership & Share Structure ---
                "insider_own": parse_finviz_val(fund_data.get('Insider Own')),
                "inst_own": parse_finviz_val(fund_data.get('Inst Own')),
                "insider_trans": parse_finviz_val(fund_data.get('Insider Trans')),
                "inst_trans": parse_finviz_val(fund_data.get('Inst Trans')),
                "float_shares": parse_finviz_val(fund_data.get('Shs Float')),
                "shares_outstanding": parse_finviz_val(fund_data.get('Shs Outstand')),
                "short_float": parse_finviz_val(fund_data.get('Short Float')),
                "short_ratio": parse_finviz_val(fund_data.get('Short Ratio')),

                # --- 9. Risk & Volatility ---
                "beta": parse_finviz_val(fund_data.get('Beta')),
                "volatility_week": parse_finviz_val(fund_data.get('Volatility W')), # Direct key
                "volatility_month": parse_finviz_val(fund_data.get('Volatility M')), # Direct key
                "atr": parse_finviz_val(fund_data.get('ATR (14)')), # Direct key

                # --- 10. Trading Liquidity ---
                "avg_volume": parse_finviz_val(fund_data.get('Avg Volume')),
                "rel_volume": parse_finviz_val(fund_data.get('Rel Volume')),

                # --- 11. Company Information ---
                "sector": fund_data.get('Sector'),
                "industry": fund_data.get('Industry'),
                "country": fund_data.get('Country'),
                "exchange": fund_data.get('Exchange'),
                "ipo_date": fund_data.get('IPO'),
                "employees": parse_finviz_val(fund_data.get('Employees')),
                
                # --- PART 2: Advanced Calculated Ratios ---
                
                # A. Valuation & Yield Ratios
                "earnings_yield": (1.0 / parse_finviz_val(fund_data.get('P/E'))) if parse_finviz_val(fund_data.get('P/E')) else None,
                "forward_earnings_yield": (1.0 / parse_finviz_val(fund_data.get('Forward P/E'))) if parse_finviz_val(fund_data.get('Forward P/E')) else None,
                "fcf_yield": fcf_yield,
                # Operating Cash Flow Yield = OCF / Market Cap
                "ocf_yield": (get_latest(q_cf, "Operating Cash Flow") * 4 / market_cap) if market_cap and get_latest(q_cf, "Operating Cash Flow") else None,
                # EBITDA Yield = EBITDA / Enterprise Value (Inverse of EV/EBITDA)
                "ebitda_yield": (1.0 / parse_finviz_val(fund_data.get('EV/EBITDA'))) if parse_finviz_val(fund_data.get('EV/EBITDA')) else None,
                "revenue_yield": (parse_finviz_val(fund_data.get('Sales')) / enterprise_value) if enterprise_value and parse_finviz_val(fund_data.get('Sales')) else None,
                # Book-to-Market = 1 / (P/B)
                "book_to_market": (1.0 / parse_finviz_val(fund_data.get('P/B'))) if parse_finviz_val(fund_data.get('P/B')) else None,
                # PEG Adjusted Yield (Earnings Yield / Growth) -> Kind of inverse PEG? Or PEG is P/E / Growth. 
                # Formula given: Earnings Yield / Growth Rate. 
                # Earnings Yield = E/P. Growth = G. Ratio = (E/P)/G = E/(P*G).
                # PEG = (P/E)/G = P/(E*G). 
                # So this is 1/PEG * (1/E^2)? No.
                # Let's stick to literal: (1/PE) / (EPS this Y / 100)
                "price_to_growth_adj_yield": ((1.0 / parse_finviz_val(fund_data.get('P/E'))) / parse_finviz_val(fund_data.get('EPS this Y'))) if parse_finviz_val(fund_data.get('P/E')) and parse_finviz_val(fund_data.get('EPS this Y')) else None,

                # B. Profitability & Efficiency
                "asset_turnover": (parse_finviz_val(fund_data.get('Sales')) / get_latest(q_bs, "Total Assets")) if get_latest(q_bs, "Total Assets") else None,
                # Operating Efficiency = Operating Income / Revenue (Same as Operating Margin)
                "operating_efficiency": parse_finviz_val(fund_data.get('Oper. Margin')), 
                # ROIC (Finviz has it)
                "roic": parse_finviz_val(fund_data.get('ROIC')),
                # CROIC = FCF / Invested Capital. Invested Capital ~ Total Equity + Total Debt - Cash? Or just Equity + Debt.
                # Let's use Equity + Debt.
                "croic": (fcf * 4 / (parse_finviz_val(fund_data.get('Market Cap')) / parse_finviz_val(fund_data.get('P/B')) + total_debt)) if parse_finviz_val(fund_data.get('P/B')) and total_debt else None, # Approx
                
                # C. Growth & Quality
                # SGR = ROE * (1 - Payout)
                "sgr": (parse_finviz_val(fund_data.get('ROE')) * (1 - (parse_finviz_val(fund_data.get('Payout')) or 0))) if parse_finviz_val(fund_data.get('ROE')) else None,
                
                # D. Leverage & Risk
                "net_debt_to_ebitda": ((total_debt - cash) / (parse_finviz_val(fund_data.get('Enterprise Value')) / parse_finviz_val(fund_data.get('EV/EBITDA')))) if parse_finviz_val(fund_data.get('EV/EBITDA')) else None, # Deriving EBITDA from EV/EBITDA
                "liquidity_cushion": (cash / total_debt) if total_debt and total_debt > 0 else None,

                # --- PART 2: Missing Advanced Ratios ---
                
                # 13. Gross Profit Efficiency = Gross Profit / Assets
                # Gross Profit = Revenue * Gross Margin
                "gross_profit_efficiency": ((parse_finviz_val(fund_data.get('Sales')) * parse_finviz_val(fund_data.get('Gross Margin'))) / get_latest(q_bs, "Total Assets")) if parse_finviz_val(fund_data.get('Sales')) and parse_finviz_val(fund_data.get('Gross Margin')) and get_latest(q_bs, "Total Assets") else None,

                # 15. Earnings Growth Efficiency = EPS Growth / PEG
                "earnings_growth_efficiency": (parse_finviz_val(fund_data.get('EPS this Y')) / parse_finviz_val(fund_data.get('PEG'))) if parse_finviz_val(fund_data.get('PEG')) and parse_finviz_val(fund_data.get('EPS this Y')) else None,

                # 16. Revenue-to-Earnings Growth Ratio = Revenue Growth / EPS Growth
                "revenue_to_earnings_growth": (parse_finviz_val(fund_data.get('Sales Q/Q')) / parse_finviz_val(fund_data.get('EPS Q/Q'))) if parse_finviz_val(fund_data.get('EPS Q/Q')) and parse_finviz_val(fund_data.get('Sales Q/Q')) else None,

                # 17. Cash Conversion Ratio = Operating Cash Flow / Net Income
                "cash_conversion_ratio": (get_latest(q_cf, "Operating Cash Flow") / parse_finviz_val(fund_data.get('Income'))) if parse_finviz_val(fund_data.get('Income')) and get_latest(q_cf, "Operating Cash Flow") else None,

                # 18. Free Cash Flow Conversion = Free Cash Flow / Net Income
                "fcf_conversion": (fcf / parse_finviz_val(fund_data.get('Income'))) if parse_finviz_val(fund_data.get('Income')) and fcf else None,

                # 20. Debt Service Ratio = Operating Cash Flow / Total Debt
                "debt_service_ratio": (get_latest(q_cf, "Operating Cash Flow") / total_debt) if total_debt and get_latest(q_cf, "Operating Cash Flow") else None,

                # 21. Financial Leverage Ratio = Total Assets / Equity
                # Equity = Market Cap / P/B ? Or Total Assets - Total Liab. Let's use Assets / (Assets - Liab)
                "financial_leverage_ratio": (get_latest(q_bs, "Total Assets") / (get_latest(q_bs, "Total Assets") - get_latest(q_bs, "Total Liabilities Net Minority Interest"))) if get_latest(q_bs, "Total Assets") and get_latest(q_bs, "Total Liabilities Net Minority Interest") and (get_latest(q_bs, "Total Assets") - get_latest(q_bs, "Total Liabilities Net Minority Interest")) != 0 else None,

                # 22. Leverage Adjusted Volatility = Beta × Debt to Equity
                "leverage_adjusted_volatility": (parse_finviz_val(fund_data.get('Beta')) * parse_finviz_val(fund_data.get('Debt/Eq'))) if parse_finviz_val(fund_data.get('Beta')) and parse_finviz_val(fund_data.get('Debt/Eq')) else None,

                # 24. Shareholder Yield = Dividend Yield + Buyback Yield
                # Buyback Yield approx = - (Repurchase of Capital Stock / Market Cap)
                # We need Repurchase of Capital Stock from Cash Flow.
                "shareholder_yield": (
                    (parse_finviz_val(fund_data.get('Dividend %')) or 0) + 
                    ((abs(get_latest(q_cf, "Repurchase Of Capital Stock")) * 4 / market_cap) if market_cap and get_latest(q_cf, "Repurchase Of Capital Stock") else 0)
                ) if market_cap else None,

                # 25. Retention Ratio = 1 - Dividend Payout Ratio
                "retention_ratio": (1 - parse_finviz_val(fund_data.get('Payout'))) if parse_finviz_val(fund_data.get('Payout')) else None,

                # 26. Reinvestment Rate = (Capital Expenditure) / Operating Cash Flow
                "reinvestment_rate": (abs(get_latest(q_cf, "Capital Expenditure")) / get_latest(q_cf, "Operating Cash Flow")) if get_latest(q_cf, "Operating Cash Flow") and get_latest(q_cf, "Capital Expenditure") else None,

                # 27. Capital Efficiency = Revenue Growth / Capital Investment (Capex/Sales?) -> Formula says Revenue Growth / Capital Investment.
                # Let's assume Capital Investment ~ Capex.
                "capital_efficiency": (parse_finviz_val(fund_data.get('Sales Q/Q')) / (abs(get_latest(q_cf, "Capital Expenditure")) / parse_finviz_val(fund_data.get('Sales')))) if parse_finviz_val(fund_data.get('Sales')) and get_latest(q_cf, "Capital Expenditure") and parse_finviz_val(fund_data.get('Sales Q/Q')) else None,

                # 28. Insider Buying Intensity = Insider Purchases / Shares Outstanding
                # Proxy: Insider Trans % (Net)
                "insider_buying_intensity": parse_finviz_val(fund_data.get('Insider Trans')),

                # 29. Institutional Accumulation Score = Change in Institutional Ownership
                # Proxy: Inst Trans %
                "institutional_accumulation": parse_finviz_val(fund_data.get('Inst Trans')),

                # 30. Float Turnover Ratio = Volume / Float
                "float_turnover": (parse_finviz_val(fund_data.get('Volume')) / parse_finviz_val(fund_data.get('Shs Float'))) if parse_finviz_val(fund_data.get('Shs Float')) and parse_finviz_val(fund_data.get('Volume')) else None,

                # 31. Volatility-to-Liquidity Ratio = ATR / Average Volume
                "volatility_liquidity_ratio": (parse_finviz_val(fund_data.get('ATR (14)')) / parse_finviz_val(fund_data.get('Avg Volume'))) if parse_finviz_val(fund_data.get('Avg Volume')) and parse_finviz_val(fund_data.get('ATR (14)')) else None,

                # 32. Turnover Stability = Average Volume / Shares Outstanding
                "turnover_stability": (parse_finviz_val(fund_data.get('Avg Volume')) / parse_finviz_val(fund_data.get('Shs Outstand'))) if parse_finviz_val(fund_data.get('Shs Outstand')) and parse_finviz_val(fund_data.get('Avg Volume')) else None,

                # --- Composite Scores (Normalized 0-100 approximations) ---
                # 33. Value Score (Earnings Yield, FCF Yield, Book-to-Market)
                "value_score": (
                    ((1.0/parse_finviz_val(fund_data.get('P/E')) if parse_finviz_val(fund_data.get('P/E')) else 0) * 100 * 0.4) + 
                    ((fcf_yield or 0) * 100 * 0.4) + 
                    ((1.0/parse_finviz_val(fund_data.get('P/B')) if parse_finviz_val(fund_data.get('P/B')) else 0) * 100 * 0.2)
                ) if fcf_yield is not None else None,

                # 34. Quality Score (ROE, Margins, Debt)
                "quality_score": (
                    ((parse_finviz_val(fund_data.get('ROE')) or 0) * 100 * 0.4) +
                    ((parse_finviz_val(fund_data.get('Profit Margin')) or 0) * 100 * 0.4) - 
                    ((parse_finviz_val(fund_data.get('Debt/Eq')) or 0) * 10 * 0.2)
                ),

                # 35. Growth Score (Rev Growth, EPS Growth)
                "growth_score": (
                    ((parse_finviz_val(fund_data.get('Sales Q/Q')) or 0) * 100 * 0.5) +
                    ((parse_finviz_val(fund_data.get('EPS Q/Q')) or 0) * 100 * 0.5)
                ),

                # 36. Low Risk Score (Beta, Volatility) - Inverse
                "low_risk_score": (
                    (1.0 / (parse_finviz_val(fund_data.get('Beta')) or 1.0)) * 50 +
                    (1.0 / (parse_finviz_val(fund_data.get('ATR')) or 1.0)) * 50
                ),

                # 37. Risk-Adjusted Return = Expected Return (say, ROE) / Volatility (ATR normalized?)
                "risk_adjusted_return": (
                    (parse_finviz_val(fund_data.get('ROE')) or 0) / (parse_finviz_val(fund_data.get('Volatility M')) if fund_data.get('Volatility M') else 0.01)
                ) if fund_data.get('Volatility M') else None,
                
                # 38. Fundamental Risk Score = Debt + Volatility + Earnings Stability (Use Debt/Eq + Beta)
                "fundamental_risk_score": (
                    (parse_finviz_val(fund_data.get('Debt/Eq')) or 0) + 
                    (parse_finviz_val(fund_data.get('Beta')) or 0)
                ),

                # Missing Part 1
                "headquarters": fund_data.get('Country'), # Actually Finviz often puts HQ in Country or separate. Finviz finance has 'Country' which is HQ usually.

                # Raw Finviz Data (Backup)
                "finviz_raw": fund_data
            }

            # 3. Update 'Current' Timeframe Record with Comprehensive Snapshot
            current_record = {
                "ticker": ticker,
                "period": now_dt, # Snapshot time
                "timeframe": "current",
                "metrics": comprehensive_metrics, # Nested comprehensive metrics
                # Keep top-level fields for backward compatibility if needed, or rely on metrics
                "revenue": parse_finviz_val(fund_data.get('Sales')), 
                "net_income": parse_finviz_val(fund_data.get('Income')),
                "fetched_at": now_dt,
            }
            
            self.collection.update_one(
                {"ticker": ticker, "timeframe": "current"},
                {"$set": current_record},
                upsert=True,
            )

            print(f"Fundamentals: updated comprehensive records for {ticker}")

        except Exception as e:
            print(f"Error in Fundamentals Module for {ticker}: {e}")


def run_fundamentals_batch(db, tickers):
    module = FundamentalsModule(db)
    for t in tickers:
        module.fetch_fundamentals_for_symbol(t)

