import React, { useState, useEffect } from 'react';
import { getFundamentals, FundamentalsRecord } from '../api/client';

interface FinancialFundamentalsProps {
  symbol: string;
}

const FinancialFundamentals: React.FC<FinancialFundamentalsProps> = ({ symbol }) => {
  const [fundamentals, setFundamentals] = useState<FundamentalsRecord | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [activePage, setActivePage] = useState<number>(0);

  useEffect(() => {
    const fetchFundamentals = async () => {
      setLoading(true);
      try {
        const rec = await getFundamentals(symbol, 'current');
        setFundamentals(rec || null);
      } catch (err) {
        console.error('Failed to fetch fundamentals:', err);
        setFundamentals(null);
      } finally {
        setLoading(false);
      }
    };

    fetchFundamentals();
  }, [symbol]);

  const renderMetric = (label: string, value: any, format: 'currency' | 'percent' | 'number' | 'string' = 'number') => {
    let formattedValue = 'N/A';
    if (value !== undefined && value !== null) {
      if (format === 'string') {
          formattedValue = String(value);
      } else if (format === 'currency') {
        formattedValue = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0, notation: "compact", compactDisplay: "short" }).format(value);
      } else if (format === 'percent') {
        formattedValue = new Intl.NumberFormat('en-US', { style: 'percent', minimumFractionDigits: 2 }).format(value); // Percentages from Finviz are decimals (0.05 = 5%)
      } else {
        formattedValue = new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 }).format(value);
      }
    }
    return <MetricCard label={label} value={formattedValue} />;
  };

  const metrics = fundamentals?.metrics || {};

  // Pages Configuration
  const pages = [
    // --- PART 1: Raw Fundamentals (Finviz) ---
    {
      title: "Valuation Metrics",
      content: (
        <>
          <div style={sectionStyle}>
            <div style={gridStyle}>
              {renderMetric("Market Cap", metrics.market_cap, 'currency')}
              {renderMetric("Enterprise Value", metrics.enterprise_value, 'currency')}
              {renderMetric("P/E", metrics.pe_ratio)}
              {renderMetric("Forward P/E", metrics.forward_pe)}
              {renderMetric("PEG Ratio", metrics.peg_ratio)}
              {renderMetric("P/S", metrics.ps_ratio)}
              {renderMetric("P/B", metrics.pb_ratio)}
              {renderMetric("Price to Cash", metrics.price_to_cash)}
              {renderMetric("Price to FCF", metrics.price_to_fcf)}
              {renderMetric("EV / EBITDA", metrics.ev_ebitda)}
              {renderMetric("EV / Revenue", metrics.ev_revenue)}
            </div>
          </div>
        </>
      )
    },
    {
      title: "Profitability Metrics",
      content: (
        <>
          <div style={sectionStyle}>
            <div style={gridStyle}>
              {renderMetric("Gross Margin", metrics.gross_margin, 'percent')}
              {renderMetric("Operating Margin", metrics.operating_margin, 'percent')}
              {renderMetric("Profit Margin", metrics.profit_margin, 'percent')}
              {renderMetric("ROA", metrics.roa, 'percent')}
              {renderMetric("ROE", metrics.roe, 'percent')}
              {renderMetric("ROI", metrics.roi, 'percent')}
            </div>
          </div>
        </>
      )
    },
    {
      title: "Growth Metrics",
      content: (
        <>
          <div style={sectionStyle}>
            <div style={gridStyle}>
              {renderMetric("EPS Growth (5Y)", metrics.eps_growth_past_5y, 'percent')}
              {renderMetric("EPS Growth (Next 5Y)", metrics.eps_growth_next_5y, 'percent')}
              {renderMetric("Revenue Growth", metrics.revenue_growth, 'percent')} {/* Note: Might need to be added to backend if missing, assuming sales_growth_past_5y or similar */}
              {renderMetric("Qtr Rev Growth (YoY)", metrics.sales_growth_qtr_over_qtr, 'percent')}
              {renderMetric("Qtr EPS Growth (YoY)", metrics.eps_growth_qtr_over_qtr, 'percent')}
              {renderMetric("Sales Growth", metrics.sales_growth_past_5y, 'percent')}
              {renderMetric("Earnings Growth", metrics.eps_growth_this_year, 'percent')} {/* Approximation if exact field missing */}
            </div>
          </div>
        </>
      )
    },
    {
      title: "Financial Health & Liquidity",
      content: (
        <>
          <div style={sectionStyle}>
            <div style={gridStyle}>
              {renderMetric("Current Ratio", metrics.current_ratio)}
              {renderMetric("Quick Ratio", metrics.quick_ratio)}
              {renderMetric("Debt to Equity", metrics.debt_to_equity)}
              {renderMetric("LT Debt to Equity", metrics.lt_debt_to_equity)}
              {renderMetric("Total Debt", metrics.total_debt, 'currency')}
              {renderMetric("Cash", metrics.total_cash, 'currency')}
              {renderMetric("Book Value/Share", metrics.book_value_per_share, 'currency')}
            </div>
          </div>
        </>
      )
    },
    {
      title: "Cash Flow Metrics",
      content: (
        <>
          <div style={sectionStyle}>
            <div style={gridStyle}>
              {renderMetric("Operating Cash Flow", metrics.operating_cash_flow, 'currency')}
              {renderMetric("Free Cash Flow", metrics.free_cash_flow, 'currency')}
              {renderMetric("Cash per Share", metrics.cash_per_share, 'currency')}
            </div>
          </div>
        </>
      )
    },
    {
      title: "Earnings & Analyst Data",
      content: (
        <>
          <div style={sectionStyle}>
            <div style={gridStyle}>
              {renderMetric("EPS (TTM)", metrics.eps_ttm, 'currency')}
              {renderMetric("EPS (Next Q)", metrics.eps_next_q, 'currency')}
              {renderMetric("EPS (Next Y)", metrics.eps_next_y, 'currency')}
              {renderMetric("EPS Surprise", metrics.eps_surprise, 'percent')} {/* Need to check if available */}
              {renderMetric("Analyst Recom", metrics.analyst_recom)}
              {renderMetric("Target Price", metrics.target_price, 'currency')}
              {renderMetric("Earnings Date", metrics.earnings_date, 'string')}
            </div>
          </div>
        </>
      )
    },
    {
      title: "Dividends",
      content: (
        <>
          <div style={sectionStyle}>
            <div style={gridStyle}>
              {renderMetric("Dividend Yield", metrics.dividend_yield, 'percent')}
              {renderMetric("Payout Ratio", metrics.payout_ratio, 'percent')}
              {renderMetric("Dividend Growth", metrics.dividend_growth, 'percent')}
              {renderMetric("Ex-Dividend Date", metrics.ex_dividend_date, 'string')}
            </div>
          </div>
        </>
      )
    },
    {
      title: "Ownership & Share Structure",
      content: (
        <>
          <div style={sectionStyle}>
            <div style={gridStyle}>
              {renderMetric("Insider Ownership", metrics.insider_own, 'percent')}
              {renderMetric("Inst. Ownership", metrics.inst_own, 'percent')}
              {renderMetric("Insider Trans", metrics.insider_trans, 'percent')}
              {renderMetric("Inst. Trans", metrics.inst_trans, 'percent')}
              {renderMetric("Float", metrics.float_shares, 'currency')}
              {renderMetric("Shares Outstanding", metrics.shares_outstanding, 'currency')}
              {renderMetric("Short Float", metrics.short_float, 'percent')}
              {renderMetric("Short Ratio", metrics.short_ratio)}
            </div>
          </div>
        </>
      )
    },
    {
      title: "Risk & Volatility",
      content: (
        <>
          <div style={sectionStyle}>
            <div style={gridStyle}>
              {renderMetric("Beta", metrics.beta)}
              {renderMetric("Volatility (W)", metrics.volatility_week, 'percent')}
              {renderMetric("Volatility (M)", metrics.volatility_month, 'percent')}
              {renderMetric("ATR", metrics.atr)}
            </div>
          </div>
        </>
      )
    },
    {
      title: "Trading Liquidity",
      content: (
        <>
          <div style={sectionStyle}>
            <div style={gridStyle}>
              {renderMetric("Average Volume", metrics.avg_volume, 'number')}
              {renderMetric("Relative Volume", metrics.rel_volume)}
            </div>
          </div>
        </>
      )
    },
    {
      title: "Company Information",
      content: (
        <>
          <div style={sectionStyle}>
            <div style={gridStyle}>
              {renderMetric("Sector", metrics.sector, 'string')}
              {renderMetric("Industry", metrics.industry, 'string')}
              {renderMetric("Country", metrics.country, 'string')}
              {renderMetric("Exchange", metrics.exchange, 'string')}
              {renderMetric("IPO Date", metrics.ipo_date, 'string')}
              {renderMetric("Employees", metrics.employees, 'number')}
              {renderMetric("Headquarters", metrics.headquarters, 'string')}
            </div>
          </div>
        </>
      )
    },

    // --- PART 2: Advanced Financial Ratios (Calculated) ---
    {
      title: "Valuation & Yield Ratios",
      content: (
        <>
          <div style={sectionStyle}>
            <div style={gridStyle}>
              {renderMetric("Earnings Yield", metrics.earnings_yield, 'percent')}
              {renderMetric("Fwd Earn Yield", metrics.forward_earnings_yield, 'percent')}
              {renderMetric("FCF Yield", metrics.fcf_yield, 'percent')}
              {renderMetric("OCF Yield", metrics.ocf_yield, 'percent')}
              {renderMetric("EBITDA Yield", metrics.ebitda_yield, 'percent')}
              {renderMetric("Revenue Yield", metrics.revenue_yield, 'percent')}
              {renderMetric("Book-to-Market", metrics.book_to_market)}
              {renderMetric("PEG Adj Yield", metrics.price_to_growth_adj_yield, 'percent')}
            </div>
          </div>
        </>
      )
    },
    {
      title: "Profitability & Efficiency",
      content: (
        <>
          <div style={sectionStyle}>
            <div style={gridStyle}>
              {renderMetric("Asset Turnover", metrics.asset_turnover)}
              {renderMetric("Oper. Efficiency", metrics.operating_efficiency, 'percent')}
              {renderMetric("ROIC", metrics.roic, 'percent')}
              {renderMetric("CROIC", metrics.croic, 'percent')}
              {renderMetric("Gr. Prof Efficiency", metrics.gross_profit_efficiency, 'percent')}
            </div>
          </div>
        </>
      )
    },
    {
      title: "Growth & Quality Ratios",
      content: (
        <>
          <div style={sectionStyle}>
            <div style={gridStyle}>
              {renderMetric("SGR", metrics.sgr, 'percent')}
              {renderMetric("Earn Growth Eff", metrics.earnings_growth_efficiency)}
              {renderMetric("Rev/Earn Growth", metrics.revenue_to_earnings_growth)}
              {renderMetric("Cash Conv Ratio", metrics.cash_conversion_ratio)}
              {renderMetric("FCF Conversion", metrics.fcf_conversion)}
            </div>
          </div>
        </>
      )
    },
    {
      title: "Leverage & Risk Ratios",
      content: (
        <>
          <div style={sectionStyle}>
            <div style={gridStyle}>
              {renderMetric("Net Debt/EBITDA", metrics.net_debt_to_ebitda)}
              {renderMetric("Debt Service Ratio", metrics.debt_service_ratio)}
              {renderMetric("Fin. Leverage", metrics.financial_leverage_ratio)}
              {renderMetric("Lev Adj Volatility", metrics.leverage_adjusted_volatility)}
              {renderMetric("Liquidity Cushion", metrics.liquidity_cushion)}
            </div>
          </div>
        </>
      )
    },
    {
      title: "Shareholder Return",
      content: (
        <>
          <div style={sectionStyle}>
            <div style={gridStyle}>
              {renderMetric("Shareholder Yield", metrics.shareholder_yield, 'percent')}
              {renderMetric("Retention Ratio", metrics.retention_ratio, 'percent')}
              {renderMetric("Reinvestment Rate", metrics.reinvestment_rate, 'percent')}
              {renderMetric("Capital Efficiency", metrics.capital_efficiency)}
            </div>
          </div>
        </>
      )
    },
    {
      title: "Ownership Ratios",
      content: (
        <>
          <div style={sectionStyle}>
            <div style={gridStyle}>
              {renderMetric("Insider Buy Int", metrics.insider_buying_intensity, 'percent')}
              {renderMetric("Inst. Accum", metrics.institutional_accumulation, 'percent')}
              {renderMetric("Float Turnover", metrics.float_turnover)}
            </div>
          </div>
        </>
      )
    },
    {
      title: "Liquidity & Stability",
      content: (
        <>
          <div style={sectionStyle}>
            <div style={gridStyle}>
              {renderMetric("Vol/Liq Ratio", metrics.volatility_liquidity_ratio)}
              {renderMetric("Turnover Stability", metrics.turnover_stability)}
            </div>
          </div>
        </>
      )
    },
    {
      title: "Composite Scores",
      content: (
        <>
          <div style={sectionStyle}>
            <div style={gridStyle}>
              {renderMetric("Value Score", metrics.value_score)}
              {renderMetric("Quality Score", metrics.quality_score)}
              {renderMetric("Growth Score", metrics.growth_score)}
              {renderMetric("Low Risk Score", metrics.low_risk_score)}
            </div>
          </div>
        </>
      )
    },
    {
      title: "Risk-Adjusted Metrics",
      content: (
        <>
          <div style={sectionStyle}>
            <div style={gridStyle}>
              {renderMetric("Risk Adj Return", metrics.risk_adjusted_return)}
              {renderMetric("Fund Risk Score", metrics.fundamental_risk_score)}
            </div>
          </div>
        </>
      )
    }
  ];

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', color: '#fff' }}>
      <div
        style={{
          marginBottom: '16px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '10px',
        }}
      >
        <h3 style={{ margin: 0 }}>Financial Fundamentals</h3>
      </div>

      <div style={{ flex: 1, overflowY: 'auto', paddingRight: '4px', display: 'flex', flexDirection: 'column' }}>
        {loading ? (
          <div style={centerStyle}>Loading fundamentals...</div>
        ) : !fundamentals ? (
          <div style={centerStyle}>No fundamentals found for {symbol}</div>
        ) : (
          <>
            {/* Header Info */}
            <div style={{
                background: 'rgba(255,255,255,0.03)',
                padding: '16px',
                borderRadius: '12px',
                border: '1px solid rgba(255,255,255,0.05)',
                marginBottom: '20px'
              }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <h1 style={{ margin: 0, fontSize: '1.8rem' }}>{fundamentals.ticker}</h1>
                  <div style={{ fontSize: '0.9rem', color: '#a1a1aa' }}>
                    Period: {new Date(fundamentals.period).toLocaleDateString()}
                  </div>
                </div>
                {/* Page Navigation */}
                <div style={{ display: 'flex', gap: '8px' }}>
                  {pages.map((_, index) => (
                    <button
                      key={index}
                      onClick={() => setActivePage(index)}
                      style={{
                        width: '10px',
                        height: '10px',
                        borderRadius: '50%',
                        background: activePage === index ? '#8b5cf6' : 'rgba(255,255,255,0.2)',
                        border: 'none',
                        cursor: 'pointer',
                        padding: 0
                      }}
                      title={`Page ${index + 1}`}
                    />
                  ))}
                </div>
              </div>
            </div>

            {/* Page Content */}
            <div style={{ flex: 1, overflowY: 'auto' }}>
                <h3 style={{ marginTop: 0, marginBottom: '16px', fontSize: '1.1rem', color: '#e5e7eb' }}>
                    {pages[activePage].title}
                </h3>
                {pages[activePage].content}
            </div>
          </>
        )}
      </div>
    </div>
  );
};

const MetricCard: React.FC<{ label: string; value: string }> = ({ label, value }) => (
    <div style={{ background: 'rgba(255,255,255,0.03)', borderRadius: '6px', padding: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
        <div style={{ color: '#a1a1aa', fontSize: '0.7rem', marginBottom: '2px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{label}</div>
        <div style={{ fontSize: '0.9rem', fontWeight: 'bold', color: '#fff', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{value}</div>
    </div>
);

// Styles
const sectionStyle: React.CSSProperties = { marginBottom: '16px' };
const sectionHeaderStyle: React.CSSProperties = { margin: '0 0 8px 0', color: '#a1a1aa', fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '0.5px' };
const gridStyle: React.CSSProperties = { display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(130px, 1fr))', gap: '8px' };
const centerStyle: React.CSSProperties = { height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#a1a1aa' };

export default FinancialFundamentals;
