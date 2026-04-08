import React, { useState, useEffect, useRef } from 'react';
import { getFundamentals, FundamentalsRecord } from '../api/client';

interface FinancialFundamentalsProps {
  symbol: string;
}

const FinancialFundamentals: React.FC<FinancialFundamentalsProps> = ({ symbol }) => {
  const [fundamentals, setFundamentals] = useState<FundamentalsRecord | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  type Mode = 'annual' | 'quarterly';
  type Selection = { mode: Mode; year: number | null; quarter?: 1 | 2 | 3 | 4 };
  const [incomeSel, setIncomeSel] = useState<Selection | null>(null);
  const [balanceSel, setBalanceSel] = useState<Selection | null>(null);
  const [cashSel, setCashSel] = useState<Selection | null>(null);
  const [incomeExpanded, setIncomeExpanded] = useState(false);
  const [balanceExpanded, setBalanceExpanded] = useState(false);
  const [cashExpanded, setCashExpanded] = useState(false);
  const [incomeDraft, setIncomeDraft] = useState<Selection | null>(null);
  const [balanceDraft, setBalanceDraft] = useState<Selection | null>(null);
  const [cashDraft, setCashDraft] = useState<Selection | null>(null);
  const earningsTimeoutsRef = useRef<number[]>([]);
  const earningsImmediateKeyRef = useRef<string | null>(null);

  useEffect(() => {
    const fetchFundamentals = async () => {
      setLoading(true);
      try {
        const rec = await getFundamentals(symbol, 'current');
        setFundamentals(rec || null);
        const earningsStr = rec?.metrics?.earnings_date ? String(rec.metrics.earnings_date) : null;
        earningsTimeoutsRef.current.forEach(id => window.clearTimeout(id));
        earningsTimeoutsRef.current = [];
        if (earningsStr) {
          const monthMap: Record<string, number> = {
            Jan: 0,
            Feb: 1,
            Mar: 2,
            Apr: 3,
            May: 4,
            Jun: 5,
            Jul: 6,
            Aug: 7,
            Sep: 8,
            Oct: 9,
            Nov: 10,
            Dec: 11
          };

          const parts = earningsStr.trim().split(/\s+/);
          const month = monthMap[parts[0]];
          const day = parts[1] ? parseInt(parts[1], 10) : Number.NaN;
          const session = parts[2] ? parts[2].toUpperCase() : '';

          const now = new Date();
          const baseHour = session === 'BMO' ? 9 : session === 'AMC' ? 17 : 12;
          const baseMinute = session === 'BMO' ? 45 : session === 'AMC' ? 15 : 0;

          if (Number.isInteger(month) && Number.isFinite(day)) {
            const candidateYear = now.getFullYear();
            let runAt = new Date(candidateYear, month, day, baseHour, baseMinute, 0, 0);
            if (runAt.getTime() < now.getTime() - 12 * 60 * 60 * 1000) {
              runAt = new Date(candidateYear + 1, month, day, baseHour, baseMinute, 0, 0);
            }

            if (runAt.getFullYear() === now.getFullYear() && runAt.getMonth() === now.getMonth() && runAt.getDate() === now.getDate()) {
              const key = `${symbol}-${runAt.getFullYear()}-${runAt.getMonth()}-${runAt.getDate()}`;
              if (earningsImmediateKeyRef.current !== key) {
                earningsImmediateKeyRef.current = key;
                const id = window.setTimeout(() => {
                  fetchFundamentals();
                }, 0);
                earningsTimeoutsRef.current.push(id);
              }
            }

            const scheduleFetchAt = (d: Date) => {
              const ms = d.getTime() - Date.now();
              if (ms <= 0) return;
              const id = window.setTimeout(() => {
                fetchFundamentals();
              }, ms);
              earningsTimeoutsRef.current.push(id);
            };

            scheduleFetchAt(runAt);
            const nextDay = new Date(runAt.getFullYear(), runAt.getMonth(), runAt.getDate() + 1, 9, 45, 0, 0);
            scheduleFetchAt(nextDay);
          }
        }
      } catch (err) {
        console.error('Failed to fetch fundamentals:', err);
        setFundamentals(null);
      } finally {
        setLoading(false);
      }
    };

    fetchFundamentals();
    const interval = window.setInterval(fetchFundamentals, 24 * 60 * 60 * 1000);
    return () => {
      window.clearInterval(interval);
      earningsTimeoutsRef.current.forEach(id => window.clearTimeout(id));
      earningsTimeoutsRef.current = [];
      earningsImmediateKeyRef.current = null;
    };
  }, [symbol]);

  const parseYear = (dateStr: string | undefined) => {
    if (!dateStr) return null;
    const d = new Date(dateStr);
    return Number.isFinite(d.getTime()) ? d.getFullYear() : null;
  };

  const latestAnnualYear = (annual: any[] | undefined) => {
    if (!annual || annual.length === 0) return null;
    let best: any | null = null;
    let bestTime = -Infinity;
    for (const p of annual) {
      const t = p?.date ? new Date(p.date).getTime() : NaN;
      if (Number.isFinite(t) && t > bestTime) {
        best = p;
        bestTime = t;
      }
    }
    return best ? parseYear(best.date) : null;
  };

  const latestAnnualEndMonth = (annual: any[] | undefined) => {
    if (!annual || annual.length === 0) return null;
    let best: any | null = null;
    let bestTime = -Infinity;
    for (const p of annual) {
      const t = p?.date ? new Date(p.date).getTime() : NaN;
      if (Number.isFinite(t) && t > bestTime) {
        best = p;
        bestTime = t;
      }
    }
    if (!best?.date) return null;
    const d = new Date(best.date);
    if (!Number.isFinite(d.getTime())) return null;
    return d.getMonth() + 1;
  };

  const fiscalYearOfDate = (dateStr: string | undefined, fiscalYearEndMonth: number) => {
    if (!dateStr) return null;
    const d = new Date(dateStr);
    if (!Number.isFinite(d.getTime())) return null;
    const month = d.getMonth() + 1;
    const year = d.getFullYear();
    return month > fiscalYearEndMonth ? year + 1 : year;
  };

  const fiscalQuarterOfDate = (dateStr: string | undefined, fiscalYearEndMonth: number) => {
    if (!dateStr) return null;
    const d = new Date(dateStr);
    if (!Number.isFinite(d.getTime())) return null;
    const month = d.getMonth() + 1;
    const fiscalYearStartMonth = (fiscalYearEndMonth % 12) + 1;
    const offset = (month - fiscalYearStartMonth + 12) % 12;
    return (Math.floor(offset / 3) + 1) as 1 | 2 | 3 | 4;
  };

  const currentFiscalYear = (fiscalYearEndMonth: number) => {
    const d = new Date();
    const month = d.getMonth() + 1;
    const year = d.getFullYear();
    return month > fiscalYearEndMonth ? year + 1 : year;
  };

  const currentFiscalQuarter = (fiscalYearEndMonth: number) => {
    const d = new Date();
    const month = d.getMonth() + 1;
    const fiscalYearStartMonth = (fiscalYearEndMonth % 12) + 1;
    const offset = (month - fiscalYearStartMonth + 12) % 12;
    return (Math.floor(offset / 3) + 1) as 1 | 2 | 3 | 4;
  };

  const defaultQuarterSelection = (annual: any[] | undefined, quarterly: any[] | undefined): Selection => {
    const fyEnd = latestAnnualEndMonth(annual) ?? 12;
    const curFY = currentFiscalYear(fyEnd);
    const curFQ = currentFiscalQuarter(fyEnd);
    const prevYear = curFQ > 1 ? curFY : curFY - 1;
    const prevQuarter = (curFQ > 1 ? (curFQ - 1) : 4) as 1 | 2 | 3 | 4;

    if (quarterly && quarterly.length > 0) {
      const hasPrev = quarterly.some(p => fiscalYearOfDate(p?.date, fyEnd) === prevYear && fiscalQuarterOfDate(p?.date, fyEnd) === prevQuarter);
      if (hasPrev) return { mode: 'quarterly', year: prevYear, quarter: prevQuarter };

      let bestYear: number | null = null;
      let bestQuarter: 1 | 2 | 3 | 4 | null = null;
      let bestIdx = -Infinity;
      for (const p of quarterly) {
        const y = fiscalYearOfDate(p?.date, fyEnd);
        const q = fiscalQuarterOfDate(p?.date, fyEnd);
        if (!y || !q) continue;
        const idx = y * 4 + q;
        if (idx > bestIdx) {
          bestIdx = idx;
          bestYear = y;
          bestQuarter = q;
        }
      }
      if (bestYear && bestQuarter) return { mode: 'quarterly', year: bestYear, quarter: bestQuarter };
    }

    return { mode: 'annual', year: latestAnnualYear(annual) };
  };

  const collectYears = (annual: any[] | undefined, quarterly: any[] | undefined, fiscalYearEndMonth: number) => {
    const setYears = new Set<number>();
    (annual || []).forEach((p: any) => {
      const y = parseYear(p?.date);
      if (y) setYears.add(y);
    });
    (quarterly || []).forEach((p: any) => {
      const y = fiscalYearOfDate(p?.date, fiscalYearEndMonth);
      if (y) setYears.add(y);
    });
    setYears.add(currentFiscalYear(fiscalYearEndMonth));
    return Array.from(setYears).sort((a, b) => b - a);
  };

  const availableQuarters = (quarterly: any[] | undefined, year: number | null, fiscalYearEndMonth: number) => {
    const quarters: Array<{ label: string; value: 1 | 2 | 3 | 4; disabled?: boolean }> = [
      { label: 'Q1', value: 1 },
      { label: 'Q2', value: 2 },
      { label: 'Q3', value: 3 },
      { label: 'Q4', value: 4 }
    ];
    if (!year) return quarters;

    const avail = new Set<number>();
    (quarterly || []).forEach((p: any) => {
      const y = fiscalYearOfDate(p?.date, fiscalYearEndMonth);
      const q = fiscalQuarterOfDate(p?.date, fiscalYearEndMonth);
      if (y && q && y === year) avail.add(q);
    });
    if (avail.size === 0) return quarters;
    return quarters.map(q => ({ ...q, disabled: !avail.has(q.value) }));
  };

  useEffect(() => {
    if (!fundamentals) return;
    if (!incomeSel) {
      setIncomeSel(defaultQuarterSelection(fundamentals.financials_annual, fundamentals.financials_quarterly));
    }
    if (!balanceSel) {
      setBalanceSel(defaultQuarterSelection(fundamentals.balance_sheet_annual, fundamentals.balance_sheet_quarterly));
    }
    if (!cashSel) {
      setCashSel(defaultQuarterSelection(fundamentals.cashflow_annual, fundamentals.cashflow_quarterly));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fundamentals]);

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
  const incomeFiscalYearEndMonth = latestAnnualEndMonth(fundamentals?.financials_annual) ?? 12;
  const balanceFiscalYearEndMonth = latestAnnualEndMonth(fundamentals?.balance_sheet_annual) ?? 12;
  const cashFiscalYearEndMonth = latestAnnualEndMonth(fundamentals?.cashflow_annual) ?? 12;
  const incomeCurrentFY = currentFiscalYear(incomeFiscalYearEndMonth);
  const balanceCurrentFY = currentFiscalYear(balanceFiscalYearEndMonth);
  const cashCurrentFY = currentFiscalYear(cashFiscalYearEndMonth);

  // Helper to infer formatting for dynamic keys
  const inferFormat = (key: string, value: any): 'currency' | 'percent' | 'number' | 'string' => {
    const k = key.toLowerCase();
    if (typeof value === 'string') return 'string';
    if (k.includes('margin') || k.includes('rate') || k.includes('yield') || k.includes('percent') || k.includes('growth') || k.includes('ratio')) return 'percent';
    if (k.includes('revenue') || k.includes('income') || k.includes('profit') || k.includes('cash') || k.includes('debt') || k.includes('assets') || k.includes('liabilities') || k.includes('equity') || k.includes('paid') || k.includes('received') || k.includes('expenditure') || k.includes('repurchase') || k.includes('value')) return 'currency';
    return 'number';
  };

  const renderFinancialsAsGrid = (
    annualData: any[] | undefined,
    quarterlyData: any[] | undefined,
    title: string,
    selected?: Selection
  ) => {
    if ((!annualData || annualData.length === 0) && (!quarterlyData || quarterlyData.length === 0)) return <div style={centerStyle}>No {title} data available</div>;

    const fiscalYearEndMonth = latestAnnualEndMonth(annualData) ?? 12;
    const earningsDate = metrics?.earnings_date ? String(metrics.earnings_date) : null;

    const formatQuarter = (dateStr: string) => {
      const y = fiscalYearOfDate(dateStr, fiscalYearEndMonth);
      const q = fiscalQuarterOfDate(dateStr, fiscalYearEndMonth);
      return y && q ? `Q${q} ${y}` : dateStr;
    };

    const pickLatestPeriod = (data: any[] | undefined) => {
      if (!data || data.length === 0) return null;

      let best = data[0];
      let bestTime = Number.NaN;

      for (const candidate of data) {
        if (!candidate) continue;
        const candidateTime = candidate?.date ? new Date(candidate.date).getTime() : Number.NaN;

        if (!Number.isFinite(bestTime)) {
          best = candidate;
          bestTime = candidateTime;
          continue;
        }

        if (!Number.isFinite(candidateTime)) continue;
        if (candidateTime > bestTime) {
          best = candidate;
          bestTime = candidateTime;
        }
      }

      return best;
    };

    const renderSinglePeriod = (period: any, isQuarterly: boolean) => {
      if (!period) return null;

      const rawKeys = Object.keys(period).filter(k => k !== 'date');
      const keys = rawKeys.slice(0, 36);
      const displayDate = isQuarterly
        ? formatQuarter(period.date)
        : typeof period.date === 'string'
          ? period.date.substring(0, 4)
          : 'N/A';

      return (
        <div>
          <h4 style={{ ...sectionHeaderStyle, color: '#fff' }}>{displayDate}</h4>
          <div style={sectionStyle}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(110px, 1fr))', gap: '8px' }}>
              {keys.map(k => (
                <React.Fragment key={k}>
                  {renderMetric(k.replace(/([A-Z])/g, ' $1').trim(), period[k], inferFormat(k, period[k]))}
                </React.Fragment>
              ))}
            </div>
          </div>
        </div>
      );
    };

    const latestAnnual = pickLatestPeriod(annualData);
    const latestQuarterly = pickLatestPeriod(quarterlyData);

    const fiscalIndexFromDate = (dateStr: string | undefined) => {
      const y = fiscalYearOfDate(dateStr, fiscalYearEndMonth);
      const q = fiscalQuarterOfDate(dateStr, fiscalYearEndMonth);
      return y && q ? y * 4 + q : null;
    };

    let periodToShow: any = null;
    let isQuarter = false;
    if (selected?.mode === 'quarterly') {
      const yr = selected.year;
      const q = selected.quarter;
      if (yr && q && quarterlyData && quarterlyData.length > 0) {
        const match = quarterlyData.find(p => fiscalYearOfDate(p?.date, fiscalYearEndMonth) === yr && fiscalQuarterOfDate(p?.date, fiscalYearEndMonth) === q);
        if (!match) {
          const latestIdx = latestQuarterly?.date ? fiscalIndexFromDate(latestQuarterly.date) : null;
          const selectedIdx = yr * 4 + q;
          const extra = earningsDate && latestIdx && selectedIdx > latestIdx ? ` (Next earnings: ${earningsDate})` : '';
          return <div style={centerStyle}>No {title} data for Q{q} {yr}{extra}</div>;
        }
        periodToShow = match;
        isQuarter = true;
      } else {
        periodToShow = latestQuarterly;
        isQuarter = true;
      }
    } else {
      const yr = selected?.year;
      if (yr && annualData && annualData.length > 0) {
        const match = annualData.find(p => parseYear(p.date) === yr);
        if (!match) {
          const latestY = latestAnnual?.date ? parseYear(latestAnnual.date) : null;
          const extra = earningsDate && latestY && yr > latestY ? ` (Next earnings: ${earningsDate})` : '';
          return <div style={centerStyle}>No {title} data for {yr}{extra}</div>;
        }
        periodToShow = match;
      } else {
        periodToShow = latestAnnual;
      }
      isQuarter = false;
    }

    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
        {renderSinglePeriod(periodToShow, isQuarter)}
      </div>
    );
  };

  const renderHoldersAsGrid = (data: any[] | undefined, title: string) => {
    if (!data || data.length === 0) return <div style={centerStyle}>No {title} data available</div>;

    // Show top 12 holders
    const holders = data.slice(0, 12);

    return (
      <div style={sectionStyle}>
        <div style={gridStyle}>
          {holders.map((holder, index) => {
            // Try to find the 'Value' or '% Out' field to display as value
            // Common keys: 'Holder', 'Shares', 'Date Reported', '% Out', 'Value'
            let val = holder['Value'] || holder['% Out'] || holder['Shares'];
            let fmt: 'currency' | 'percent' | 'number' = 'number';
            
            if (holder['Value']) fmt = 'currency';
            else if (holder['% Out']) fmt = 'percent';

            return (
              <React.Fragment key={index}>
                {renderMetric(holder['Holder'] || `Holder ${index+1}`, val, fmt)}
              </React.Fragment>
            );
          })}
        </div>
      </div>
    );
  };

  // Pages Configuration
  // Grouping categories to fit 2-3 per page
  const pages = [
    // --- Page 1: Valuation & Profitability ---
    {
      title: "Valuation & Profitability",
      content: (
        <>
          <h4 style={sectionHeaderStyle}>Valuation Metrics</h4>
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

          <h4 style={sectionHeaderStyle}>Profitability Metrics</h4>
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

    // --- Page 2: Growth & Financial Health ---
    {
      title: "Growth & Health",
      content: (
        <>
          <h4 style={sectionHeaderStyle}>Growth Metrics</h4>
          <div style={sectionStyle}>
            <div style={gridStyle}>
              {renderMetric("EPS Growth (5Y)", metrics.eps_growth_past_5y, 'percent')}
              {renderMetric("EPS Growth (Next 5Y)", metrics.eps_growth_next_5y, 'percent')}
              {renderMetric("Revenue Growth", metrics.revenue_growth, 'percent')}
              {renderMetric("Qtr Rev Growth (YoY)", metrics.sales_growth_qtr_over_qtr, 'percent')}
              {renderMetric("Qtr EPS Growth (YoY)", metrics.eps_growth_qtr_over_qtr, 'percent')}
              {renderMetric("Sales Growth", metrics.sales_growth_past_5y, 'percent')}
              {renderMetric("Earnings Growth", metrics.eps_growth_this_year, 'percent')}
            </div>
          </div>

          <h4 style={sectionHeaderStyle}>Financial Health & Liquidity</h4>
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

          <h4 style={sectionHeaderStyle}>Cash Flow Metrics</h4>
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

    // --- Page 3: Earnings, Dividends & Ownership ---
    {
      title: "Earnings, Divs & Ownership",
      content: (
        <>
          <h4 style={sectionHeaderStyle}>Earnings & Analyst Data</h4>
          <div style={sectionStyle}>
            <div style={gridStyle}>
              {renderMetric("EPS (TTM)", metrics.eps_ttm, 'currency')}
              {renderMetric("EPS (Next Q)", metrics.eps_next_q, 'currency')}
              {renderMetric("EPS (Next Y)", metrics.eps_next_y, 'currency')}
              {renderMetric("EPS Surprise", metrics.eps_surprise, 'percent')}
              {renderMetric("Analyst Recom", metrics.analyst_recom)}
              {renderMetric("Target Price", metrics.target_price, 'currency')}
              {renderMetric("Earnings Date", metrics.earnings_date, 'string')}
            </div>
          </div>

          <h4 style={sectionHeaderStyle}>Dividends</h4>
          <div style={sectionStyle}>
            <div style={gridStyle}>
              {renderMetric("Dividend Yield", metrics.dividend_yield, 'percent')}
              {renderMetric("Payout Ratio", metrics.payout_ratio, 'percent')}
              {renderMetric("Dividend Growth", metrics.dividend_growth, 'percent')}
              {renderMetric("Ex-Dividend Date", metrics.ex_dividend_date, 'string')}
            </div>
          </div>

          <h4 style={sectionHeaderStyle}>Ownership & Share Structure</h4>
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

    // --- Page 4: Risk, Liquidity & Info ---
    {
      title: "Risk, Liquidity & Info",
      content: (
        <>
          <h4 style={sectionHeaderStyle}>Risk & Volatility</h4>
          <div style={sectionStyle}>
            <div style={gridStyle}>
              {renderMetric("Beta", metrics.beta)}
              {renderMetric("Volatility (W)", metrics.volatility_week, 'percent')}
              {renderMetric("Volatility (M)", metrics.volatility_month, 'percent')}
              {renderMetric("ATR", metrics.atr)}
            </div>
          </div>

          <h4 style={sectionHeaderStyle}>Trading Liquidity</h4>
          <div style={sectionStyle}>
            <div style={gridStyle}>
              {renderMetric("Average Volume", metrics.avg_volume, 'number')}
              {renderMetric("Relative Volume", metrics.rel_volume)}
            </div>
          </div>

          <h4 style={sectionHeaderStyle}>Company Information</h4>
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

    // --- Page 5: Advanced Ratios (Valuation, Profitability, Growth) ---
    {
      title: "Advanced Ratios I",
      content: (
        <>
          <h4 style={sectionHeaderStyle}>Valuation & Yield Ratios</h4>
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

          <h4 style={sectionHeaderStyle}>Profitability & Efficiency</h4>
          <div style={sectionStyle}>
            <div style={gridStyle}>
              {renderMetric("Asset Turnover", metrics.asset_turnover)}
              {renderMetric("Oper. Efficiency", metrics.operating_efficiency, 'percent')}
              {renderMetric("ROIC", metrics.roic, 'percent')}
              {renderMetric("CROIC", metrics.croic, 'percent')}
              {renderMetric("Gr. Prof Efficiency", metrics.gross_profit_efficiency, 'percent')}
            </div>
          </div>

          <h4 style={sectionHeaderStyle}>Growth & Quality Ratios</h4>
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

    // --- Page 6: Advanced Ratios (Leverage, Shareholder, Ownership) ---
    {
      title: "Advanced Ratios II",
      content: (
        <>
          <h4 style={sectionHeaderStyle}>Leverage & Risk Ratios</h4>
          <div style={sectionStyle}>
            <div style={gridStyle}>
              {renderMetric("Net Debt/EBITDA", metrics.net_debt_to_ebitda)}
              {renderMetric("Debt Service Ratio", metrics.debt_service_ratio)}
              {renderMetric("Fin. Leverage", metrics.financial_leverage_ratio)}
              {renderMetric("Lev Adj Volatility", metrics.leverage_adjusted_volatility)}
              {renderMetric("Liquidity Cushion", metrics.liquidity_cushion)}
            </div>
          </div>

          <h4 style={sectionHeaderStyle}>Shareholder Return</h4>
          <div style={sectionStyle}>
            <div style={gridStyle}>
              {renderMetric("Shareholder Yield", metrics.shareholder_yield, 'percent')}
              {renderMetric("Retention Ratio", metrics.retention_ratio, 'percent')}
              {renderMetric("Reinvestment Rate", metrics.reinvestment_rate, 'percent')}
              {renderMetric("Capital Efficiency", metrics.capital_efficiency)}
            </div>
          </div>

          <h4 style={sectionHeaderStyle}>Ownership Ratios</h4>
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

    // --- Page 7: Scores & Liquidity ---
    {
      title: "Scores & Stability",
      content: (
        <>
           <h4 style={sectionHeaderStyle}>Liquidity & Stability</h4>
          <div style={sectionStyle}>
            <div style={gridStyle}>
              {renderMetric("Vol/Liq Ratio", metrics.volatility_liquidity_ratio)}
              {renderMetric("Turnover Stability", metrics.turnover_stability)}
            </div>
          </div>

          <h4 style={sectionHeaderStyle}>Composite Scores</h4>
          <div style={sectionStyle}>
            <div style={gridStyle}>
              {renderMetric("Value Score", metrics.value_score)}
              {renderMetric("Quality Score", metrics.quality_score)}
              {renderMetric("Growth Score", metrics.growth_score)}
              {renderMetric("Low Risk Score", metrics.low_risk_score)}
            </div>
          </div>

          <h4 style={sectionHeaderStyle}>Risk-Adjusted Metrics</h4>
          <div style={sectionStyle}>
            <div style={gridStyle}>
              {renderMetric("Risk Adj Return", metrics.risk_adjusted_return)}
              {renderMetric("Fund Risk Score", metrics.fundamental_risk_score)}
            </div>
          </div>
        </>
      )
    },
    // --- PART 3: Core Fundamentals (YFinance Tables) ---
    {
      title: "Income Statement",
      content: (
        <>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
            <div style={{ fontSize: '0.85rem', color: '#a1a1aa' }}>
              Showing: {incomeSel?.mode === 'quarterly' ? `Q${incomeSel?.quarter} ${incomeSel?.year ?? ''}` : `Annual ${incomeSel?.year ?? ''}`}
            </div>
            <button
              onClick={() => {
                const defYear = incomeSel?.year ?? latestAnnualYear(fundamentals?.financials_annual) ?? null;
                setIncomeDraft(incomeSel ?? { mode: 'annual', year: defYear, quarter: 1 });
                setIncomeExpanded(v => !v);
              }}
              style={{ background: '#111827', color: '#e5e7eb', border: '1px solid #374151', borderRadius: 6, padding: '6px 10px', cursor: 'pointer' }}
            >
              {incomeExpanded ? 'Collapse' : 'Expand'}
            </button>
          </div>
          {incomeExpanded && (
            <div style={{ marginBottom: '12px', padding: '12px', borderRadius: 10, border: '1px solid rgba(255,255,255,0.08)', background: 'rgba(255,255,255,0.03)' }}>
              <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
                <div style={{ flex: '1 1 120px' }}>
                  <div style={{ marginBottom: 6, color: '#a1a1aa', fontSize: '0.8rem' }}>Mode</div>
                  <div style={{ display: 'flex', gap: 8 }}>
                    <button
                      onClick={() => setIncomeDraft(prev => ({ ...(prev ?? { mode: 'annual', year: latestAnnualYear(fundamentals?.financials_annual), quarter: 1 }), mode: 'annual' }))}
                      style={{
                        padding: '6px 10px',
                        borderRadius: 6,
                        border: '1px solid #374151',
                        background: (incomeDraft?.mode ?? 'annual') === 'annual' ? '#111827' : 'transparent',
                        color: '#e5e7eb',
                        cursor: 'pointer'
                      }}
                    >
                      Annual
                    </button>
                    <button
                      onClick={() => setIncomeDraft(prev => ({ ...(prev ?? { mode: 'quarterly', year: latestAnnualYear(fundamentals?.financials_annual), quarter: 1 }), mode: 'quarterly' }))}
                      style={{
                        padding: '6px 10px',
                        borderRadius: 6,
                        border: '1px solid #374151',
                        background: (incomeDraft?.mode ?? 'annual') === 'quarterly' ? '#111827' : 'transparent',
                        color: '#e5e7eb',
                        cursor: 'pointer'
                      }}
                    >
                      Quarterly
                    </button>
                  </div>
                </div>
                <div style={{ flex: '1 1 180px' }}>
                  <div style={{ marginBottom: 6, color: '#a1a1aa', fontSize: '0.8rem' }}>Year</div>
                  <select
                    value={incomeDraft?.year ?? ''}
                    onChange={e => {
                      const nextYear = e.target.value ? parseInt(e.target.value, 10) : null;
                      setIncomeDraft(prev => {
                        const base = prev ?? { mode: 'annual', year: null, quarter: 1 };
                        let nextQuarter = base.quarter ?? 1;
                        if (base.mode === 'quarterly') {
                          const opts = availableQuarters(fundamentals?.financials_quarterly, nextYear, incomeFiscalYearEndMonth);
                          const selectedOpt = opts.find(o => o.value === nextQuarter);
                          if (selectedOpt?.disabled) {
                            nextQuarter = opts.find(o => !o.disabled)?.value ?? 1;
                          }
                        }
                        return { ...base, year: nextYear, quarter: nextQuarter };
                      });
                    }}
                    style={{ width: '100%', padding: 8, background: '#0b0f17', color: '#e5e7eb', border: '1px solid #374151', borderRadius: 6 }}
                  >
                    <option value="">Select year</option>
                    {collectYears(fundamentals?.financials_annual, fundamentals?.financials_quarterly, incomeFiscalYearEndMonth).map(y => (
                      <option key={y} value={y}>
                        {y === incomeCurrentFY ? `Current (${y})` : y}
                      </option>
                    ))}
                  </select>
                </div>
                <div style={{ flex: '1 1 160px', opacity: (incomeDraft?.mode ?? 'annual') === 'quarterly' ? 1 : 0.5 }}>
                  <div style={{ marginBottom: 6, color: '#a1a1aa', fontSize: '0.8rem' }}>Quarter</div>
                  <select
                    disabled={(incomeDraft?.mode ?? 'annual') !== 'quarterly'}
                    value={incomeDraft?.quarter ?? 1}
                    onChange={e => setIncomeDraft(prev => ({ ...(prev ?? { mode: 'quarterly', year: latestAnnualYear(fundamentals?.financials_annual), quarter: 1 }), quarter: parseInt(e.target.value, 10) as 1 | 2 | 3 | 4 }))}
                    style={{ width: '100%', padding: 8, background: '#0b0f17', color: '#e5e7eb', border: '1px solid #374151', borderRadius: 6 }}
                  >
                    {availableQuarters(fundamentals?.financials_quarterly, incomeDraft?.year ?? null, incomeFiscalYearEndMonth).map(q => (
                      <option key={q.value} value={q.value} disabled={q.disabled}>
                        {q.label}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 12 }}>
                <button
                  onClick={() => {
                    setIncomeDraft(null);
                    setIncomeExpanded(false);
                  }}
                  style={{ background: 'transparent', color: '#e5e7eb', border: '1px solid #374151', borderRadius: 6, padding: '8px 12px', cursor: 'pointer' }}
                >
                  Cancel
                </button>
                <button
                  onClick={() => {
                    if (incomeDraft) setIncomeSel(incomeDraft);
                    setIncomeExpanded(false);
                  }}
                  style={{ background: '#2563eb', color: '#fff', border: '1px solid #1d4ed8', borderRadius: 6, padding: '8px 12px', cursor: 'pointer' }}
                >
                  Apply
                </button>
              </div>
            </div>
          )}
          <div style={sectionStyle}>
            {renderFinancialsAsGrid(
              fundamentals?.financials_annual,
              fundamentals?.financials_quarterly,
              "Income Statement",
              incomeExpanded
                ? (incomeDraft ?? incomeSel ?? { mode: 'annual', year: latestAnnualYear(fundamentals?.financials_annual) })
                : (incomeSel ?? { mode: 'annual', year: latestAnnualYear(fundamentals?.financials_annual) })
            )}
          </div>
        </>
      )
    },
    {
      title: "Balance Sheet",
      content: (
        <>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
            <div style={{ fontSize: '0.85rem', color: '#a1a1aa' }}>
              Showing: {balanceSel?.mode === 'quarterly' ? `Q${balanceSel?.quarter} ${balanceSel?.year ?? ''}` : `Annual ${balanceSel?.year ?? ''}`}
            </div>
            <button
              onClick={() => {
                const defYear = balanceSel?.year ?? latestAnnualYear(fundamentals?.balance_sheet_annual) ?? null;
                setBalanceDraft(balanceSel ?? { mode: 'annual', year: defYear, quarter: 1 });
                setBalanceExpanded(v => !v);
              }}
              style={{ background: '#111827', color: '#e5e7eb', border: '1px solid #374151', borderRadius: 6, padding: '6px 10px', cursor: 'pointer' }}
            >
              {balanceExpanded ? 'Collapse' : 'Expand'}
            </button>
          </div>
          {balanceExpanded && (
            <div style={{ marginBottom: '12px', padding: '12px', borderRadius: 10, border: '1px solid rgba(255,255,255,0.08)', background: 'rgba(255,255,255,0.03)' }}>
              <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
                <div style={{ flex: '1 1 120px' }}>
                  <div style={{ marginBottom: 6, color: '#a1a1aa', fontSize: '0.8rem' }}>Mode</div>
                  <div style={{ display: 'flex', gap: 8 }}>
                    <button
                      onClick={() => setBalanceDraft(prev => ({ ...(prev ?? { mode: 'annual', year: latestAnnualYear(fundamentals?.balance_sheet_annual), quarter: 1 }), mode: 'annual' }))}
                      style={{
                        padding: '6px 10px',
                        borderRadius: 6,
                        border: '1px solid #374151',
                        background: (balanceDraft?.mode ?? 'annual') === 'annual' ? '#111827' : 'transparent',
                        color: '#e5e7eb',
                        cursor: 'pointer'
                      }}
                    >
                      Annual
                    </button>
                    <button
                      onClick={() => setBalanceDraft(prev => ({ ...(prev ?? { mode: 'quarterly', year: latestAnnualYear(fundamentals?.balance_sheet_annual), quarter: 1 }), mode: 'quarterly' }))}
                      style={{
                        padding: '6px 10px',
                        borderRadius: 6,
                        border: '1px solid #374151',
                        background: (balanceDraft?.mode ?? 'annual') === 'quarterly' ? '#111827' : 'transparent',
                        color: '#e5e7eb',
                        cursor: 'pointer'
                      }}
                    >
                      Quarterly
                    </button>
                  </div>
                </div>
                <div style={{ flex: '1 1 180px' }}>
                  <div style={{ marginBottom: 6, color: '#a1a1aa', fontSize: '0.8rem' }}>Year</div>
                  <select
                    value={balanceDraft?.year ?? ''}
                    onChange={e => {
                      const nextYear = e.target.value ? parseInt(e.target.value, 10) : null;
                      setBalanceDraft(prev => {
                        const base = prev ?? { mode: 'annual', year: null, quarter: 1 };
                        let nextQuarter = base.quarter ?? 1;
                        if (base.mode === 'quarterly') {
                          const opts = availableQuarters(fundamentals?.balance_sheet_quarterly, nextYear, balanceFiscalYearEndMonth);
                          const selectedOpt = opts.find(o => o.value === nextQuarter);
                          if (selectedOpt?.disabled) {
                            nextQuarter = opts.find(o => !o.disabled)?.value ?? 1;
                          }
                        }
                        return { ...base, year: nextYear, quarter: nextQuarter };
                      });
                    }}
                    style={{ width: '100%', padding: 8, background: '#0b0f17', color: '#e5e7eb', border: '1px solid #374151', borderRadius: 6 }}
                  >
                    <option value="">Select year</option>
                    {collectYears(fundamentals?.balance_sheet_annual, fundamentals?.balance_sheet_quarterly, balanceFiscalYearEndMonth).map(y => (
                      <option key={y} value={y}>
                        {y === balanceCurrentFY ? `Current (${y})` : y}
                      </option>
                    ))}
                  </select>
                </div>
                <div style={{ flex: '1 1 160px', opacity: (balanceDraft?.mode ?? 'annual') === 'quarterly' ? 1 : 0.5 }}>
                  <div style={{ marginBottom: 6, color: '#a1a1aa', fontSize: '0.8rem' }}>Quarter</div>
                  <select
                    disabled={(balanceDraft?.mode ?? 'annual') !== 'quarterly'}
                    value={balanceDraft?.quarter ?? 1}
                    onChange={e => setBalanceDraft(prev => ({ ...(prev ?? { mode: 'quarterly', year: latestAnnualYear(fundamentals?.balance_sheet_annual), quarter: 1 }), quarter: parseInt(e.target.value, 10) as 1 | 2 | 3 | 4 }))}
                    style={{ width: '100%', padding: 8, background: '#0b0f17', color: '#e5e7eb', border: '1px solid #374151', borderRadius: 6 }}
                  >
                    {availableQuarters(fundamentals?.balance_sheet_quarterly, balanceDraft?.year ?? null, balanceFiscalYearEndMonth).map(q => (
                      <option key={q.value} value={q.value} disabled={q.disabled}>
                        {q.label}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 12 }}>
                <button
                  onClick={() => {
                    setBalanceDraft(null);
                    setBalanceExpanded(false);
                  }}
                  style={{ background: 'transparent', color: '#e5e7eb', border: '1px solid #374151', borderRadius: 6, padding: '8px 12px', cursor: 'pointer' }}
                >
                  Cancel
                </button>
                <button
                  onClick={() => {
                    if (balanceDraft) setBalanceSel(balanceDraft);
                    setBalanceExpanded(false);
                  }}
                  style={{ background: '#2563eb', color: '#fff', border: '1px solid #1d4ed8', borderRadius: 6, padding: '8px 12px', cursor: 'pointer' }}
                >
                  Apply
                </button>
              </div>
            </div>
          )}
          <div style={sectionStyle}>
            {renderFinancialsAsGrid(
              fundamentals?.balance_sheet_annual,
              fundamentals?.balance_sheet_quarterly,
              "Balance Sheet",
              balanceExpanded
                ? (balanceDraft ?? balanceSel ?? { mode: 'annual', year: latestAnnualYear(fundamentals?.balance_sheet_annual) })
                : (balanceSel ?? { mode: 'annual', year: latestAnnualYear(fundamentals?.balance_sheet_annual) })
            )}
          </div>
        </>
      )
    },
    {
      title: "Cash Flow",
      content: (
        <>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
            <div style={{ fontSize: '0.85rem', color: '#a1a1aa' }}>
              Showing: {cashSel?.mode === 'quarterly' ? `Q${cashSel?.quarter} ${cashSel?.year ?? ''}` : `Annual ${cashSel?.year ?? ''}`}
            </div>
            <button
              onClick={() => {
                const defYear = cashSel?.year ?? latestAnnualYear(fundamentals?.cashflow_annual) ?? null;
                setCashDraft(cashSel ?? { mode: 'annual', year: defYear, quarter: 1 });
                setCashExpanded(v => !v);
              }}
              style={{ background: '#111827', color: '#e5e7eb', border: '1px solid #374151', borderRadius: 6, padding: '6px 10px', cursor: 'pointer' }}
            >
              {cashExpanded ? 'Collapse' : 'Expand'}
            </button>
          </div>
          {cashExpanded && (
            <div style={{ marginBottom: '12px', padding: '12px', borderRadius: 10, border: '1px solid rgba(255,255,255,0.08)', background: 'rgba(255,255,255,0.03)' }}>
              <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
                <div style={{ flex: '1 1 120px' }}>
                  <div style={{ marginBottom: 6, color: '#a1a1aa', fontSize: '0.8rem' }}>Mode</div>
                  <div style={{ display: 'flex', gap: 8 }}>
                    <button
                      onClick={() => setCashDraft(prev => ({ ...(prev ?? { mode: 'annual', year: latestAnnualYear(fundamentals?.cashflow_annual), quarter: 1 }), mode: 'annual' }))}
                      style={{
                        padding: '6px 10px',
                        borderRadius: 6,
                        border: '1px solid #374151',
                        background: (cashDraft?.mode ?? 'annual') === 'annual' ? '#111827' : 'transparent',
                        color: '#e5e7eb',
                        cursor: 'pointer'
                      }}
                    >
                      Annual
                    </button>
                    <button
                      onClick={() => setCashDraft(prev => ({ ...(prev ?? { mode: 'quarterly', year: latestAnnualYear(fundamentals?.cashflow_annual), quarter: 1 }), mode: 'quarterly' }))}
                      style={{
                        padding: '6px 10px',
                        borderRadius: 6,
                        border: '1px solid #374151',
                        background: (cashDraft?.mode ?? 'annual') === 'quarterly' ? '#111827' : 'transparent',
                        color: '#e5e7eb',
                        cursor: 'pointer'
                      }}
                    >
                      Quarterly
                    </button>
                  </div>
                </div>
                <div style={{ flex: '1 1 180px' }}>
                  <div style={{ marginBottom: 6, color: '#a1a1aa', fontSize: '0.8rem' }}>Year</div>
                  <select
                    value={cashDraft?.year ?? ''}
                    onChange={e => {
                      const nextYear = e.target.value ? parseInt(e.target.value, 10) : null;
                      setCashDraft(prev => {
                        const base = prev ?? { mode: 'annual', year: null, quarter: 1 };
                        let nextQuarter = base.quarter ?? 1;
                        if (base.mode === 'quarterly') {
                          const opts = availableQuarters(fundamentals?.cashflow_quarterly, nextYear, cashFiscalYearEndMonth);
                          const selectedOpt = opts.find(o => o.value === nextQuarter);
                          if (selectedOpt?.disabled) {
                            nextQuarter = opts.find(o => !o.disabled)?.value ?? 1;
                          }
                        }
                        return { ...base, year: nextYear, quarter: nextQuarter };
                      });
                    }}
                    style={{ width: '100%', padding: 8, background: '#0b0f17', color: '#e5e7eb', border: '1px solid #374151', borderRadius: 6 }}
                  >
                    <option value="">Select year</option>
                    {collectYears(fundamentals?.cashflow_annual, fundamentals?.cashflow_quarterly, cashFiscalYearEndMonth).map(y => (
                      <option key={y} value={y}>
                        {y === cashCurrentFY ? `Current (${y})` : y}
                      </option>
                    ))}
                  </select>
                </div>
                <div style={{ flex: '1 1 160px', opacity: (cashDraft?.mode ?? 'annual') === 'quarterly' ? 1 : 0.5 }}>
                  <div style={{ marginBottom: 6, color: '#a1a1aa', fontSize: '0.8rem' }}>Quarter</div>
                  <select
                    disabled={(cashDraft?.mode ?? 'annual') !== 'quarterly'}
                    value={cashDraft?.quarter ?? 1}
                    onChange={e => setCashDraft(prev => ({ ...(prev ?? { mode: 'quarterly', year: latestAnnualYear(fundamentals?.cashflow_annual), quarter: 1 }), quarter: parseInt(e.target.value, 10) as 1 | 2 | 3 | 4 }))}
                    style={{ width: '100%', padding: 8, background: '#0b0f17', color: '#e5e7eb', border: '1px solid #374151', borderRadius: 6 }}
                  >
                    {availableQuarters(fundamentals?.cashflow_quarterly, cashDraft?.year ?? null, cashFiscalYearEndMonth).map(q => (
                      <option key={q.value} value={q.value} disabled={q.disabled}>
                        {q.label}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 12 }}>
                <button
                  onClick={() => {
                    setCashDraft(null);
                    setCashExpanded(false);
                  }}
                  style={{ background: 'transparent', color: '#e5e7eb', border: '1px solid #374151', borderRadius: 6, padding: '8px 12px', cursor: 'pointer' }}
                >
                  Cancel
                </button>
                <button
                  onClick={() => {
                    if (cashDraft) setCashSel(cashDraft);
                    setCashExpanded(false);
                  }}
                  style={{ background: '#2563eb', color: '#fff', border: '1px solid #1d4ed8', borderRadius: 6, padding: '8px 12px', cursor: 'pointer' }}
                >
                  Apply
                </button>
              </div>
            </div>
          )}
          <div style={sectionStyle}>
            {renderFinancialsAsGrid(
              fundamentals?.cashflow_annual,
              fundamentals?.cashflow_quarterly,
              "Cash Flow",
              cashExpanded
                ? (cashDraft ?? cashSel ?? { mode: 'annual', year: latestAnnualYear(fundamentals?.cashflow_annual) })
                : (cashSel ?? { mode: 'annual', year: latestAnnualYear(fundamentals?.cashflow_annual) })
            )}
          </div>
        </>
      )
    },
    {
      title: "Institutional Holders",
      content: (
        <>
          <div style={sectionStyle}>
            {renderHoldersAsGrid(fundamentals?.institutional_holders, "Institutional Holders")}
          </div>
        </>
      )
    }
  ];

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', color: '#fff' }}>
      <div style={{ marginBottom: '16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h3 style={{ margin: 0 }}>Financial Fundamentals</h3>
      </div>

      <div style={{ flex: 1, overflowY: 'auto', paddingRight: '4px', display: 'flex', flexDirection: 'column' }}>
        {loading ? (
          <div style={centerStyle}>Loading fundamentals...</div>
        ) : !fundamentals ? (
          <div style={centerStyle}>No fundamentals found for {symbol}</div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '24px', height: '100%' }}>
            {/* Header Info */}
            <div style={{
                background: 'rgba(255,255,255,0.03)',
                padding: '16px',
                borderRadius: '12px',
                border: '1px solid rgba(255,255,255,0.05)',
                flexShrink: 0
              }}>
              <div>
                <h1 style={{ margin: 0, fontSize: '1.8rem' }}>{fundamentals.ticker}</h1>
                <div style={{ fontSize: '0.9rem', color: '#a1a1aa' }}>
                  Period: {new Date(fundamentals.period).toLocaleDateString()}
                </div>
              </div>
            </div>

            {/* Horizontal Scrollable Pages */}
            <div style={{ 
              flex: 1, 
              display: 'flex', 
              flexDirection: 'row', 
              overflowX: 'auto', 
              gap: '20px', 
              scrollSnapType: 'x mandatory',
              paddingBottom: '10px' // Space for scrollbar
            }}>
              {pages.map((page, index) => (
                <div key={index} style={{ 
                  minWidth: '100%', 
                  scrollSnapAlign: 'start', 
                  display: 'flex', 
                  flexDirection: 'column'
                }}>
                  <h3 style={{ 
                    marginTop: 0, 
                    marginBottom: '12px', 
                    fontSize: '1.1rem', 
                    color: '#e5e7eb',
                    borderBottom: '1px solid rgba(255,255,255,0.1)',
                    paddingBottom: '8px'
                  }}>
                    {page.title}
                  </h3>
                  <div style={{ flex: 1, overflowY: 'auto' }}>
                    {page.content}
                  </div>
                </div>
              ))}
            </div>
          </div>
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
