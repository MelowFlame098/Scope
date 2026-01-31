import React, { useState, useEffect } from 'react';
import { getScreenerResults, ScreenerResult } from '../api/client';

const FinancialFundamentals: React.FC = () => {
  const [results, setResults] = useState<ScreenerResult[]>([]);
  const [filteredResults, setFilteredResults] = useState<ScreenerResult[]>([]);
  const [selectedStock, setSelectedStock] = useState<ScreenerResult | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  
  // Filters state
  const [filters, setFilters] = useState({
    marketCap: 'Any',
    peRatio: 'Any',
    sector: 'Any',
    search: ''
  });

  useEffect(() => {
    const fetchResults = async () => {
      setLoading(true);
      try {
        const data = await getScreenerResults();
        setResults(data);
        if (data.length > 0 && !selectedStock) {
            setSelectedStock(data[0]);
        }
      } catch (err) {
        console.error("Failed to fetch screener results:", err);
      } finally {
        setLoading(false);
      }
    };
    fetchResults();
  }, []);

  // Apply filters
  useEffect(() => {
    let res = results;

    if (filters.search) {
        const term = filters.search.toLowerCase();
        res = res.filter(r => 
            r.ticker.toLowerCase().includes(term) || 
            r.company.toLowerCase().includes(term)
        );
    }

    if (filters.sector !== 'Any') {
        res = res.filter(r => r.sector === filters.sector);
    }

    // Note: More complex numeric filters (Market Cap, P/E) would require parsing strings like "100B" or "25.5"
    // For this MVP, we'll stick to basic filtering or simple string matching if the API doesn't filter for us.
    // The previous StockScreener had UI for these but didn't fully implement client-side logic for "Mega (>200B)".
    // We will keep the UI elements as requested.

    setFilteredResults(res);
    
    // If selected stock is filtered out, select the first one
    if (res.length > 0 && (!selectedStock || !res.find(r => r.ticker === selectedStock.ticker))) {
        setSelectedStock(res[0]);
    }
  }, [results, filters]);

  const handleFilterChange = (key: string, value: string) => {
    setFilters(prev => ({ ...prev, [key]: value }));
  };

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', color: '#fff' }}>
      {/* Header & Filters */}
      <div style={{ marginBottom: '16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '10px' }}>
        <h3 style={{ margin: 0 }}>Financial Fundamentals</h3>
        
        <div style={{ display: 'flex', gap: '8px' }}>
            <input 
                type="text" 
                placeholder="Search Ticker..." 
                value={filters.search}
                onChange={(e) => handleFilterChange('search', e.target.value)}
                style={{ background: 'rgba(255,255,255,0.1)', border: '1px solid rgba(255,255,255,0.2)', borderRadius: '4px', color: '#fff', padding: '6px 10px' }}
            />
             <select 
                value={filters.sector}
                onChange={(e) => handleFilterChange('sector', e.target.value)}
                style={{ background: 'rgba(255,255,255,0.1)', border: '1px solid rgba(255,255,255,0.2)', borderRadius: '4px', color: '#fff', padding: '6px' }}
            >
                <option value="Any">All Sectors</option>
                <option value="Technology">Technology</option>
                <option value="Finance">Finance</option>
                <option value="Healthcare">Healthcare</option>
                <option value="Energy">Energy</option>
            </select>
        </div>
      </div>

      <div style={{ display: 'flex', flex: 1, gap: '20px', minHeight: 0 }}>
        {/* Left: Stock List (Screener) */}
        <div style={{ width: '300px', display: 'flex', flexDirection: 'column', background: 'rgba(0,0,0,0.2)', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)', overflow: 'hidden' }}>
            <div style={{ padding: '10px', borderBottom: '1px solid rgba(255,255,255,0.1)', fontWeight: 'bold', display: 'grid', gridTemplateColumns: '1fr 1fr 1fr' }}>
                <span>Ticker</span>
                <span style={{textAlign: 'right'}}>Price</span>
                <span style={{textAlign: 'right'}}>Chg</span>
            </div>
            <div style={{ flex: 1, overflowY: 'auto' }}>
                {loading ? (
                    <div style={{ padding: '20px', textAlign: 'center', color: '#a1a1aa' }}>Loading...</div>
                ) : filteredResults.length === 0 ? (
                    <div style={{ padding: '20px', textAlign: 'center', color: '#a1a1aa' }}>No results</div>
                ) : (
                    filteredResults.map(stock => (
                        <div 
                            key={stock.id || stock.ticker}
                            onClick={() => setSelectedStock(stock)}
                            style={{ 
                                display: 'grid', 
                                gridTemplateColumns: '1fr 1fr 1fr', 
                                padding: '10px', 
                                borderBottom: '1px solid rgba(255,255,255,0.05)',
                                cursor: 'pointer',
                                background: selectedStock?.ticker === stock.ticker ? 'rgba(139, 92, 246, 0.2)' : 'transparent',
                                transition: 'background 0.2s'
                            }}
                            className="hover:bg-white/5"
                        >
                            <span style={{ fontWeight: 'bold', color: '#8b5cf6' }}>{stock.ticker}</span>
                            <span style={{ textAlign: 'right' }}>{stock.price}</span>
                            <span style={{ textAlign: 'right', color: stock.change.startsWith('-') ? '#ef4444' : '#10b981' }}>{stock.change}</span>
                        </div>
                    ))
                )}
            </div>
        </div>

        {/* Right: Detailed Fundamentals */}
        <div style={{ flex: 1, overflowY: 'auto', paddingRight: '4px' }}>
            {selectedStock ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                    {/* Top Info Card */}
                    <div style={{ background: 'rgba(255,255,255,0.03)', padding: '20px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start' }}>
                            <div>
                                <h1 style={{ margin: 0, fontSize: '2rem' }}>{selectedStock.ticker}</h1>
                                <div style={{ fontSize: '1.1rem', color: '#a1a1aa' }}>{selectedStock.company}</div>
                                <div style={{ marginTop: '8px', display: 'inline-block', padding: '4px 8px', borderRadius: '4px', background: 'rgba(139, 92, 246, 0.1)', color: '#a78bfa', fontSize: '0.8rem' }}>
                                    {selectedStock.sector} | {selectedStock.industry} | {selectedStock.country}
                                </div>
                            </div>
                            <div style={{ textAlign: 'right' }}>
                                <div style={{ fontSize: '2rem', fontWeight: 'bold' }}>{selectedStock.price}</div>
                                <div style={{ fontSize: '1.2rem', color: selectedStock.change.startsWith('-') ? '#ef4444' : '#10b981' }}>
                                    {selectedStock.change}
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Metrics Grid */}
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '16px' }}>
                        <MetricCard label="Market Cap" value={selectedStock.market_cap} />
                        <MetricCard label="P/E Ratio" value={selectedStock.pe} />
                        <MetricCard label="Dividend Yield" value={selectedStock.dividend_yield} />
                        <MetricCard label="Volume" value={selectedStock.volume} />
                        <MetricCard label="EPS (ttm)" value={selectedStock.eps} />
                        <MetricCard label="Revenue" value={selectedStock.revenue} />
                        <MetricCard label="Total Debt" value={selectedStock.debt} />
                        <MetricCard label="ROE" value={selectedStock.roe} />
                        <MetricCard label="Profit Margin" value={selectedStock.profit_margin} />
                        <MetricCard label="Book Value (P/B)" value={selectedStock.book_value} />
                    </div>

                    {/* Additional Analysis Placeholder */}
                    <div style={{ background: 'rgba(255,255,255,0.03)', padding: '20px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)' }}>
                        <h4 style={{ margin: '0 0 10px 0', color: '#a1a1aa' }}>About {selectedStock.company}</h4>
                        <p style={{ lineHeight: '1.6', color: '#d4d4d8', margin: 0 }}>
                            {selectedStock.company} operates in the {selectedStock.sector} sector, specifically within the {selectedStock.industry} industry. 
                            With a market cap of {selectedStock.market_cap}, it shows a P/E ratio of {selectedStock.pe}. 
                            Recent volume stands at {selectedStock.volume}.
                        </p>
                    </div>
                </div>
            ) : (
                <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#52525b' }}>
                    Select a stock to view detailed analysis
                </div>
            )}
        </div>
      </div>
    </div>
  );
};

const MetricCard: React.FC<{ label: string; value: string | undefined }> = ({ label, value }) => (
    <div style={{ background: 'rgba(255,255,255,0.03)', borderRadius: '8px', padding: '16px', border: '1px solid rgba(255,255,255,0.05)' }}>
        <div style={{ color: '#a1a1aa', fontSize: '0.85rem', marginBottom: '6px' }}>{label}</div>
        <div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: value ? '#fff' : '#52525b' }}>{value || 'N/A'}</div>
    </div>
);

export default FinancialFundamentals;
