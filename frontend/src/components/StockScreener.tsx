import React, { useState, useEffect } from 'react';
import { getScreenerResults, ScreenerResult } from '../api/client';

const StockScreener: React.FC = () => {
  const [filters, setFilters] = useState({
    marketCap: 'Any',
    peRatio: 'Any',
    dividendYield: 'Any',
    price: 'Any',
    volume: 'Any',
    sector: 'Any'
  });

  const [results, setResults] = useState<ScreenerResult[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    // Simulate fetching with filters
    const fetchResults = async () => {
      setLoading(true);
      try {
        const data = await getScreenerResults();
        setResults(data);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    fetchResults();
  }, [filters]);

  const handleFilterChange = (key: string, value: string) => {
    setFilters(prev => ({ ...prev, [key]: value }));
  };

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div style={{ marginBottom: '20px' }}>
        <h3 style={{ marginTop: 0, marginBottom: '16px', color: '#fff' }}>Stock Screener</h3>
        
        {/* Filter Grid */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '12px' }}>
          <FilterSelect 
            label="Market Cap" 
            value={filters.marketCap} 
            options={['Any', 'Mega (>200B)', 'Large (10B-200B)', 'Mid (2B-10B)', 'Small (<2B)']} 
            onChange={(v) => handleFilterChange('marketCap', v)} 
          />
          <FilterSelect 
            label="P/E Ratio" 
            value={filters.peRatio} 
            options={['Any', 'Under 15', '15-25', 'Over 25', 'Under 50']} 
            onChange={(v) => handleFilterChange('peRatio', v)} 
          />
          <FilterSelect 
            label="Dividend Yield" 
            value={filters.dividendYield} 
            options={['Any', 'None', '> 1%', '> 3%', '> 5%', 'High (>10%)']} 
            onChange={(v) => handleFilterChange('dividendYield', v)} 
          />
          <FilterSelect 
            label="Volume" 
            value={filters.volume} 
            options={['Any', '> 100K', '> 500K', '> 1M', '> 10M']} 
            onChange={(v) => handleFilterChange('volume', v)} 
          />
          <FilterSelect 
            label="Sector" 
            value={filters.sector} 
            options={['Any', 'Technology', 'Finance', 'Healthcare', 'Energy', 'Consumer']} 
            onChange={(v) => handleFilterChange('sector', v)} 
          />
        </div>
      </div>

      {/* Results Table */}
      <div style={{ flex: 1, overflow: 'auto', background: 'rgba(0,0,0,0.2)', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.9rem' }}>
          <thead style={{ position: 'sticky', top: 0, background: '#0a0a0a', zIndex: 10 }}>
            <tr style={{ textAlign: 'left', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
              <th style={{ padding: '12px', color: '#a1a1aa' }}>Ticker</th>
              <th style={{ padding: '12px', color: '#a1a1aa' }}>Company</th>
              <th style={{ padding: '12px', color: '#a1a1aa' }}>Sector</th>
              <th style={{ padding: '12px', color: '#a1a1aa' }}>Price</th>
              <th style={{ padding: '12px', color: '#a1a1aa' }}>Change</th>
              <th style={{ padding: '12px', color: '#a1a1aa' }}>P/E</th>
              <th style={{ padding: '12px', color: '#a1a1aa' }}>Volume</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr><td colSpan={7} style={{ padding: '20px', textAlign: 'center', color: '#a1a1aa' }}>Loading results...</td></tr>
            ) : results.length > 0 ? (
              results.map((item) => (
                <tr key={item.id} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)', transition: 'background 0.2s' }} className="hover:bg-white/5">
                  <td style={{ padding: '12px', fontWeight: 'bold', color: '#8b5cf6' }}>{item.ticker}</td>
                  <td style={{ padding: '12px' }}>{item.company || item.ticker}</td>
                  <td style={{ padding: '12px', color: '#a1a1aa' }}>{item.sector || '-'}</td>
                  <td style={{ padding: '12px' }}>{item.price}</td>
                  <td style={{ padding: '12px', color: item.change.includes('-') ? '#ef4444' : '#10b981' }}>{item.change}</td>
                  <td style={{ padding: '12px' }}>{item.pe || '-'}</td>
                  <td style={{ padding: '12px', color: '#a1a1aa' }}>{item.volume}</td>
                </tr>
              ))
            ) : (
              <tr><td colSpan={7} style={{ padding: '20px', textAlign: 'center', color: '#a1a1aa' }}>No matches found</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};

const FilterSelect: React.FC<{ label: string; value: string; options: string[]; onChange: (v: string) => void }> = ({ label, value, options, onChange }) => (
  <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
    <label style={{ fontSize: '0.75rem', color: '#a1a1aa' }}>{label}</label>
    <select 
      value={value} 
      onChange={(e) => onChange(e.target.value)}
      style={{ 
        background: 'rgba(255,255,255,0.05)', 
        border: '1px solid rgba(255,255,255,0.1)', 
        borderRadius: '6px', 
        color: '#fff', 
        padding: '8px',
        fontSize: '0.85rem',
        outline: 'none'
      }}
    >
      {options.map(opt => <option key={opt} value={opt} style={{ background: '#18181b' }}>{opt}</option>)}
    </select>
  </div>
);

export default StockScreener;
