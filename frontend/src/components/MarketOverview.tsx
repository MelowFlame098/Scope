import React, { useEffect, useState } from 'react';
import { getMarketMovers } from '../api/client';

interface Mover {
  symbol: string;
  name: string;
  price: string;
  change_percent: string;
}

const MarketOverview: React.FC = () => {
  const [topMovers, setTopMovers] = useState<Mover[]>([]);
  const [worstMovers, setWorstMovers] = useState<Mover[]>([]);

  // Mock data for indices and movers
  const indices = [
    { name: 'S&P 500', value: '4,783.45', change: '+0.45%', desc: 'Standard & Poor\'s 500' },
    { name: 'NASDAQ', value: '15,055.65', change: '+0.75%', desc: 'Nasdaq Composite' },
    { name: 'QQQ', value: '369.25', change: '+0.82%', desc: 'Invesco QQQ Trust (Nasdaq-100)' },
    // Replaced DOW with Sector/Theme ETFs
    { name: 'ICLN', value: '14.52', change: '+1.20%', desc: 'iShares Global Clean Energy ETF' },
    { name: 'VEGN', value: '35.18', change: '+0.95%', desc: 'US Vegan Climate ETF (Wildlife/Eco-friendly)' },
    { name: 'SOXX', value: '562.40', change: '+2.10%', desc: 'iShares Semiconductor ETF' },
    { name: 'DTCR', value: '15.85', change: '+0.50%', desc: 'Global X Data Center REITs ETF' },
    { name: 'VGT', value: '485.30', change: '+1.45%', desc: 'Vanguard Information Technology ETF' },
    { name: 'XRT', value: '72.15', change: '-0.35%', desc: 'SPDR S&P Retail ETF' },
  ];

  useEffect(() => {
    const fetchMovers = async () => {
      try {
        const data = await getMarketMovers();
        setTopMovers(data.top || []);
        setWorstMovers(data.worst || []);
      } catch (error) {
        console.error("Failed to fetch movers:", error);
      }
    };

    fetchMovers();
    // Poll every 5 seconds
    const interval = setInterval(fetchMovers, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{ 
      gridColumn: '1 / -1', 
      overflowX: 'auto', 
      display: 'flex', 
      gap: '16px', 
      paddingBottom: '8px',
      scrollbarWidth: 'none' // Hide scrollbar for cleaner look
    }}>
      {/* Indices Group */}
      <div style={{ display: 'flex', gap: '16px', paddingRight: '24px', borderRight: '1px solid rgba(255,255,255,0.1)' }}>
        {indices.map(idx => (
          <div 
            key={idx.name} 
            className="glass-card" 
            style={{ minWidth: '140px', padding: '12px', cursor: 'help' }}
            title={idx.desc}
          >
            <div style={{ fontSize: '0.8rem', color: '#a1a1aa' }}>{idx.name}</div>
            <div style={{ fontSize: '1.1rem', fontWeight: 'bold' }}>{idx.value}</div>
            <div style={{ 
              fontSize: '0.8rem', 
              color: idx.change.startsWith('+') ? '#10b981' : '#ef4444' 
            }}>
              {idx.change}
            </div>
          </div>
        ))}
      </div>

      {/* Top Movers Group */}
      <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
        <span style={{ color: '#a1a1aa', fontSize: '0.8rem', writingMode: 'vertical-rl', transform: 'rotate(180deg)', whiteSpace: 'nowrap' }}>NASDAQ GAINERS</span>
        {topMovers.length > 0 ? topMovers.map(mover => (
          <div key={mover.symbol} className="glass-card" style={{ minWidth: '120px', padding: '12px' }} title={mover.name}>
            <div style={{ fontWeight: 'bold' }}>{mover.symbol}</div>
            <div style={{ color: '#10b981', fontWeight: 'bold' }}>+{parseFloat(mover.change_percent).toFixed(2)}%</div>
            <div style={{ fontSize: '0.7rem', color: '#a1a1aa', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{mover.name}</div>
          </div>
        )) : <div style={{ color: '#a1a1aa', padding: '10px' }}>Loading...</div>}
      </div>

      {/* Worst Movers Group */}
      <div style={{ display: 'flex', gap: '16px', alignItems: 'center', marginLeft: '16px', paddingLeft: '24px', borderLeft: '1px solid rgba(255,255,255,0.1)' }}>
        <span style={{ color: '#a1a1aa', fontSize: '0.8rem', writingMode: 'vertical-rl', transform: 'rotate(180deg)', whiteSpace: 'nowrap' }}>NASDAQ LOSERS</span>
        {worstMovers.length > 0 ? worstMovers.map(mover => (
          <div key={mover.symbol} className="glass-card" style={{ minWidth: '120px', padding: '12px' }} title={mover.name}>
            <div style={{ fontWeight: 'bold' }}>{mover.symbol}</div>
            <div style={{ color: '#ef4444', fontWeight: 'bold' }}>{parseFloat(mover.change_percent).toFixed(2)}%</div>
            <div style={{ fontSize: '0.7rem', color: '#a1a1aa', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{mover.name}</div>
          </div>
        )) : <div style={{ color: '#a1a1aa', padding: '10px' }}>Loading...</div>}
      </div>
    </div>
  );
};

export default MarketOverview;
