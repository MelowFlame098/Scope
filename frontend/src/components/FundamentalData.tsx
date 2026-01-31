import React, { useState } from 'react';

interface FundamentalMetrics {
  eps: string;
  revenue: string;
  debt: string;
  roe: string;
  profitMargin: string;
  bookValue: string;
}

const FundamentalData: React.FC = () => {
  const [symbol, setSymbol] = useState<string>('AAPL');
  
  // Mock data - In real app, fetch from API
  const metrics: FundamentalMetrics = {
    eps: '6.13',
    revenue: '$383.29B',
    debt: '$111.09B',
    roe: '160.09%',
    profitMargin: '25.31%',
    bookValue: '$4.18'
  };

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
        <h3 style={{ margin: 0, color: '#fff' }}>Fundamentals</h3>
        <input 
          type="text" 
          value={symbol}
          onChange={(e) => setSymbol(e.target.value.toUpperCase())}
          style={{ 
            background: 'rgba(255,255,255,0.1)', 
            border: '1px solid rgba(255,255,255,0.2)', 
            borderRadius: '4px',
            color: '#fff',
            padding: '4px 8px',
            width: '80px',
            fontSize: '0.8rem'
          }}
        />
      </div>
      
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', flex: 1 }}>
        <MetricItem label="EPS" value={metrics.eps} />
        <MetricItem label="Revenue" value={metrics.revenue} />
        <MetricItem label="Debt" value={metrics.debt} />
        <MetricItem label="ROE" value={metrics.roe} />
        <MetricItem label="Profit Margin" value={metrics.profitMargin} />
        <MetricItem label="Book Value" value={metrics.bookValue} />
      </div>
    </div>
  );
};

const MetricItem: React.FC<{ label: string; value: string }> = ({ label, value }) => (
  <div style={{ 
    background: 'rgba(255,255,255,0.03)', 
    borderRadius: '8px', 
    padding: '10px',
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'center'
  }}>
    <div style={{ fontSize: '0.75rem', color: '#a1a1aa', marginBottom: '4px' }}>{label}</div>
    <div style={{ fontSize: '1rem', fontWeight: 'bold', color: '#fff' }}>{value}</div>
  </div>
);

export default FundamentalData;
