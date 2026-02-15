import React from 'react';
import StockChart from './StockChart';
import InsiderWidget from './InsiderWidget';
import SectorHeatmap from './SectorHeatmap';
import MarketOverview from './MarketOverview';
import FinancialFundamentals from './FinancialFundamentals';

const Dashboard: React.FC = () => {
  // We'll keep one symbol for the main chart for now
  const chartSymbol = 'AAPL'; 

  return (
    <div className="dashboard-grid">
      <div style={{ gridColumn: '1 / -1', marginBottom: '1rem' }}>
        <h1 style={{ background: 'linear-gradient(to right, #fff, #a1a1aa)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
          Scope Financial Dashboard
        </h1>
      </div>
      
      {/* Market Overview (Indices & Movers) */}
      <MarketOverview />

      {/* Main Chart Area */}
      <div className="glass-card" style={{ gridColumn: '1 / -1', minHeight: '400px' }}>
          <h3 style={{ marginTop: 0, marginBottom: '20px' }}>{chartSymbol} Live Chart</h3>
          <StockChart symbol={chartSymbol} />
      </div>

      {/* Financial Fundamentals */}
      <div className="glass-card" style={{ gridColumn: '1 / -1', minHeight: '320px', overflow: 'hidden' }}>
        <FinancialFundamentals symbol={chartSymbol} />
      </div>

      {/* Insider Widget */}
      <div className="glass-card" style={{ gridColumn: 'span 6', height: '250px', overflow: 'hidden' }}>
        <InsiderWidget />
      </div>

      {/* Sector Heatmap */}
      <div className="glass-card" style={{ gridColumn: 'span 6', height: '250px', overflow: 'hidden' }}>
        <SectorHeatmap />
      </div>
    </div>
  );
};

export default Dashboard;
