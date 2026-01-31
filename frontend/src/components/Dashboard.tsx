import React, { useEffect, useState } from 'react';
import { getPrice } from '../api/client';
import StockChart from './StockChart';
import InsiderWidget from './InsiderWidget';
import SectorHeatmap from './SectorHeatmap';
import MarketOverview from './MarketOverview';
import FinancialFundamentals from './FinancialFundamentals';

interface PriceData {
  symbol: string;
  price: string;
  timestamp: string;
}

const Dashboard: React.FC = () => {
  const [chartData, setChartData] = useState<any[]>([]);
  // We'll keep one symbol for the main chart for now
  const chartSymbol = 'AAPL'; 

  useEffect(() => {
    const fetchData = async () => {
      const now = new Date().toLocaleTimeString();

      try {
        const data = await getPrice(chartSymbol);
        
        setChartData(prev => [
          ...prev.slice(-19), // Keep last 20 points
          { timestamp: now, price: parseFloat(data.price) }
        ]);
      } catch (error) {
        console.error(`Failed to fetch price for ${chartSymbol}`, error);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 1000); // Poll every second

    return () => clearInterval(interval);
  }, []);

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
          <StockChart data={chartData} symbol={chartSymbol} />
      </div>

      {/* Financial Fundamentals & Screener (Combined) */}
      <div className="glass-card" style={{ gridColumn: '1 / -1', minHeight: '600px', overflow: 'hidden' }}>
        <FinancialFundamentals />
      </div>

      {/* Insider Widget */}
      <div className="glass-card" style={{ gridColumn: 'span 6', minHeight: '300px' }}>
        <InsiderWidget />
      </div>

      {/* Sector Heatmap */}
      <div className="glass-card" style={{ gridColumn: 'span 6', minHeight: '300px' }}>
        <SectorHeatmap />
      </div>
    </div>
  );
};

export default Dashboard;
