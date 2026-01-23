import React, { useEffect, useState } from 'react';
import { getPrice } from '../api/client';
import StockChart from './StockChart';

interface PriceData {
  symbol: string;
  price: string;
  timestamp: string;
}

const Dashboard: React.FC = () => {
  const [prices, setPrices] = useState<Record<string, PriceData>>({});
  const [chartData, setChartData] = useState<any[]>([]);
  const symbols = ['AAPL', 'GOOGL', 'TSLA', 'MSFT', 'AMZN'];

  useEffect(() => {
    const fetchData = async () => {
      const newPrices: Record<string, PriceData> = {};
      const now = new Date().toLocaleTimeString();

      for (const symbol of symbols) {
        try {
          const data = await getPrice(symbol);
          newPrices[symbol] = data;
          
          if (symbol === 'AAPL') {
            setChartData(prev => [
              ...prev.slice(-19), // Keep last 20 points
              { timestamp: now, price: parseFloat(data.price) }
            ]);
          }
        } catch (error) {
          console.error(`Failed to fetch price for ${symbol}`, error);
        }
      }
      setPrices(newPrices);
    };

    fetchData();
    const interval = setInterval(fetchData, 1000); // Poll every second

    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{ padding: '20px' }}>
      <h1>Scope Financial Dashboard</h1>
      
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '20px', marginBottom: '20px' }}>
        {symbols.map(symbol => (
          <div key={symbol} style={{ border: '1px solid #ccc', padding: '15px', borderRadius: '8px' }}>
            <h3>{symbol}</h3>
            <p style={{ fontSize: '24px', fontWeight: 'bold' }}>
              ${prices[symbol]?.price ? parseFloat(prices[symbol].price).toFixed(2) : 'Loading...'}
            </p>
          </div>
        ))}
      </div>

      <div style={{ border: '1px solid #ccc', padding: '20px', borderRadius: '8px' }}>
        <StockChart data={chartData} symbol="AAPL" />
      </div>
    </div>
  );
};

export default Dashboard;
