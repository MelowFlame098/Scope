import React, { useEffect, useState } from 'react';
import { getScreenerResults, ScreenerResult } from '../api/client';

const ScreenerWidget: React.FC = () => {
  const [results, setResults] = useState<ScreenerResult[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    const fetchScreenerData = async () => {
      try {
        // Fetch top gainers or default results
        const data = await getScreenerResults(); 
        setResults(data);
      } catch (error) {
        console.error("Failed to fetch screener results", error);
      } finally {
        setLoading(false);
      }
    };

    fetchScreenerData();
  }, []);

  if (loading) {
    return <div>Loading Screener...</div>;
  }

  return (
    <div style={{ border: '1px solid #ccc', padding: '15px', borderRadius: '8px', height: '300px', overflowY: 'auto' }}>
      <h3>Top Movers</h3>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr style={{ textAlign: 'left', borderBottom: '1px solid #eee' }}>
            <th>Ticker</th>
            <th>Price</th>
            <th>Change</th>
            <th>Vol</th>
          </tr>
        </thead>
        <tbody>
          {results.map((item) => (
            <tr key={item.id} style={{ borderBottom: '1px solid #f9f9f9' }}>
              <td style={{ padding: '8px 0', fontWeight: 'bold' }}>{item.ticker}</td>
              <td>{item.price}</td>
              <td style={{ 
                color: item.change.includes('-') ? 'red' : 'green',
                fontWeight: 'bold'
              }}>
                {item.change}
              </td>
              <td style={{ fontSize: '0.8em', color: '#666' }}>{item.volume}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default ScreenerWidget;
