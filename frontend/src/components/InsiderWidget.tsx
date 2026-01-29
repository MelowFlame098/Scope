import React, { useEffect, useState } from 'react';
import { getInsiderTrades, InsiderTrade } from '../api/client';

const InsiderWidget: React.FC = () => {
  const [trades, setTrades] = useState<InsiderTrade[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    const fetchInsiderData = async () => {
      try {
        const data = await getInsiderTrades();
        setTrades(data);
      } catch (error) {
        console.error("Failed to fetch insider trades", error);
      } finally {
        setLoading(false);
      }
    };

    fetchInsiderData();
  }, []);

  if (loading) {
    return <div>Loading Insider Trades...</div>;
  }

  return (
    <div style={{ border: '1px solid #ccc', padding: '15px', borderRadius: '8px', height: '300px', overflowY: 'auto' }}>
      <h3>Insider Activity</h3>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.9em' }}>
        <thead>
          <tr style={{ textAlign: 'left', borderBottom: '1px solid #eee' }}>
            <th>Ticker</th>
            <th>Owner</th>
            <th>Trans</th>
            <th>Value</th>
            <th>Date</th>
          </tr>
        </thead>
        <tbody>
          {trades.map((trade) => (
            <tr key={trade.id} style={{ borderBottom: '1px solid #f9f9f9' }}>
              <td style={{ padding: '8px 0', fontWeight: 'bold' }}>{trade.ticker}</td>
              <td style={{ maxWidth: '100px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={trade.owner}>
                {trade.owner}
              </td>
              <td style={{ 
                color: trade.transaction.toLowerCase().includes('buy') ? 'green' : 'red',
                fontWeight: 'bold' 
              }}>
                {trade.transaction}
              </td>
              <td>{trade.value}</td>
              <td style={{ color: '#666' }}>{trade.date}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default InsiderWidget;
