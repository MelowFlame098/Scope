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
    return <div style={{ color: '#a1a1aa' }}>Loading Insider Trades...</div>;
  }

  return (
    <div style={{ height: '100%', overflowY: 'auto' }}>
      <h3 style={{ marginTop: 0, marginBottom: '1rem', color: '#fff' }}>Insider Activity</h3>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.9em' }}>
        <thead>
          <tr style={{ textAlign: 'left', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
            <th style={{ color: '#a1a1aa', padding: '8px 0' }}>Ticker</th>
            <th style={{ color: '#a1a1aa', padding: '8px 0' }}>Owner</th>
            <th style={{ color: '#a1a1aa', padding: '8px 0' }}>Trans</th>
            <th style={{ color: '#a1a1aa', padding: '8px 0' }}>Value</th>
            <th style={{ color: '#a1a1aa', padding: '8px 0' }}>Date</th>
          </tr>
        </thead>
        <tbody>
          {trades.map((trade) => (
            <tr key={trade.id} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
              <td style={{ padding: '12px 0', fontWeight: 'bold', color: '#fff' }}>{trade.ticker}</td>
              <td style={{ maxWidth: '100px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', padding: '12px 0', color: '#a1a1aa' }} title={trade.owner}>
                {trade.owner}
              </td>
              <td style={{ 
                padding: '12px 0',
                color: trade.transaction.toLowerCase().includes('buy') ? '#10b981' : '#ef4444',
                fontWeight: 'bold' 
              }}>
                {trade.transaction}
              </td>
              <td style={{ padding: '12px 0', color: '#fff' }}>
                {new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(trade.value)}
              </td>
              <td style={{ padding: '12px 0', color: '#a1a1aa' }}>{trade.date}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default InsiderWidget;
