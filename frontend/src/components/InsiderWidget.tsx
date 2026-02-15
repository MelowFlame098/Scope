import React, { useEffect, useState } from 'react';
import { getInsiderTrades, InsiderTrade } from '../api/client';

const InsiderWidget: React.FC = () => {
  const [trades, setTrades] = useState<InsiderTrade[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    const fetchInsiderData = async () => {
      try {
        const data = await getInsiderTrades();
        // Sort by Date (Desc), then Value (Desc), then Transaction (Asc)
        const sortedData = data.sort((a, b) => {
          // 1. Date
          const dateA = new Date(a.date).getTime();
          const dateB = new Date(b.date).getTime();
          if (dateA !== dateB) return dateB - dateA;

          // 2. Value
          if (a.value !== b.value) return b.value - a.value;

          // 3. Transaction Type
          return a.transaction.localeCompare(b.transaction);
        });
        setTrades(sortedData);
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
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <h3 style={{ marginTop: 0, marginBottom: '1rem', color: '#fff', flexShrink: 0 }}>Insider Activity</h3>
      <div style={{ flex: 1, overflowY: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85em' }}>
          <thead style={{ position: 'sticky', top: 0, background: '#0f172a', zIndex: 10 }}>
            <tr style={{ textAlign: 'left', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
              <th style={{ color: '#a1a1aa', padding: '8px 4px' }}>Ticker</th>
              <th style={{ color: '#a1a1aa', padding: '8px 4px' }}>Owner</th>
              <th style={{ color: '#a1a1aa', padding: '8px 4px' }}>Trans</th>
              <th style={{ color: '#a1a1aa', padding: '8px 4px' }}>Value</th>
              <th style={{ color: '#a1a1aa', padding: '8px 4px' }}>Date</th>
            </tr>
          </thead>
          <tbody>
            {trades.map((trade) => (
              <tr key={trade.id} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                <td style={{ padding: '8px 4px', fontWeight: 'bold', color: '#fff' }}>{trade.ticker}</td>
                <td style={{ maxWidth: '80px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', padding: '8px 4px', color: '#a1a1aa' }} title={trade.owner}>
                  {trade.owner}
                </td>
                <td style={{ 
                  padding: '8px 4px',
                  color: trade.transaction.toLowerCase().includes('buy') ? '#10b981' : '#ef4444',
                  fontWeight: 'bold' 
                }}>
                  {trade.transaction}
                </td>
                <td style={{ padding: '8px 4px', color: '#fff' }}>
                  {new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0, notation: "compact" }).format(trade.value)}
                </td>
                <td style={{ padding: '8px 4px', color: '#a1a1aa', whiteSpace: 'nowrap' }}>{new Date(trade.date).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default InsiderWidget;
