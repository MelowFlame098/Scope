import React, { useEffect, useState } from 'react';
import { getSectorPerformance, SectorPerformance } from '../api/client';

const SectorHeatmap: React.FC = () => {
  const [sectors, setSectors] = useState<SectorPerformance[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    const fetchSectorData = async () => {
      try {
        const data = await getSectorPerformance();
        setSectors(data);
      } catch (error) {
        console.error("Failed to fetch sector performance", error);
      } finally {
        setLoading(false);
      }
    };

    fetchSectorData();
  }, []);

  if (loading) {
    return <div style={{ color: '#a1a1aa' }}>Loading Sectors...</div>;
  }

  // Helper to get color based on change
  const getColor = (change: number) => {
    if (change > 2) return '#059669'; // Emerald-600
    if (change > 0) return '#10b981'; // Emerald-500
    if (change === 0) return '#52525b'; // Zinc-600
    if (change > -2) return '#ef4444'; // Red-500
    return '#b91c1c'; // Red-700
  };

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <h3 style={{ marginTop: 0, marginBottom: '1rem', color: '#fff', flexShrink: 0 }}>Sector Performance</h3>
      <div style={{ flex: 1, overflowY: 'auto' }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(110px, 1fr))', gap: '8px' }}>
          {sectors.map((sector) => {
            const changeVal = sector.change;
            const color = getColor(changeVal);
            
            return (
              <div 
                key={sector.id}
                style={{
                  backgroundColor: color,
                  color: '#fff',
                  padding: '10px',
                  borderRadius: '6px',
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'center',
                  alignItems: 'center',
                  textAlign: 'center',
                  height: '70px',
                  transition: 'transform 0.2s',
                  cursor: 'default'
                }}
                onMouseEnter={(e) => e.currentTarget.style.transform = 'scale(1.05)'}
                onMouseLeave={(e) => e.currentTarget.style.transform = 'scale(1)'}
              >
                <div style={{ fontSize: '0.75rem', fontWeight: 500, marginBottom: '2px', lineHeight: '1.2' }}>{sector.name}</div>
                <div style={{ fontSize: '1rem', fontWeight: 'bold' }}>{changeVal.toFixed(2)}%</div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default SectorHeatmap;
