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

  // Helper to parse percentage string like "1.50%" or "-0.23%"
  const getChangeValue = (changeStr: string) => {
    return parseFloat(changeStr.replace('%', ''));
  };

  // Helper to get color based on change
  const getColor = (change: number) => {
    if (change > 2) return '#059669'; // Emerald-600
    if (change > 0) return '#10b981'; // Emerald-500
    if (change === 0) return '#52525b'; // Zinc-600
    if (change > -2) return '#ef4444'; // Red-500
    return '#b91c1c'; // Red-700
  };

  return (
    <div style={{ height: '100%', overflowY: 'auto' }}>
      <h3 style={{ marginTop: 0, marginBottom: '1rem', color: '#fff' }}>Sector Performance</h3>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(120px, 1fr))', gap: '10px' }}>
        {sectors.map((sector) => {
          const changeVal = getChangeValue(sector.change);
          const color = getColor(changeVal);
          
          return (
            <div 
              key={sector.id}
              style={{
                backgroundColor: color,
                color: '#fff',
                padding: '12px',
                borderRadius: '8px',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
                alignItems: 'center',
                textAlign: 'center',
                height: '80px',
                transition: 'transform 0.2s',
                cursor: 'default'
              }}
              onMouseEnter={(e) => e.currentTarget.style.transform = 'scale(1.05)'}
              onMouseLeave={(e) => e.currentTarget.style.transform = 'scale(1)'}
            >
              <div style={{ fontSize: '0.8rem', fontWeight: 500, marginBottom: '4px' }}>{sector.name}</div>
              <div style={{ fontSize: '1.1rem', fontWeight: 'bold' }}>{sector.change}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default SectorHeatmap;
