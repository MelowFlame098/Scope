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
    return <div>Loading Sectors...</div>;
  }

  // Helper to parse percentage string like "1.50%" or "-0.23%"
  const getChangeValue = (changeStr: string) => {
    return parseFloat(changeStr.replace('%', ''));
  };

  // Helper to get color based on change
  const getColor = (change: number) => {
    if (change > 2) return '#006400'; // Dark Green
    if (change > 0) return '#32CD32'; // Lime Green
    if (change === 0) return '#808080'; // Grey
    if (change > -2) return '#FF6347'; // Tomato
    return '#8B0000'; // Dark Red
  };

  return (
    <div style={{ border: '1px solid #ccc', padding: '15px', borderRadius: '8px', height: '300px', overflowY: 'auto' }}>
      <h3>Sector Performance</h3>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(120px, 1fr))', gap: '10px' }}>
        {sectors.map((sector) => {
          const changeVal = getChangeValue(sector.change);
          const color = getColor(changeVal);
          
          return (
            <div 
              key={sector.id} 
              style={{ 
                backgroundColor: color, 
                color: 'white', 
                padding: '10px', 
                borderRadius: '4px',
                textAlign: 'center',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
                minHeight: '80px'
              }}
            >
              <div style={{ fontWeight: 'bold', fontSize: '0.9em', marginBottom: '5px' }}>{sector.name}</div>
              <div style={{ fontSize: '1.1em' }}>{sector.change}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default SectorHeatmap;
