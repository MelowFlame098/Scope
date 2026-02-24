import React, { useState, useEffect } from 'react';
import ReactApexChart from 'react-apexcharts';
import { ApexOptions } from 'apexcharts';
import { getStockCandles, Candle } from '../api/client';

interface StockChartProps {
  symbol: string;
  data?: any[]; // Keep for compatibility but ignore
}

const timeframes = [
  '1m', '5m', '15m', '30m', '1h', '12h', '24h', 
  '3d', '1w', '1mo', '3mo', '6mo', '1y', 'ytd'
];

const StockChart: React.FC<StockChartProps> = ({ symbol }) => {
  const [series, setSeries] = useState([{ data: [] as any[] }]);
  const [timeframe, setTimeframe] = useState('5m');

  useEffect(() => {
    const fetchCandles = async () => {
      try {
        const candles = await getStockCandles(symbol, timeframe);
        const data = candles.map((c: Candle) => ({
          x: new Date(c.timestamp),
          y: [
            c.open ? Number(Number(c.open).toFixed(2)) : null,
            c.high ? Number(Number(c.high).toFixed(2)) : null,
            c.low ? Number(Number(c.low).toFixed(2)) : null,
            c.close ? Number(Number(c.close).toFixed(2)) : null
          ]
        }));
        setSeries([{ data }]);
      } catch (error) {
        console.error('Failed to fetch candles', error);
      }
    };

    fetchCandles();
    // Poll every 5s to update the current candle
    const interval = setInterval(fetchCandles, 5000);

    return () => clearInterval(interval);
  }, [symbol, timeframe]);

  const options: ApexOptions = {
    chart: {
      type: 'candlestick',
      height: 350,
      background: 'transparent',
      toolbar: {
        show: false
      },
      animations: {
        enabled: false // Disable animations for smoother updates
      }
    },
    title: {
      text: undefined,
      align: 'left',
      style: {
        color: '#fff'
      }
    },
    xaxis: {
      type: 'datetime',
      labels: {
        style: {
          colors: '#a1a1aa'
        }
      },
      axisBorder: {
        show: false
      },
      axisTicks: {
        show: false
      }
    },
    tooltip: {
      theme: 'dark',
      shared: true,
      intersect: false,
      y: {
        formatter: (val: number) => {
          if (typeof val === 'number') {
            return val.toFixed(2);
          }
          return val;
        }
      },
      // Explicitly format tooltip items
      x: {
        format: 'dd MMM yyyy HH:mm'
      }
    },
    yaxis: {
      tooltip: {
        enabled: true
      },
      labels: {
        style: {
          colors: '#a1a1aa'
        },
        formatter: (val) => val.toFixed(2)
      }
    },
    grid: {
      borderColor: 'rgba(255,255,255,0.1)'
    },
    plotOptions: {
      candlestick: {
        colors: {
          upward: '#10b981',
          downward: '#ef4444'
        }
      }
    },
    theme: {
      mode: 'dark'
    }
  };

  return (
    <div style={{ width: '100%', height: '100%' }}>
      <div style={{ marginBottom: '15px', display: 'flex', gap: '5px', flexWrap: 'wrap' }}>
        {timeframes.map((tf) => (
          <button
            key={tf}
            onClick={() => setTimeframe(tf)}
            style={{
              padding: '4px 8px',
              backgroundColor: timeframe === tf ? '#8b5cf6' : 'rgba(255,255,255,0.1)',
              border: '1px solid ' + (timeframe === tf ? '#8b5cf6' : 'rgba(255,255,255,0.2)'),
              borderRadius: '4px',
              color: '#fff',
              cursor: 'pointer',
              fontSize: '11px',
              transition: 'all 0.2s'
            }}
          >
            {tf}
          </button>
        ))}
      </div>
      <div style={{ height: '350px' }}>
        <ReactApexChart options={options} series={series} type="candlestick" height="100%" />
      </div>
    </div>
  );
};

export default StockChart;
