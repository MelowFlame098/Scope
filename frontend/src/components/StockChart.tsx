import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

interface PricePoint {
  timestamp: string;
  price: number;
}

interface StockChartProps {
  data: PricePoint[];
  symbol: string;
}

const StockChart: React.FC<StockChartProps> = ({ data, symbol }) => {
  return (
    <div style={{ width: '100%', height: 300 }}>
      <h3>{symbol} Price</h3>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
        <XAxis 
          dataKey="timestamp" 
          stroke="#a1a1aa" 
          tick={{ fill: '#a1a1aa', fontSize: 12 }}
          tickLine={{ stroke: '#a1a1aa' }}
        />
        <YAxis 
          domain={['auto', 'auto']} 
          stroke="#a1a1aa"
          tick={{ fill: '#a1a1aa', fontSize: 12 }}
          tickLine={{ stroke: '#a1a1aa' }}
        />
        <Tooltip 
          contentStyle={{ 
            backgroundColor: '#18181b', 
            border: '1px solid #3f3f46',
            borderRadius: '8px',
            color: '#fff'
          }}
          itemStyle={{ color: '#fff' }}
          labelStyle={{ color: '#a1a1aa' }}
        />
        <Legend />
        <Line 
          type="monotone" 
          dataKey="price" 
          stroke="#8b5cf6" 
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 6, fill: '#8b5cf6', stroke: '#fff' }} 
        />
      </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default StockChart;
