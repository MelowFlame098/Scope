"use client";

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { 
  ArrowUpIcon,
  ArrowDownIcon,
  ChartBarIcon
} from '@heroicons/react/24/outline';

interface OrderBookEntry {
  price: number;
  size: number;
  total: number;
}

interface OrderBookProps {
  symbol: string;
  onPriceSelect?: (price: number) => void;
}

export const OrderBook: React.FC<OrderBookProps> = ({ symbol, onPriceSelect }) => {
  const [bids, setBids] = useState<OrderBookEntry[]>([]);
  const [asks, setAsks] = useState<OrderBookEntry[]>([]);
  const [spread, setSpread] = useState(0);
  const [spreadPercent, setSpreadPercent] = useState(0);
  const [lastPrice, setLastPrice] = useState(0);

  // Mock data generation
  useEffect(() => {
    const generateOrderBook = () => {
      const basePrice = 155.50;
      setLastPrice(basePrice);

      // Generate bids (buy orders) - prices below current price
      const mockBids: OrderBookEntry[] = [];
      let totalBids = 0;
      for (let i = 0; i < 15; i++) {
        const price = basePrice - (i + 1) * 0.01;
        const size = Math.floor(Math.random() * 1000) + 100;
        totalBids += size;
        mockBids.push({ price, size, total: totalBids });
      }

      // Generate asks (sell orders) - prices above current price
      const mockAsks: OrderBookEntry[] = [];
      let totalAsks = 0;
      for (let i = 0; i < 15; i++) {
        const price = basePrice + (i + 1) * 0.01;
        const size = Math.floor(Math.random() * 1000) + 100;
        totalAsks += size;
        mockAsks.push({ price, size, total: totalAsks });
      }

      setBids(mockBids);
      setAsks(mockAsks.reverse()); // Reverse to show highest ask first

      // Calculate spread
      const bestBid = mockBids[0]?.price || 0;
      const bestAsk = mockAsks[mockAsks.length - 1]?.price || 0;
      const currentSpread = bestAsk - bestBid;
      setSpread(currentSpread);
      setSpreadPercent((currentSpread / bestBid) * 100);
    };

    generateOrderBook();
    const interval = setInterval(generateOrderBook, 2000);
    return () => clearInterval(interval);
  }, [symbol]);

  const formatPrice = (price: number) => price.toFixed(2);
  const formatSize = (size: number) => size.toLocaleString();

  const getVolumeBarWidth = (total: number, maxTotal: number) => {
    return Math.max((total / maxTotal) * 100, 5);
  };

  const maxBidTotal = Math.max(...bids.map(b => b.total));
  const maxAskTotal = Math.max(...asks.map(a => a.total));

  return (
    <Card className="bg-gray-800 border-gray-700 h-full">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-white text-lg flex items-center">
            <ChartBarIcon className="w-5 h-5 mr-2" />
            Order Book - {symbol}
          </CardTitle>
          <Badge variant="outline" className="text-gray-300">
            Live
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="p-0">
        {/* Spread Information */}
        <div className="px-4 py-2 bg-gray-750 border-b border-gray-600">
          <div className="flex justify-between items-center text-sm">
            <span className="text-gray-400">Spread</span>
            <div className="flex items-center space-x-2">
              <span className="text-white">${spread.toFixed(2)}</span>
              <span className="text-gray-400">({spreadPercent.toFixed(3)}%)</span>
            </div>
          </div>
        </div>

        {/* Order Book Headers */}
        <div className="px-4 py-2 bg-gray-750 border-b border-gray-600">
          <div className="grid grid-cols-3 text-xs text-gray-400 font-medium">
            <span>Price</span>
            <span className="text-right">Size</span>
            <span className="text-right">Total</span>
          </div>
        </div>

        <div className="max-h-96 overflow-y-auto">
          {/* Asks (Sell Orders) */}
          <div className="border-b border-gray-600">
            {asks.map((ask, index) => (
              <div
                key={`ask-${index}`}
                className="relative px-4 py-1 hover:bg-gray-700 cursor-pointer group"
                onClick={() => onPriceSelect?.(ask.price)}
              >
                {/* Volume Bar */}
                <div
                  className="absolute right-0 top-0 h-full bg-red-500/10 transition-all duration-300"
                  style={{ width: `${getVolumeBarWidth(ask.total, maxAskTotal)}%` }}
                />
                
                <div className="relative grid grid-cols-3 text-sm">
                  <span className="text-red-400 font-mono">
                    {formatPrice(ask.price)}
                  </span>
                  <span className="text-right text-gray-300 font-mono">
                    {formatSize(ask.size)}
                  </span>
                  <span className="text-right text-gray-400 font-mono">
                    {formatSize(ask.total)}
                  </span>
                </div>

                {/* Hover indicator */}
                <div className="absolute left-0 top-0 w-1 h-full bg-red-400 opacity-0 group-hover:opacity-100 transition-opacity" />
              </div>
            ))}
          </div>

          {/* Current Price */}
          <div className="px-4 py-3 bg-gray-750 border-b border-gray-600">
            <div className="flex items-center justify-center space-x-2">
              <ArrowUpIcon className="w-4 h-4 text-green-400" />
              <span className="text-lg font-bold text-white font-mono">
                ${formatPrice(lastPrice)}
              </span>
              <ArrowDownIcon className="w-4 h-4 text-red-400" />
            </div>
            <div className="text-center text-xs text-gray-400 mt-1">
              Last Price
            </div>
          </div>

          {/* Bids (Buy Orders) */}
          <div>
            {bids.map((bid, index) => (
              <div
                key={`bid-${index}`}
                className="relative px-4 py-1 hover:bg-gray-700 cursor-pointer group"
                onClick={() => onPriceSelect?.(bid.price)}
              >
                {/* Volume Bar */}
                <div
                  className="absolute right-0 top-0 h-full bg-green-500/10 transition-all duration-300"
                  style={{ width: `${getVolumeBarWidth(bid.total, maxBidTotal)}%` }}
                />
                
                <div className="relative grid grid-cols-3 text-sm">
                  <span className="text-green-400 font-mono">
                    {formatPrice(bid.price)}
                  </span>
                  <span className="text-right text-gray-300 font-mono">
                    {formatSize(bid.size)}
                  </span>
                  <span className="text-right text-gray-400 font-mono">
                    {formatSize(bid.total)}
                  </span>
                </div>

                {/* Hover indicator */}
                <div className="absolute left-0 top-0 w-1 h-full bg-green-400 opacity-0 group-hover:opacity-100 transition-opacity" />
              </div>
            ))}
          </div>
        </div>

        {/* Order Book Actions */}
        <div className="px-4 py-3 bg-gray-750 border-t border-gray-600">
          <div className="flex space-x-2">
            <Button size="sm" variant="outline" className="flex-1 text-xs">
              Zoom In
            </Button>
            <Button size="sm" variant="outline" className="flex-1 text-xs">
              Zoom Out
            </Button>
            <Button size="sm" variant="outline" className="flex-1 text-xs">
              Reset
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};