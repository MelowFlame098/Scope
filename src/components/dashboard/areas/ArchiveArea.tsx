"use client";

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  ArchiveBoxIcon,
  DocumentTextIcon,
  ChartBarIcon,
  CalendarIcon,
  MagnifyingGlassIcon,
  PlusIcon,
  EyeIcon,
  PencilIcon,
  TrashIcon,
  ArrowDownTrayIcon,
  ShareIcon,
  TagIcon,
  ClockIcon
} from '@heroicons/react/24/outline';

interface ArchiveAreaProps {
  user: any;
}

interface JournalEntry {
  id: string;
  title: string;
  content: string;
  date: Date;
  tags: string[];
  trades: number;
  pnl: number;
  mood: 'bullish' | 'bearish' | 'neutral';
  market: string;
  lessons: string[];
}

interface Report {
  id: string;
  title: string;
  type: 'daily' | 'weekly' | 'monthly' | 'custom';
  period: string;
  generated: Date;
  trades: number;
  winRate: number;
  pnl: number;
  sharpeRatio: number;
  maxDrawdown: number;
  status: 'completed' | 'generating' | 'failed';
}

interface TradeRecord {
  id: string;
  symbol: string;
  action: 'BUY' | 'SELL';
  quantity: number;
  entryPrice: number;
  exitPrice?: number;
  entryDate: Date;
  exitDate?: Date;
  pnl?: number;
  status: 'open' | 'closed';
  strategy: string;
  notes: string;
}

export const ArchiveArea: React.FC<ArchiveAreaProps> = ({ user }) => {
  const [activeTab, setActiveTab] = useState('journal');
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedPeriod, setSelectedPeriod] = useState('all');

  const [journalEntries, setJournalEntries] = useState<JournalEntry[]>([
    {
      id: '1',
      title: 'Strong Tech Rally - NVDA Breakout',
      content: 'Today saw a significant breakout in NVDA above $870 resistance. The volume was exceptional, and the AI narrative continues to drive momentum. Entered position at $875 with tight stop at $850. Market sentiment is extremely bullish on AI stocks.',
      date: new Date(Date.now() - 86400000),
      tags: ['NVDA', 'AI', 'Breakout', 'Tech'],
      trades: 3,
      pnl: 2450,
      mood: 'bullish',
      market: 'Tech Rally',
      lessons: ['Volume confirmation is crucial', 'AI narrative still strong', 'Tight stops in volatile markets']
    },
    {
      id: '2',
      title: 'Market Correction - Defensive Positioning',
      content: 'Fed hawkish comments triggered a broad market selloff. Reduced position sizes and moved to defensive stocks. AAPL held up well, but growth stocks got hammered. Good reminder to respect risk management.',
      date: new Date(Date.now() - 172800000),
      tags: ['Fed', 'Correction', 'Risk Management', 'Defensive'],
      trades: 5,
      pnl: -890,
      mood: 'bearish',
      market: 'Correction',
      lessons: ['Fed policy matters', 'Defensive positioning in uncertainty', 'Position sizing is key']
    },
    {
      id: '3',
      title: 'Earnings Season Strategy',
      content: 'Preparing for earnings season with a focus on companies with strong guidance. MSFT and GOOGL looking strong. Using options strategies to limit downside while maintaining upside exposure.',
      date: new Date(Date.now() - 259200000),
      tags: ['Earnings', 'MSFT', 'GOOGL', 'Options'],
      trades: 7,
      pnl: 1650,
      mood: 'neutral',
      market: 'Earnings Season',
      lessons: ['Guidance matters more than beats', 'Options for risk management', 'Sector rotation during earnings']
    }
  ]);

  const [reports, setReports] = useState<Report[]>([
    {
      id: '1',
      title: 'December 2024 Performance Report',
      type: 'monthly',
      period: 'Dec 2024',
      generated: new Date(Date.now() - 86400000),
      trades: 45,
      winRate: 68,
      pnl: 12450,
      sharpeRatio: 1.85,
      maxDrawdown: -5.2,
      status: 'completed'
    },
    {
      id: '2',
      title: 'Week of Dec 16-20, 2024',
      type: 'weekly',
      period: 'Dec 16-20, 2024',
      generated: new Date(Date.now() - 172800000),
      trades: 12,
      winRate: 75,
      pnl: 3200,
      sharpeRatio: 2.1,
      maxDrawdown: -2.1,
      status: 'completed'
    },
    {
      id: '3',
      title: 'Q4 2024 Quarterly Review',
      type: 'custom',
      period: 'Q4 2024',
      generated: new Date(Date.now() - 259200000),
      trades: 156,
      winRate: 64,
      pnl: 28750,
      sharpeRatio: 1.65,
      maxDrawdown: -8.5,
      status: 'generating'
    }
  ]);

  const [tradeRecords, setTradeRecords] = useState<TradeRecord[]>([
    {
      id: '1',
      symbol: 'NVDA',
      action: 'BUY',
      quantity: 100,
      entryPrice: 875.50,
      exitPrice: 890.25,
      entryDate: new Date(Date.now() - 86400000),
      exitDate: new Date(Date.now() - 43200000),
      pnl: 1475,
      status: 'closed',
      strategy: 'Breakout',
      notes: 'Clean breakout above resistance with volume'
    },
    {
      id: '2',
      symbol: 'AAPL',
      action: 'BUY',
      quantity: 200,
      entryPrice: 185.90,
      entryDate: new Date(Date.now() - 172800000),
      status: 'open',
      strategy: 'Support Bounce',
      notes: 'Holding above key support level'
    },
    {
      id: '3',
      symbol: 'TSLA',
      action: 'SELL',
      quantity: 50,
      entryPrice: 240.50,
      exitPrice: 235.80,
      entryDate: new Date(Date.now() - 259200000),
      exitDate: new Date(Date.now() - 216000000),
      pnl: -235,
      status: 'closed',
      strategy: 'Short Squeeze',
      notes: 'Stopped out on momentum shift'
    }
  ]);

  const filteredJournalEntries = journalEntries.filter(entry =>
    entry.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
    entry.content.toLowerCase().includes(searchTerm.toLowerCase()) ||
    entry.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase()))
  );

  const filteredReports = reports.filter(report =>
    report.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
    report.type.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const filteredTrades = tradeRecords.filter(trade =>
    trade.symbol.toLowerCase().includes(searchTerm.toLowerCase()) ||
    trade.strategy.toLowerCase().includes(searchTerm.toLowerCase()) ||
    trade.notes.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const formatDate = (date: Date) => {
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  };

  const formatCurrency = (amount: number) => {
    const sign = amount >= 0 ? '+' : '';
    return `${sign}$${amount.toLocaleString()}`;
  };

  const getPnlColor = (pnl: number) => {
    return pnl >= 0 ? 'text-green-400' : 'text-red-400';
  };

  const getMoodColor = (mood: string) => {
    switch (mood) {
      case 'bullish': return 'bg-green-600';
      case 'bearish': return 'bg-red-600';
      case 'neutral': return 'bg-yellow-600';
      default: return 'bg-gray-600';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'bg-green-600';
      case 'generating': return 'bg-yellow-600';
      case 'failed': return 'bg-red-600';
      default: return 'bg-gray-600';
    }
  };

  return (
    <div className="h-full flex flex-col">
      <Tabs value={activeTab} onValueChange={setActiveTab} className="h-full">
        <TabsList className="grid w-full grid-cols-4 bg-gray-800">
          <TabsTrigger value="journal">Journal</TabsTrigger>
          <TabsTrigger value="reports">Reports</TabsTrigger>
          <TabsTrigger value="trades">Trade History</TabsTrigger>
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
        </TabsList>

        {/* Journal Tab */}
        <TabsContent value="journal" className="flex-1 mt-4">
          <div className="space-y-4 h-full">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className="relative">
                  <MagnifyingGlassIcon className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                  <Input
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    placeholder="Search journal entries..."
                    className="pl-10 bg-gray-700 border-gray-600 text-white w-64"
                  />
                </div>
              </div>
              <Button className="bg-blue-600 hover:bg-blue-700">
                <PlusIcon className="w-4 h-4 mr-2" />
                New Entry
              </Button>
            </div>

            <div className="space-y-4 overflow-y-auto">
              {filteredJournalEntries.map((entry) => (
                <Card key={entry.id} className="bg-gray-800 border-gray-700">
                  <CardHeader className="pb-3">
                    <div className="flex items-start justify-between">
                      <div>
                        <CardTitle className="text-white text-lg mb-2">{entry.title}</CardTitle>
                        <div className="flex items-center space-x-4 text-sm text-gray-400">
                          <div className="flex items-center">
                            <CalendarIcon className="w-4 h-4 mr-1" />
                            {formatDate(entry.date)}
                          </div>
                          <div className="flex items-center">
                            <ChartBarIcon className="w-4 h-4 mr-1" />
                            {entry.trades} trades
                          </div>
                          <div className={`flex items-center font-medium ${getPnlColor(entry.pnl)}`}>
                            {formatCurrency(entry.pnl)}
                          </div>
                          <Badge className={getMoodColor(entry.mood)}>
                            {entry.mood}
                          </Badge>
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Button variant="ghost" size="sm">
                          <EyeIcon className="w-4 h-4" />
                        </Button>
                        <Button variant="ghost" size="sm">
                          <PencilIcon className="w-4 h-4" />
                        </Button>
                        <Button variant="ghost" size="sm">
                          <TrashIcon className="w-4 h-4" />
                        </Button>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent className="pt-0">
                    <p className="text-gray-300 mb-4 line-clamp-3">{entry.content}</p>
                    
                    <div className="flex flex-wrap gap-2 mb-4">
                      {entry.tags.map((tag, index) => (
                        <Badge key={index} variant="outline" className="text-xs">
                          <TagIcon className="w-3 h-3 mr-1" />
                          {tag}
                        </Badge>
                      ))}
                    </div>

                    {entry.lessons.length > 0 && (
                      <div className="bg-gray-700/50 rounded-lg p-3">
                        <h4 className="text-white font-medium mb-2">Key Lessons:</h4>
                        <ul className="text-gray-300 text-sm space-y-1">
                          {entry.lessons.map((lesson, index) => (
                            <li key={index} className="flex items-start">
                              <span className="text-blue-400 mr-2">•</span>
                              {lesson}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        </TabsContent>

        {/* Reports Tab */}
        <TabsContent value="reports" className="flex-1 mt-4">
          <div className="space-y-4 h-full">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className="relative">
                  <MagnifyingGlassIcon className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                  <Input
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    placeholder="Search reports..."
                    className="pl-10 bg-gray-700 border-gray-600 text-white w-64"
                  />
                </div>
              </div>
              <Button className="bg-blue-600 hover:bg-blue-700">
                <PlusIcon className="w-4 h-4 mr-2" />
                Generate Report
              </Button>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 overflow-y-auto">
              {filteredReports.map((report) => (
                <Card key={report.id} className="bg-gray-800 border-gray-700">
                  <CardHeader className="pb-3">
                    <div className="flex items-start justify-between">
                      <div>
                        <CardTitle className="text-white text-lg mb-2">{report.title}</CardTitle>
                        <div className="flex items-center space-x-3 text-sm text-gray-400">
                          <Badge className={getStatusColor(report.status)}>
                            {report.status}
                          </Badge>
                          <span>{report.period}</span>
                          <div className="flex items-center">
                            <ClockIcon className="w-4 h-4 mr-1" />
                            {formatDate(report.generated)}
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Button variant="ghost" size="sm">
                          <EyeIcon className="w-4 h-4" />
                        </Button>
                        <Button variant="ghost" size="sm">
                          <ArrowDownTrayIcon className="w-4 h-4" />
                        </Button>
                        <Button variant="ghost" size="sm">
                          <ShareIcon className="w-4 h-4" />
                        </Button>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent className="pt-0">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-3">
                        <div>
                          <div className="text-gray-400 text-xs">Total Trades</div>
                          <div className="text-white font-medium">{report.trades}</div>
                        </div>
                        <div>
                          <div className="text-gray-400 text-xs">Win Rate</div>
                          <div className="text-green-400 font-medium">{report.winRate}%</div>
                        </div>
                        <div>
                          <div className="text-gray-400 text-xs">Sharpe Ratio</div>
                          <div className="text-blue-400 font-medium">{report.sharpeRatio}</div>
                        </div>
                      </div>
                      <div className="space-y-3">
                        <div>
                          <div className="text-gray-400 text-xs">P&L</div>
                          <div className={`font-medium ${getPnlColor(report.pnl)}`}>
                            {formatCurrency(report.pnl)}
                          </div>
                        </div>
                        <div>
                          <div className="text-gray-400 text-xs">Max Drawdown</div>
                          <div className="text-red-400 font-medium">{report.maxDrawdown}%</div>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        </TabsContent>

        {/* Trade History Tab */}
        <TabsContent value="trades" className="flex-1 mt-4">
          <div className="space-y-4 h-full">
            <div className="flex items-center justify-between">
              <div className="relative">
                <MagnifyingGlassIcon className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                <Input
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  placeholder="Search trades..."
                  className="pl-10 bg-gray-700 border-gray-600 text-white w-64"
                />
              </div>
              <Button className="bg-blue-600 hover:bg-blue-700">
                <ArrowDownTrayIcon className="w-4 h-4 mr-2" />
                Export CSV
              </Button>
            </div>

            <Card className="bg-gray-800 border-gray-700 h-full">
              <CardContent className="p-0">
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead className="bg-gray-700/50">
                      <tr>
                        <th className="text-left p-4 text-gray-300 font-medium">Symbol</th>
                        <th className="text-left p-4 text-gray-300 font-medium">Action</th>
                        <th className="text-right p-4 text-gray-300 font-medium">Quantity</th>
                        <th className="text-right p-4 text-gray-300 font-medium">Entry</th>
                        <th className="text-right p-4 text-gray-300 font-medium">Exit</th>
                        <th className="text-right p-4 text-gray-300 font-medium">P&L</th>
                        <th className="text-left p-4 text-gray-300 font-medium">Strategy</th>
                        <th className="text-left p-4 text-gray-300 font-medium">Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredTrades.map((trade) => (
                        <tr key={trade.id} className="border-b border-gray-700/50 hover:bg-gray-700/25">
                          <td className="p-4 text-white font-medium">{trade.symbol}</td>
                          <td className="p-4">
                            <Badge className={trade.action === 'BUY' ? 'bg-green-600' : 'bg-red-600'}>
                              {trade.action}
                            </Badge>
                          </td>
                          <td className="p-4 text-right text-white">{trade.quantity}</td>
                          <td className="p-4 text-right text-white font-mono">
                            ${trade.entryPrice}
                          </td>
                          <td className="p-4 text-right text-white font-mono">
                            {trade.exitPrice ? `$${trade.exitPrice}` : '-'}
                          </td>
                          <td className={`p-4 text-right font-medium ${
                            trade.pnl ? getPnlColor(trade.pnl) : 'text-gray-400'
                          }`}>
                            {trade.pnl ? formatCurrency(trade.pnl) : '-'}
                          </td>
                          <td className="p-4 text-gray-300">{trade.strategy}</td>
                          <td className="p-4">
                            <Badge className={trade.status === 'open' ? 'bg-blue-600' : 'bg-gray-600'}>
                              {trade.status}
                            </Badge>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Analytics Tab */}
        <TabsContent value="analytics" className="flex-1 mt-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 h-full">
            <Card className="bg-gray-800 border-gray-700">
              <CardHeader>
                <CardTitle className="text-white flex items-center">
                  <ChartBarIcon className="w-5 h-5 mr-2" />
                  Performance Metrics
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-300">Total Trades</span>
                    <span className="text-white font-medium">156</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-300">Win Rate</span>
                    <span className="text-green-400 font-medium">68%</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-300">Total P&L</span>
                    <span className="text-green-400 font-medium">+$28,750</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-300">Best Trade</span>
                    <span className="text-green-400 font-medium">+$2,450</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-300">Worst Trade</span>
                    <span className="text-red-400 font-medium">-$890</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-300">Avg Trade</span>
                    <span className="text-white font-medium">+$184</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-300">Sharpe Ratio</span>
                    <span className="text-blue-400 font-medium">1.75</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-300">Max Drawdown</span>
                    <span className="text-red-400 font-medium">-8.5%</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-gray-800 border-gray-700">
              <CardHeader>
                <CardTitle className="text-white">Monthly Performance</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {['Dec 2024', 'Nov 2024', 'Oct 2024', 'Sep 2024'].map((month, index) => {
                    const pnl = [12450, 8750, -2100, 5600][index];
                    return (
                      <div key={month} className="flex justify-between items-center p-3 bg-gray-700/50 rounded-lg">
                        <span className="text-white font-medium">{month}</span>
                        <span className={`font-medium ${getPnlColor(pnl)}`}>
                          {formatCurrency(pnl)}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default ArchiveArea;