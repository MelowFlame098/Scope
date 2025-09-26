"use client";

import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { 
  Play, 
  Pause, 
  Square, 
  RotateCcw, 
  Settings, 
  Activity,
  AlertCircle,
  CheckCircle,
  Clock,
  Zap,
  Database,
  GitBranch,
  Monitor,
  TrendingUp,
  TrendingDown,
  BarChart3,
  RefreshCw,
  Download,
  Upload,
  Eye,
  Edit,
  Trash2,
  Plus,
  X
} from 'lucide-react';

interface Pipeline {
  id: string;
  name: string;
  description: string;
  status: 'running' | 'stopped' | 'paused' | 'error' | 'completed';
  type: 'data-ingestion' | 'analysis' | 'trading' | 'reporting' | 'monitoring';
  lastRun: string;
  nextRun?: string;
  duration: string;
  successRate: number;
  priority: 'low' | 'medium' | 'high' | 'critical';
}

interface PipelineMetrics {
  totalPipelines: number;
  runningPipelines: number;
  successfulRuns: number;
  failedRuns: number;
  avgExecutionTime: string;
  dataProcessed: string;
}

const PipelineControlPanel: React.FC = () => {
  const [selectedPipeline, setSelectedPipeline] = useState<string | null>(null);
  const [filterStatus, setFilterStatus] = useState('all');
  const [filterType, setFilterType] = useState('all');

  // Mock pipeline data
  const pipelines: Pipeline[] = [
    {
      id: '1',
      name: 'Market Data Ingestion',
      description: 'Real-time market data collection and processing',
      status: 'running',
      type: 'data-ingestion',
      lastRun: '2 minutes ago',
      nextRun: 'Continuous',
      duration: '24/7',
      successRate: 99.8,
      priority: 'critical'
    },
    {
      id: '2',
      name: 'Technical Analysis Engine',
      description: 'Automated technical indicator calculations',
      status: 'running',
      type: 'analysis',
      lastRun: '5 minutes ago',
      nextRun: 'Every 5 minutes',
      duration: '2.3s',
      successRate: 97.5,
      priority: 'high'
    },
    {
      id: '3',
      name: 'Risk Assessment Pipeline',
      description: 'Portfolio risk analysis and monitoring',
      status: 'completed',
      type: 'analysis',
      lastRun: '15 minutes ago',
      nextRun: 'Every 30 minutes',
      duration: '45s',
      successRate: 98.2,
      priority: 'high'
    },
    {
      id: '4',
      name: 'Automated Trading Bot',
      description: 'Execute trades based on predefined strategies',
      status: 'paused',
      type: 'trading',
      lastRun: '1 hour ago',
      nextRun: 'Manual',
      duration: '1.2s',
      successRate: 94.7,
      priority: 'critical'
    },
    {
      id: '5',
      name: 'Daily Report Generator',
      description: 'Generate daily performance and analytics reports',
      status: 'error',
      type: 'reporting',
      lastRun: '2 hours ago',
      nextRun: 'Daily at 6 PM',
      duration: '12s',
      successRate: 89.3,
      priority: 'medium'
    },
    {
      id: '6',
      name: 'System Health Monitor',
      description: 'Monitor system performance and alerts',
      status: 'running',
      type: 'monitoring',
      lastRun: '30 seconds ago',
      nextRun: 'Every minute',
      duration: '0.8s',
      successRate: 99.9,
      priority: 'high'
    }
  ];

  const metrics: PipelineMetrics = {
    totalPipelines: pipelines.length,
    runningPipelines: pipelines.filter(p => p.status === 'running').length,
    successfulRuns: 1247,
    failedRuns: 23,
    avgExecutionTime: '3.2s',
    dataProcessed: '2.4 TB'
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running':
        return <Play className="h-4 w-4 text-green-500" />;
      case 'paused':
        return <Pause className="h-4 w-4 text-yellow-500" />;
      case 'stopped':
        return <Square className="h-4 w-4 text-gray-500" />;
      case 'error':
        return <AlertCircle className="h-4 w-4 text-red-500" />;
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-blue-500" />;
      default:
        return <Clock className="h-4 w-4 text-gray-500" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'bg-green-100 text-green-800';
      case 'paused': return 'bg-yellow-100 text-yellow-800';
      case 'stopped': return 'bg-gray-100 text-gray-800';
      case 'error': return 'bg-red-100 text-red-800';
      case 'completed': return 'bg-blue-100 text-blue-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'critical': return 'bg-red-100 text-red-800';
      case 'high': return 'bg-orange-100 text-orange-800';
      case 'medium': return 'bg-blue-100 text-blue-800';
      case 'low': return 'bg-gray-100 text-gray-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'data-ingestion': return <Database className="h-4 w-4" />;
      case 'analysis': return <BarChart3 className="h-4 w-4" />;
      case 'trading': return <TrendingUp className="h-4 w-4" />;
      case 'reporting': return <Download className="h-4 w-4" />;
      case 'monitoring': return <Monitor className="h-4 w-4" />;
      default: return <GitBranch className="h-4 w-4" />;
    }
  };

  const handlePipelineAction = (pipelineId: string, action: string) => {
    console.log(`${action} pipeline:`, pipelineId);
    // Implement pipeline control logic here
  };

  const filteredPipelines = pipelines.filter(pipeline => {
    const statusMatch = filterStatus === 'all' || pipeline.status === filterStatus;
    const typeMatch = filterType === 'all' || pipeline.type === filterType;
    return statusMatch && typeMatch;
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Pipeline Control Panel</h2>
          <p className="text-gray-600">Monitor and manage automated trading pipelines</p>
        </div>
        
        <div className="flex items-center space-x-2">
          <Button variant="outline" size="sm">
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
          <Button variant="outline" size="sm">
            <Settings className="h-4 w-4 mr-2" />
            Settings
          </Button>
          <Button className="bg-blue-600 hover:bg-blue-700">
            <Plus className="h-4 w-4 mr-2" />
            New Pipeline
          </Button>
        </div>
      </div>

      {/* Metrics Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Total Pipelines</p>
                <p className="text-2xl font-bold text-gray-900">{metrics.totalPipelines}</p>
              </div>
              <GitBranch className="h-8 w-8 text-blue-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Running</p>
                <p className="text-2xl font-bold text-green-600">{metrics.runningPipelines}</p>
              </div>
              <Activity className="h-8 w-8 text-green-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Successful Runs</p>
                <p className="text-2xl font-bold text-blue-600">{metrics.successfulRuns}</p>
              </div>
              <CheckCircle className="h-8 w-8 text-blue-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Failed Runs</p>
                <p className="text-2xl font-bold text-red-600">{metrics.failedRuns}</p>
              </div>
              <AlertCircle className="h-8 w-8 text-red-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Avg Execution</p>
                <p className="text-2xl font-bold text-gray-900">{metrics.avgExecutionTime}</p>
              </div>
              <Clock className="h-8 w-8 text-gray-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Data Processed</p>
                <p className="text-2xl font-bold text-purple-600">{metrics.dataProcessed}</p>
              </div>
              <Database className="h-8 w-8 text-purple-500" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-4">
        <div className="flex items-center space-x-2">
          <label className="text-sm font-medium text-gray-700">Status:</label>
          <select 
            value={filterStatus}
            onChange={(e) => setFilterStatus(e.target.value)}
            className="px-3 py-1 border border-gray-300 rounded-md text-sm"
          >
            <option value="all">All</option>
            <option value="running">Running</option>
            <option value="paused">Paused</option>
            <option value="stopped">Stopped</option>
            <option value="error">Error</option>
            <option value="completed">Completed</option>
          </select>
        </div>

        <div className="flex items-center space-x-2">
          <label className="text-sm font-medium text-gray-700">Type:</label>
          <select 
            value={filterType}
            onChange={(e) => setFilterType(e.target.value)}
            className="px-3 py-1 border border-gray-300 rounded-md text-sm"
          >
            <option value="all">All</option>
            <option value="data-ingestion">Data Ingestion</option>
            <option value="analysis">Analysis</option>
            <option value="trading">Trading</option>
            <option value="reporting">Reporting</option>
            <option value="monitoring">Monitoring</option>
          </select>
        </div>
      </div>

      {/* Pipelines List */}
      <div className="space-y-4">
        {filteredPipelines.map((pipeline) => (
          <Card key={pipeline.id} className="hover:shadow-md transition-shadow">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4">
                  <div className="flex items-center space-x-2">
                    {getTypeIcon(pipeline.type)}
                    <div>
                      <h3 className="font-semibold text-gray-900">{pipeline.name}</h3>
                      <p className="text-sm text-gray-600">{pipeline.description}</p>
                    </div>
                  </div>
                </div>

                <div className="flex items-center space-x-4">
                  <div className="text-right">
                    <div className="flex items-center space-x-2 mb-1">
                      {getStatusIcon(pipeline.status)}
                      <Badge className={`text-xs ${getStatusColor(pipeline.status)}`}>
                        {pipeline.status}
                      </Badge>
                      <Badge className={`text-xs ${getPriorityColor(pipeline.priority)}`}>
                        {pipeline.priority}
                      </Badge>
                    </div>
                    <p className="text-xs text-gray-500">Success: {pipeline.successRate}%</p>
                  </div>

                  <div className="flex items-center space-x-1">
                    {pipeline.status === 'running' ? (
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handlePipelineAction(pipeline.id, 'pause')}
                      >
                        <Pause className="h-4 w-4" />
                      </Button>
                    ) : (
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handlePipelineAction(pipeline.id, 'start')}
                      >
                        <Play className="h-4 w-4" />
                      </Button>
                    )}
                    
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handlePipelineAction(pipeline.id, 'stop')}
                    >
                      <Square className="h-4 w-4" />
                    </Button>
                    
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handlePipelineAction(pipeline.id, 'restart')}
                    >
                      <RotateCcw className="h-4 w-4" />
                    </Button>
                    
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setSelectedPipeline(pipeline.id)}
                    >
                      <Eye className="h-4 w-4" />
                    </Button>
                    
                    <Button
                      variant="outline"
                      size="sm"
                    >
                      <Edit className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </div>

              <div className="mt-4 grid grid-cols-1 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <span className="text-gray-500">Last Run:</span>
                  <span className="ml-2 font-medium">{pipeline.lastRun}</span>
                </div>
                <div>
                  <span className="text-gray-500">Next Run:</span>
                  <span className="ml-2 font-medium">{pipeline.nextRun}</span>
                </div>
                <div>
                  <span className="text-gray-500">Duration:</span>
                  <span className="ml-2 font-medium">{pipeline.duration}</span>
                </div>
                <div>
                  <span className="text-gray-500">Type:</span>
                  <span className="ml-2 font-medium capitalize">{pipeline.type.replace('-', ' ')}</span>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Pipeline Details Modal/Panel */}
      {selectedPipeline && (
        <Card className="border-2 border-blue-200">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>Pipeline Details</CardTitle>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setSelectedPipeline(null)}
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <h4 className="font-medium mb-2">Execution History</h4>
                  <div className="space-y-2">
                    {[
                      { time: '10:30 AM', status: 'success', duration: '2.1s' },
                      { time: '10:25 AM', status: 'success', duration: '2.3s' },
                      { time: '10:20 AM', status: 'success', duration: '1.9s' },
                      { time: '10:15 AM', status: 'error', duration: '0.5s' },
                      { time: '10:10 AM', status: 'success', duration: '2.2s' }
                    ].map((run, index) => (
                      <div key={index} className="flex items-center justify-between text-sm">
                        <span>{run.time}</span>
                        <div className="flex items-center space-x-2">
                          <Badge className={`text-xs ${
                            run.status === 'success' 
                              ? 'bg-green-100 text-green-800' 
                              : 'bg-red-100 text-red-800'
                          }`}>
                            {run.status}
                          </Badge>
                          <span className="text-gray-500">{run.duration}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div>
                  <h4 className="font-medium mb-2">Performance Metrics</h4>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Success Rate:</span>
                      <span className="font-medium">97.5%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Avg Duration:</span>
                      <span className="font-medium">2.1s</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Total Runs:</span>
                      <span className="font-medium">1,247</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Failed Runs:</span>
                      <span className="font-medium">31</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default PipelineControlPanel;