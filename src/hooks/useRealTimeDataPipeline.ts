import { useState, useEffect, useCallback, useRef } from 'react';
import { 
  realTimeDataPipeline, 
  PipelineData, 
  PipelineEvent, 
  PipelineStatus,
  PipelineConfig 
} from '../services/realTimeDataPipeline';
import { FinR1Output } from '../types/ai';

interface UseRealTimeDataPipelineReturn {
  // Pipeline status
  status: PipelineStatus;
  isRunning: boolean;
  
  // Data
  data: Map<string, PipelineData>;
  latestAnalysis: Map<string, FinR1Output>;
  
  // Controls
  start: () => void;
  stop: () => void;
  pause: () => void;
  resume: () => void;
  updateConfig: (config: Partial<PipelineConfig>) => void;
  
  // Data access
  getDataForSymbol: (symbol: string) => PipelineData | undefined;
  getAnalysisForSymbol: (symbol: string) => FinR1Output | undefined;
  
  // Events
  lastEvent: PipelineEvent | null;
  eventHistory: PipelineEvent[];
  
  // Statistics
  stats: {
    totalUpdates: number;
    lastUpdateTime: string | null;
    averageProcessingTime: number;
    errorCount: number;
  };
}

export const useRealTimeDataPipeline = (): UseRealTimeDataPipelineReturn => {
  const [status, setStatus] = useState<PipelineStatus>(realTimeDataPipeline.getStatus());
  const [data, setData] = useState<Map<string, PipelineData>>(new Map());
  const [latestAnalysis, setLatestAnalysis] = useState<Map<string, FinR1Output>>(new Map());
  const [lastEvent, setLastEvent] = useState<PipelineEvent | null>(null);
  const [eventHistory, setEventHistory] = useState<PipelineEvent[]>([]);
  const [stats, setStats] = useState({
    totalUpdates: 0,
    lastUpdateTime: null as string | null,
    averageProcessingTime: 0,
    errorCount: 0
  });
  
  const processingTimes = useRef<number[]>([]);
  const hookId = useRef(`hook_${Date.now()}_${Math.random()}`);

  // Event handler for pipeline events
  const handlePipelineEvent = useCallback((event: PipelineEvent) => {
    setLastEvent(event);
    setEventHistory(prev => {
      const newHistory = [...prev, event];
      // Keep only last 100 events to prevent memory issues
      return newHistory.slice(-100);
    });

    switch (event.type) {
      case 'status_change':
        setStatus(event.data.status);
        break;
        
      case 'data_update':
        const pipelineData = event.data as PipelineData;
        
        setData(prev => {
          const newData = new Map(prev);
          newData.set(pipelineData.symbol, pipelineData);
          return newData;
        });
        
        // Update statistics
        setStats(prev => {
          processingTimes.current.push(pipelineData.processingTime);
          // Keep only last 50 processing times for average calculation
          if (processingTimes.current.length > 50) {
            processingTimes.current = processingTimes.current.slice(-50);
          }
          
          const averageProcessingTime = processingTimes.current.reduce((a, b) => a + b, 0) / processingTimes.current.length;
          
          return {
            ...prev,
            totalUpdates: prev.totalUpdates + 1,
            lastUpdateTime: pipelineData.timestamp,
            averageProcessingTime
          };
        });
        break;
        
      case 'analysis_complete':
        const { symbol, analysis } = event.data;
        setLatestAnalysis(prev => {
          const newAnalysis = new Map(prev);
          newAnalysis.set(symbol, analysis);
          return newAnalysis;
        });
        break;
        
      case 'error':
        setStats(prev => ({
          ...prev,
          errorCount: prev.errorCount + 1
        }));
        console.error('Pipeline error:', event.data);
        break;
    }
  }, []);

  // Subscribe to pipeline events on mount
  useEffect(() => {
    realTimeDataPipeline.subscribe(hookId.current, handlePipelineEvent);
    
    // Initialize with current data
    setData(realTimeDataPipeline.getAllData());
    setStatus(realTimeDataPipeline.getStatus());
    
    return () => {
      realTimeDataPipeline.unsubscribe(hookId.current);
    };
  }, [handlePipelineEvent]);

  // Control functions
  const start = useCallback(() => {
    realTimeDataPipeline.start();
  }, []);

  const stop = useCallback(() => {
    realTimeDataPipeline.stop();
  }, []);

  const pause = useCallback(() => {
    realTimeDataPipeline.pause();
  }, []);

  const resume = useCallback(() => {
    realTimeDataPipeline.resume();
  }, []);

  const updateConfig = useCallback((config: Partial<PipelineConfig>) => {
    realTimeDataPipeline.updateConfig(config);
  }, []);

  // Data access functions
  const getDataForSymbol = useCallback((symbol: string) => {
    return data.get(symbol);
  }, [data]);

  const getAnalysisForSymbol = useCallback((symbol: string) => {
    return latestAnalysis.get(symbol);
  }, [latestAnalysis]);

  return {
    status,
    isRunning: status === 'running',
    data,
    latestAnalysis,
    start,
    stop,
    pause,
    resume,
    updateConfig,
    getDataForSymbol,
    getAnalysisForSymbol,
    lastEvent,
    eventHistory,
    stats
  };
};

// Hook for specific symbol data
export const useSymbolPipelineData = (symbol: string) => {
  const pipeline = useRealTimeDataPipeline();
  
  const symbolData = pipeline.getDataForSymbol(symbol);
  const symbolAnalysis = pipeline.getAnalysisForSymbol(symbol);
  
  return {
    ...pipeline,
    symbolData,
    symbolAnalysis,
    hasData: !!symbolData,
    hasAnalysis: !!symbolAnalysis,
    lastUpdate: symbolData?.timestamp,
    processingTime: symbolData?.processingTime
  };
};