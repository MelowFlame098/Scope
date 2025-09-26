import { aiAnalysisService } from './aiAnalysisService';
import { FinR1Output, MarketSentiment, TradingSignal } from '../types/ai';

// Real-time data pipeline configuration
interface PipelineConfig {
  updateInterval: number; // milliseconds
  enableScreenCapture: boolean;
  enableWebScraping: boolean;
  enableAIAnalysis: boolean;
  symbols: string[];
}

// Pipeline data structure
interface PipelineData {
  timestamp: string;
  symbol: string;
  screenCapture?: {
    chartImage: string; // base64 encoded image
    ocrText?: string;
    detectedPrices?: number[];
  };
  webScrapingData?: {
    sentiment: MarketSentiment[];
    signals: TradingSignal[];
    newsHeadlines?: string[];
  };
  aiAnalysis?: FinR1Output;
  processingTime: number;
}

// Event types for real-time updates
type PipelineEventType = 'data_update' | 'analysis_complete' | 'error' | 'status_change';

interface PipelineEvent {
  type: PipelineEventType;
  data: any;
  timestamp: string;
}

// Pipeline status
type PipelineStatus = 'idle' | 'running' | 'paused' | 'error';

class RealTimeDataPipeline {
  private config: PipelineConfig;
  private status: PipelineStatus = 'idle';
  private intervalId: NodeJS.Timeout | null = null;
  private subscribers: Map<string, (event: PipelineEvent) => void> = new Map();
  private dataCache: Map<string, PipelineData> = new Map();

  constructor(config: PipelineConfig) {
    this.config = config;
  }

  // Start the real-time pipeline
  start(): void {
    if (this.status === 'running') {
      console.warn('Pipeline is already running');
      return;
    }

    this.status = 'running';
    this.emit('status_change', { status: this.status });

    // Start the main processing loop
    this.intervalId = setInterval(() => {
      this.processPipelineData();
    }, this.config.updateInterval);

    console.log('Real-time data pipeline started');
  }

  // Stop the pipeline
  stop(): void {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }

    this.status = 'idle';
    this.emit('status_change', { status: this.status });
    console.log('Real-time data pipeline stopped');
  }

  // Pause the pipeline
  pause(): void {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }

    this.status = 'paused';
    this.emit('status_change', { status: this.status });
    console.log('Real-time data pipeline paused');
  }

  // Resume the pipeline
  resume(): void {
    if (this.status !== 'paused') {
      console.warn('Pipeline is not paused');
      return;
    }

    this.status = 'running';
    this.emit('status_change', { status: this.status });

    this.intervalId = setInterval(() => {
      this.processPipelineData();
    }, this.config.updateInterval);

    console.log('Real-time data pipeline resumed');
  }

  // Subscribe to pipeline events
  subscribe(id: string, callback: (event: PipelineEvent) => void): void {
    this.subscribers.set(id, callback);
  }

  // Unsubscribe from pipeline events
  unsubscribe(id: string): void {
    this.subscribers.delete(id);
  }

  // Get current pipeline status
  getStatus(): PipelineStatus {
    return this.status;
  }

  // Get cached data for a symbol
  getData(symbol: string): PipelineData | undefined {
    return this.dataCache.get(symbol);
  }

  // Get all cached data
  getAllData(): Map<string, PipelineData> {
    return new Map(this.dataCache);
  }

  // Update pipeline configuration
  updateConfig(newConfig: Partial<PipelineConfig>): void {
    this.config = { ...this.config, ...newConfig };
    
    // Restart if running to apply new config
    if (this.status === 'running') {
      this.stop();
      this.start();
    }
  }

  // Main processing function
  private async processPipelineData(): Promise<void> {
    const startTime = Date.now();

    try {
      for (const symbol of this.config.symbols) {
        const pipelineData: PipelineData = {
          timestamp: new Date().toISOString(),
          symbol,
          processingTime: 0
        };

        // Step 1: Screen capture (if enabled)
        if (this.config.enableScreenCapture) {
          pipelineData.screenCapture = await this.captureScreen(symbol);
        }

        // Step 2: Web scraping (if enabled)
        if (this.config.enableWebScraping) {
          pipelineData.webScrapingData = await this.scrapeWebData(symbol);
        }

        // Step 3: AI analysis (if enabled)
        if (this.config.enableAIAnalysis) {
          pipelineData.aiAnalysis = await this.performAIAnalysis(symbol, pipelineData);
        }

        pipelineData.processingTime = Date.now() - startTime;

        // Cache the data
        this.dataCache.set(symbol, pipelineData);

        // Emit data update event
        this.emit('data_update', pipelineData);

        // Emit analysis complete event if AI analysis was performed
        if (pipelineData.aiAnalysis) {
          this.emit('analysis_complete', {
            symbol,
            analysis: pipelineData.aiAnalysis,
            processingTime: pipelineData.processingTime
          });
        }
      }
    } catch (error) {
      console.error('Pipeline processing error:', error);
      this.status = 'error';
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      this.emit('error', { error: errorMessage, timestamp: new Date().toISOString() });
    }
  }

  // Mock screen capture function
  private async captureScreen(symbol: string): Promise<PipelineData['screenCapture']> {
    // Simulate screen capture delay
    await new Promise(resolve => setTimeout(resolve, 200));
    
    // In a real implementation, this would:
    // 1. Capture the current screen or specific window
    // 2. Use OCR to extract text and numbers
    // 3. Detect chart patterns and price levels
    
    return {
      chartImage: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==', // 1x1 transparent pixel
      ocrText: `${symbol} Price: $145.67 Volume: 1.2M`,
      detectedPrices: [145.67, 144.23, 146.89]
    };
  }

  // Mock web scraping function
  private async scrapeWebData(symbol: string): Promise<PipelineData['webScrapingData']> {
    // Simulate web scraping delay
    await new Promise(resolve => setTimeout(resolve, 500));
    
    // In a real implementation, this would call the actual market scraper service
    const sentiment = await aiAnalysisService.getMarketSentiment(symbol);
    const signals = await aiAnalysisService.getTradingSignals(symbol);
    
    return {
      sentiment,
      signals,
      newsHeadlines: [
        `${symbol} reports strong quarterly earnings`,
        `Analysts upgrade ${symbol} price target`,
        `${symbol} announces new product launch`
      ]
    };
  }

  // AI analysis function
  private async performAIAnalysis(symbol: string, data: PipelineData): Promise<FinR1Output> {
    // Use the existing AI analysis service
    return await aiAnalysisService.getAnalysis(symbol);
  }

  // Event emission helper
  private emit(type: PipelineEventType, data: any): void {
    const event: PipelineEvent = {
      type,
      data,
      timestamp: new Date().toISOString()
    };

    this.subscribers.forEach(callback => {
      try {
        callback(event);
      } catch (error) {
        console.error('Error in pipeline event callback:', error);
      }
    });
  }
}

// Default pipeline configuration
const defaultConfig: PipelineConfig = {
  updateInterval: 30000, // 30 seconds
  enableScreenCapture: true,
  enableWebScraping: true,
  enableAIAnalysis: true,
  symbols: ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
};

// Export singleton instance
export const realTimeDataPipeline = new RealTimeDataPipeline(defaultConfig);

// Export types and classes
export type { PipelineConfig, PipelineData, PipelineEvent, PipelineStatus };
export { RealTimeDataPipeline };