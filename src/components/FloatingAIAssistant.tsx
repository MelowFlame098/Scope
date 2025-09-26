'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Button } from './ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Loader2, Brain, Camera, TrendingUp, AlertTriangle, X, Minimize2, Maximize2 } from 'lucide-react';
import { chartAnalysisApi } from '../lib/api';

interface FloatingAIAssistantProps {
  symbol?: string;
  onClose?: () => void;
  position?: { x: number; y: number };
  chartContainerRef?: React.RefObject<HTMLElement>;
}

interface AnalysisResult {
  success: boolean;
  chart_analysis?: any;
  market_analysis?: any;
  combined_recommendation?: any;
  error?: string;
}

interface CaptureRegion {
  x: number;
  y: number;
  width: number;
  height: number;
}

export default function FloatingAIAssistant({ 
  symbol = 'AAPL', 
  onClose,
  position = { x: 20, y: 20 },
  chartContainerRef
}: FloatingAIAssistantProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [captureRegion, setCaptureRegion] = useState<CaptureRegion | null>(null);
  const [isSelecting, setIsSelecting] = useState(false);
  const [dragPosition, setDragPosition] = useState(position);
  const [isDragging, setIsDragging] = useState(false);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  
  const assistantRef = useRef<HTMLDivElement>(null);
  const selectionRef = useRef<HTMLDivElement>(null);

  // Handle dragging
  const handleMouseDown = (e: React.MouseEvent) => {
    if (isExpanded) return; // Don't drag when expanded
    
    setIsDragging(true);
    const rect = assistantRef.current?.getBoundingClientRect();
    if (rect) {
      setDragOffset({
        x: e.clientX - rect.left,
        y: e.clientY - rect.top
      });
    }
  };

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (isDragging) {
        setDragPosition({
          x: e.clientX - dragOffset.x,
          y: e.clientY - dragOffset.y
        });
      }
    };

    const handleMouseUp = () => {
      setIsDragging(false);
    };

    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, dragOffset]);

  // Screen capture region selection
  const startRegionSelection = () => {
    setIsSelecting(true);
    setCaptureRegion(null);
  };

  const handleRegionSelect = (region: CaptureRegion) => {
    setCaptureRegion(region);
    setIsSelecting(false);
  };

  // Enhanced analysis with screen capture
  const performEnhancedAnalysis = async () => {
    setIsAnalyzing(true);
    setAnalysisResult(null);

    try {
      let analysisData;
      
      if (captureRegion) {
        // Use specific region
        analysisData = {
          capture_region: captureRegion,
          symbol,
          timeframe: '1D',
          custom_indicators: [],
          enable_forecasting: true,
          enable_execution: true,
          include_market_data: true
        };
      } else {
        // Use chart container bounds if available
        if (chartContainerRef?.current) {
          const rect = chartContainerRef.current.getBoundingClientRect();
          analysisData = {
            capture_region: {
              x: Math.round(rect.left),
              y: Math.round(rect.top),
              width: Math.round(rect.width),
              height: Math.round(rect.height)
            },
            symbol,
            timeframe: '1D',
            custom_indicators: [],
            enable_forecasting: true,
            enable_execution: true,
            include_market_data: true
          };
        } else {
          // Fallback to full screen
          analysisData = {
            symbol,
            timeframe: '1D',
            custom_indicators: [],
            enable_forecasting: true,
            enable_execution: true,
            include_market_data: true
          };
        }
      }

      const response = await fetch('/api/screen-analysis/enhanced-analysis', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(analysisData)
      });

      const result = await response.json();
      setAnalysisResult(result);

    } catch (error) {
      console.error('Analysis failed:', error);
      setAnalysisResult({
        success: false,
        error: error instanceof Error ? error.message : 'Analysis failed'
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Quick market analysis without screen capture
  const performQuickAnalysis = async () => {
    setIsAnalyzing(true);
    setAnalysisResult(null);

    try {
      const response = await fetch('/api/screen-analysis/analyze-market', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          symbol,
          include_sentiment: true,
          include_signals: true,
          include_fundamentals: true
        })
      });

      const result = await response.json();
      setAnalysisResult({ 
        success: result.success,
        market_analysis: result,
        error: result.error
      });

    } catch (error) {
      console.error('Quick analysis failed:', error);
      setAnalysisResult({
        success: false,
        error: error instanceof Error ? error.message : 'Quick analysis failed'
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getRecommendationColor = (recommendation: string) => {
    switch (recommendation?.toUpperCase()) {
      case 'BUY': return 'bg-green-500';
      case 'SELL': return 'bg-red-500';
      default: return 'bg-yellow-500';
    }
  };

  const getRecommendationIcon = (recommendation: string) => {
    switch (recommendation?.toUpperCase()) {
      case 'BUY': return <TrendingUp className="h-4 w-4" />;
      case 'SELL': return <AlertTriangle className="h-4 w-4" />;
      default: return <Minimize2 className="h-4 w-4" />;
    }
  };

  if (!isExpanded) {
    return (
      <div
        ref={assistantRef}
        className="fixed z-50 cursor-move"
        style={{
          left: dragPosition.x,
          top: dragPosition.y,
          userSelect: 'none'
        }}
        onMouseDown={handleMouseDown}
      >
        <Button
          onClick={() => setIsExpanded(true)}
          className="rounded-full w-12 h-12 bg-blue-600 hover:bg-blue-700 shadow-lg border-2 border-white"
          disabled={isAnalyzing}
        >
          {isAnalyzing ? (
            <Loader2 className="h-6 w-6 animate-spin" />
          ) : (
            <Brain className="h-6 w-6" />
          )}
        </Button>
        
        {analysisResult?.combined_recommendation && (
          <div className="absolute -top-2 -right-2">
            <Badge 
              className={`${getRecommendationColor(analysisResult.combined_recommendation.recommendation)} text-white text-xs px-1 py-0.5`}
            >
              {analysisResult.combined_recommendation.recommendation}
            </Badge>
          </div>
        )}
      </div>
    );
  }

  return (
    <>
      {/* Region Selection Overlay */}
      {isSelecting && (
        <RegionSelector onRegionSelect={handleRegionSelect} onCancel={() => setIsSelecting(false)} />
      )}
      
      {/* Expanded AI Assistant */}
      <div
        ref={assistantRef}
        className="fixed z-50 w-96 max-h-[80vh] overflow-hidden"
        style={{
          left: Math.min(dragPosition.x, window.innerWidth - 400),
          top: Math.min(dragPosition.y, window.innerHeight - 600)
        }}
      >
        <Card className="shadow-2xl border-2 border-blue-200">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-lg flex items-center gap-2">
                <Brain className="h-5 w-5 text-blue-600" />
                AI Chart Assistant
              </CardTitle>
              <div className="flex gap-1">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setIsExpanded(false)}
                  className="h-8 w-8 p-0"
                >
                  <Minimize2 className="h-4 w-4" />
                </Button>
                {onClose && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={onClose}
                    className="h-8 w-8 p-0"
                  >
                    <X className="h-4 w-4" />
                  </Button>
                )}
              </div>
            </div>
            <div className="text-sm text-gray-600">
              Analyzing: <Badge variant="outline">{symbol}</Badge>
            </div>
          </CardHeader>
          
          <CardContent className="space-y-4 max-h-[60vh] overflow-y-auto">
            {/* Action Buttons */}
            <div className="grid grid-cols-2 gap-2">
              <Button
                onClick={performEnhancedAnalysis}
                disabled={isAnalyzing}
                className="flex items-center gap-2"
                size="sm"
              >
                <Camera className="h-4 w-4" />
                Screen Analysis
              </Button>
              
              <Button
                onClick={performQuickAnalysis}
                disabled={isAnalyzing}
                variant="outline"
                className="flex items-center gap-2"
                size="sm"
              >
                <TrendingUp className="h-4 w-4" />
                Market Data
              </Button>
            </div>
            
            {/* Region Selection */}
            <div className="space-y-2">
              <Button
                onClick={startRegionSelection}
                variant="outline"
                size="sm"
                className="w-full"
                disabled={isAnalyzing}
              >
                Select Chart Region
              </Button>
              
              {captureRegion && (
                <div className="text-xs text-gray-600 bg-gray-50 p-2 rounded">
                  Region: {captureRegion.width}×{captureRegion.height} at ({captureRegion.x}, {captureRegion.y})
                </div>
              )}
            </div>

            {/* Loading State */}
            {isAnalyzing && (
              <div className="flex items-center justify-center py-8">
                <div className="text-center">
                  <Loader2 className="h-8 w-8 animate-spin mx-auto mb-2 text-blue-600" />
                  <p className="text-sm text-gray-600">Analyzing chart and market data...</p>
                </div>
              </div>
            )}

            {/* Analysis Results */}
            {analysisResult && !isAnalyzing && (
              <div className="space-y-4">
                {analysisResult.success ? (
                  <>
                    {/* Combined Recommendation */}
                    {analysisResult.combined_recommendation && (
                      <div className="bg-gray-50 p-3 rounded-lg">
                        <div className="flex items-center gap-2 mb-2">
                          {getRecommendationIcon(analysisResult.combined_recommendation.recommendation)}
                          <span className="font-semibold">
                            {analysisResult.combined_recommendation.recommendation}
                          </span>
                          <Badge className={getRecommendationColor(analysisResult.combined_recommendation.recommendation)}>
                            {(analysisResult.combined_recommendation.confidence * 100).toFixed(1)}%
                          </Badge>
                        </div>
                        <div className="text-sm text-gray-600">
                          Score: {analysisResult.combined_recommendation.final_score?.toFixed(3)}
                        </div>
                        {analysisResult.combined_recommendation.reasoning && (
                          <div className="mt-2 text-xs text-gray-500">
                            {analysisResult.combined_recommendation.reasoning.map((reason: string, idx: number) => (
                              <div key={idx}>• {reason}</div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}

                    {/* Market Analysis Summary */}
                    {analysisResult.market_analysis?.overall_recommendation && (
                      <div className="border-l-4 border-blue-500 pl-3">
                        <h4 className="font-medium text-sm">Market Sentiment</h4>
                        <div className="text-sm text-gray-600">
                          {analysisResult.market_analysis.overall_recommendation.recommendation} 
                          ({(analysisResult.market_analysis.overall_recommendation.confidence * 100).toFixed(1)}%)
                        </div>
                      </div>
                    )}

                    {/* Chart Analysis Summary */}
                    {analysisResult.chart_analysis?.execution_signals && (
                      <div className="border-l-4 border-green-500 pl-3">
                        <h4 className="font-medium text-sm">Chart Analysis</h4>
                        <div className="text-sm text-gray-600">
                          {analysisResult.chart_analysis.execution_signals.primary_signal?.action || 'No clear signal'}
                        </div>
                      </div>
                    )}

                    {/* Trading Signals */}
                    {analysisResult.market_analysis?.trading_signals && (
                      <div className="space-y-2">
                        <h4 className="font-medium text-sm">Trading Signals</h4>
                        {analysisResult.market_analysis.trading_signals.slice(0, 3).map((signal: any, idx: number) => (
                          <div key={idx} className="text-xs bg-white p-2 rounded border">
                            <div className="flex justify-between">
                              <span className="font-medium">{signal.type?.toUpperCase()}</span>
                              <span className="text-gray-500">{(signal.strength * 100).toFixed(0)}%</span>
                            </div>
                            <div className="text-gray-600 mt-1">{signal.source}</div>
                          </div>
                        ))}
                      </div>
                    )}
                  </>
                ) : (
                  <div className="text-red-600 text-sm bg-red-50 p-3 rounded">
                    <AlertTriangle className="h-4 w-4 inline mr-2" />
                    {analysisResult.error || 'Analysis failed'}
                  </div>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </>
  );
}

// Region Selector Component
interface RegionSelectorProps {
  onRegionSelect: (region: CaptureRegion) => void;
  onCancel: () => void;
}

function RegionSelector({ onRegionSelect, onCancel }: RegionSelectorProps) {
  const [isSelecting, setIsSelecting] = useState(false);
  const [startPoint, setStartPoint] = useState<{ x: number; y: number } | null>(null);
  const [currentPoint, setCurrentPoint] = useState<{ x: number; y: number } | null>(null);

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsSelecting(true);
    setStartPoint({ x: e.clientX, y: e.clientY });
    setCurrentPoint({ x: e.clientX, y: e.clientY });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isSelecting && startPoint) {
      setCurrentPoint({ x: e.clientX, y: e.clientY });
    }
  };

  const handleMouseUp = () => {
    if (isSelecting && startPoint && currentPoint) {
      const region = {
        x: Math.min(startPoint.x, currentPoint.x),
        y: Math.min(startPoint.y, currentPoint.y),
        width: Math.abs(currentPoint.x - startPoint.x),
        height: Math.abs(currentPoint.y - startPoint.y)
      };
      
      if (region.width > 50 && region.height > 50) {
        onRegionSelect(region);
      }
    }
    setIsSelecting(false);
    setStartPoint(null);
    setCurrentPoint(null);
  };

  const getSelectionStyle = () => {
    if (!startPoint || !currentPoint) return {};
    
    return {
      left: Math.min(startPoint.x, currentPoint.x),
      top: Math.min(startPoint.y, currentPoint.y),
      width: Math.abs(currentPoint.x - startPoint.x),
      height: Math.abs(currentPoint.y - startPoint.y)
    };
  };

  return (
    <div
      className="fixed inset-0 z-[60] bg-black bg-opacity-30 cursor-crosshair"
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onContextMenu={(e) => {
        e.preventDefault();
        onCancel();
      }}
    >
      <div className="absolute top-4 left-1/2 transform -translate-x-1/2 bg-white px-4 py-2 rounded-lg shadow-lg">
        <p className="text-sm">Click and drag to select chart region. Right-click to cancel.</p>
      </div>
      
      {isSelecting && startPoint && currentPoint && (
        <div
          className="absolute border-2 border-blue-500 bg-blue-200 bg-opacity-20"
          style={getSelectionStyle()}
        />
      )}
    </div>
  );
}