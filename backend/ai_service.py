import asyncio
import aiohttp
import openai
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import os
from dotenv import load_dotenv
import yfinance as yf
from textblob import TextBlob
import ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

from schemas import AIInsightResponse, AnalysisRequest

load_dotenv()

class AIService:
    def __init__(self):
        # API keys
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        
        # Initialize OpenAI if key is available
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
        
        # Cache for AI insights
        self.cache = {}
        self.cache_ttl = 600  # 10 minutes cache
        
        # ML models cache
        self.models_cache = {}
        
        # Technical indicators configuration
        self.indicators_config = {
            'sma_periods': [20, 50, 200],
            'ema_periods': [12, 26],
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'bb_std': 2
        }
    
    async def analyze_asset(self, request: AnalysisRequest) -> AIInsightResponse:
        """Analyze an asset and provide AI-generated insights."""
        try:
            # Check cache first
            cache_key = f"analysis_{request.symbol}_{request.timeframe}_{request.analysis_type}"
            if self._is_cache_valid(cache_key):
                return self.cache[cache_key]['data']
            
            # Get market data
            market_data = await self._get_market_data(request.symbol, request.timeframe)
            
            if not market_data:
                return self._get_mock_analysis(request)
            
            # Perform different types of analysis
            if request.analysis_type == 'technical':
                insight = await self._technical_analysis(request.symbol, market_data)
            elif request.analysis_type == 'fundamental':
                insight = await self._fundamental_analysis(request.symbol, market_data)
            elif request.analysis_type == 'sentiment':
                insight = await self._sentiment_analysis(request.symbol)
            elif request.analysis_type == 'prediction':
                insight = await self._price_prediction(request.symbol, market_data)
            else:
                insight = await self._comprehensive_analysis(request.symbol, market_data)
            
            # Cache the result
            self.cache[cache_key] = {
                'data': insight,
                'timestamp': datetime.now()
            }
            
            return insight
            
        except Exception as e:
            print(f"Error in asset analysis: {e}")
            return self._get_mock_analysis(request)
    
    async def explain_data(self, data: Dict[str, Any], context: str = "") -> str:
        """Explain complex financial data in simple terms."""
        try:
            if self.openai_api_key:
                return await self._openai_explanation(data, context)
            else:
                return self._rule_based_explanation(data, context)
                
        except Exception as e:
            print(f"Error explaining data: {e}")
            return "Unable to generate explanation at this time."
    
    async def get_insights(self, category: Optional[str] = None, limit: int = 10) -> List[AIInsightResponse]:
        """Get AI-generated market insights."""
        try:
            # Check cache first
            cache_key = f"insights_{category}_{limit}"
            if self._is_cache_valid(cache_key):
                return self.cache[cache_key]['data']
            
            insights = []
            
            # Generate insights for popular assets
            popular_assets = ['BTC-USD', 'ETH-USD', 'AAPL', 'MSFT', 'GOOGL', 'TSLA']
            
            for symbol in popular_assets[:limit]:
                try:
                    # Create analysis request
                    request = AnalysisRequest(
                        symbol=symbol,
                        timeframe='1d',
                        analysis_type='comprehensive'
                    )
                    
                    insight = await self.analyze_asset(request)
                    
                    # Filter by category if specified
                    if category and insight.category != category:
                        continue
                    
                    insights.append(insight)
                    
                except Exception as e:
                    print(f"Error generating insight for {symbol}: {e}")
                    continue
            
            # Add market-wide insights
            market_insights = await self._generate_market_insights()
            insights.extend(market_insights)
            
            # Sort by confidence and recency
            insights.sort(key=lambda x: (x.confidence, x.created_at), reverse=True)
            
            result = insights[:limit]
            
            # Cache the result
            self.cache[cache_key] = {
                'data': result,
                'timestamp': datetime.now()
            }
            
            return result
            
        except Exception as e:
            print(f"Error getting insights: {e}")
            return self._get_mock_insights(category, limit)
    
    async def _technical_analysis(self, symbol: str, data: pd.DataFrame) -> AIInsightResponse:
        """Perform technical analysis on asset data."""
        try:
            # Calculate technical indicators
            indicators = self._calculate_indicators(data)
            
            # Generate signals
            signals = self._generate_technical_signals(indicators)
            
            # Create analysis text
            analysis_text = self._format_technical_analysis(symbol, indicators, signals)
            
            # Determine overall sentiment
            bullish_signals = sum(1 for signal in signals.values() if signal == 'bullish')
            bearish_signals = sum(1 for signal in signals.values() if signal == 'bearish')
            
            if bullish_signals > bearish_signals:
                sentiment = 'bullish'
                confidence = min(0.9, 0.5 + (bullish_signals - bearish_signals) * 0.1)
            elif bearish_signals > bullish_signals:
                sentiment = 'bearish'
                confidence = min(0.9, 0.5 + (bearish_signals - bullish_signals) * 0.1)
            else:
                sentiment = 'neutral'
                confidence = 0.5
            
            return AIInsightResponse(
                id=f"tech_{symbol}_{int(datetime.now().timestamp())}",
                title=f"Technical Analysis: {symbol}",
                content=analysis_text,
                summary=f"Technical analysis shows {sentiment} signals for {symbol}",
                insight_type="analysis",
                category="technical",
                related_symbols=[symbol],
                confidence=confidence,
                sentiment=sentiment,
                tags=["technical-analysis", "indicators", "signals"],
                metadata={
                    "indicators": indicators,
                    "signals": signals,
                    "timeframe": "1d"
                },
                created_at=datetime.now()
            )
            
        except Exception as e:
            print(f"Error in technical analysis: {e}")
            return self._get_mock_analysis_response(symbol, "technical")
    
    async def _fundamental_analysis(self, symbol: str, data: pd.DataFrame) -> AIInsightResponse:
        """Perform fundamental analysis on asset data."""
        try:
            # Get fundamental data (mock for now)
            fundamentals = await self._get_fundamental_data(symbol)
            
            # Analyze fundamentals
            analysis = self._analyze_fundamentals(fundamentals)
            
            # Create analysis text
            analysis_text = self._format_fundamental_analysis(symbol, fundamentals, analysis)
            
            return AIInsightResponse(
                id=f"fund_{symbol}_{int(datetime.now().timestamp())}",
                title=f"Fundamental Analysis: {symbol}",
                content=analysis_text,
                summary=f"Fundamental analysis of {symbol} shows {analysis['overall_rating']}",
                insight_type="analysis",
                category="fundamental",
                related_symbols=[symbol],
                confidence=analysis['confidence'],
                sentiment=analysis['sentiment'],
                tags=["fundamental-analysis", "valuation", "financials"],
                metadata={
                    "fundamentals": fundamentals,
                    "analysis": analysis
                },
                created_at=datetime.now()
            )
            
        except Exception as e:
            print(f"Error in fundamental analysis: {e}")
            return self._get_mock_analysis_response(symbol, "fundamental")
    
    async def _sentiment_analysis(self, symbol: str) -> AIInsightResponse:
        """Perform sentiment analysis on asset-related news and social media."""
        try:
            # Get recent news and social sentiment (mock for now)
            sentiment_data = await self._get_sentiment_data(symbol)
            
            # Analyze sentiment
            analysis = self._analyze_sentiment_data(sentiment_data)
            
            # Create analysis text
            analysis_text = self._format_sentiment_analysis(symbol, sentiment_data, analysis)
            
            return AIInsightResponse(
                id=f"sent_{symbol}_{int(datetime.now().timestamp())}",
                title=f"Sentiment Analysis: {symbol}",
                content=analysis_text,
                summary=f"Market sentiment for {symbol} is {analysis['overall_sentiment']}",
                insight_type="analysis",
                category="sentiment",
                related_symbols=[symbol],
                confidence=analysis['confidence'],
                sentiment=analysis['overall_sentiment'],
                tags=["sentiment-analysis", "social-media", "news"],
                metadata={
                    "sentiment_data": sentiment_data,
                    "analysis": analysis
                },
                created_at=datetime.now()
            )
            
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            return self._get_mock_analysis_response(symbol, "sentiment")
    
    async def _price_prediction(self, symbol: str, data: pd.DataFrame) -> AIInsightResponse:
        """Generate price predictions using ML models."""
        try:
            # Prepare features
            features = self._prepare_ml_features(data)
            
            # Train or get cached model
            model = await self._get_ml_model(symbol, features)
            
            # Make predictions
            predictions = self._make_predictions(model, features)
            
            # Create analysis text
            analysis_text = self._format_prediction_analysis(symbol, predictions)
            
            # Determine confidence based on model performance
            confidence = min(0.8, max(0.3, predictions.get('confidence', 0.5)))
            
            return AIInsightResponse(
                id=f"pred_{symbol}_{int(datetime.now().timestamp())}",
                title=f"Price Prediction: {symbol}",
                content=analysis_text,
                summary=f"ML model predicts {predictions['direction']} movement for {symbol}",
                insight_type="prediction",
                category="ml",
                related_symbols=[symbol],
                confidence=confidence,
                sentiment=predictions['sentiment'],
                tags=["price-prediction", "machine-learning", "forecast"],
                metadata={
                    "predictions": predictions,
                    "model_type": "random_forest",
                    "timeframe": "1d"
                },
                created_at=datetime.now()
            )
            
        except Exception as e:
            print(f"Error in price prediction: {e}")
            return self._get_mock_analysis_response(symbol, "prediction")
    
    async def _comprehensive_analysis(self, symbol: str, data: pd.DataFrame) -> AIInsightResponse:
        """Perform comprehensive analysis combining multiple approaches."""
        try:
            # Get all analysis types
            technical = await self._technical_analysis(symbol, data)
            fundamental = await self._fundamental_analysis(symbol, data)
            sentiment = await self._sentiment_analysis(symbol)
            prediction = await self._price_prediction(symbol, data)
            
            # Combine insights
            combined_analysis = self._combine_analyses([technical, fundamental, sentiment, prediction])
            
            return AIInsightResponse(
                id=f"comp_{symbol}_{int(datetime.now().timestamp())}",
                title=f"Comprehensive Analysis: {symbol}",
                content=combined_analysis['content'],
                summary=combined_analysis['summary'],
                insight_type="analysis",
                category="comprehensive",
                related_symbols=[symbol],
                confidence=combined_analysis['confidence'],
                sentiment=combined_analysis['sentiment'],
                tags=["comprehensive", "multi-factor", "analysis"],
                metadata={
                    "component_analyses": {
                        "technical": technical.metadata,
                        "fundamental": fundamental.metadata,
                        "sentiment": sentiment.metadata,
                        "prediction": prediction.metadata
                    }
                },
                created_at=datetime.now()
            )
            
        except Exception as e:
            print(f"Error in comprehensive analysis: {e}")
            return self._get_mock_analysis_response(symbol, "comprehensive")
    
    async def _get_market_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Get market data for analysis."""
        try:
            # Map timeframes
            period_map = {
                '1h': '5d',
                '4h': '1mo',
                '1d': '1y',
                '1w': '5y'
            }
            
            period = period_map.get(timeframe, '1y')
            
            # Fetch data using yfinance
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=timeframe)
            
            if data.empty:
                return None
            
            return data
            
        except Exception as e:
            print(f"Error fetching market data for {symbol}: {e}")
            return None
    
    def _calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators."""
        indicators = {}
        
        try:
            # Simple Moving Averages
            for period in self.indicators_config['sma_periods']:
                indicators[f'sma_{period}'] = ta.trend.sma_indicator(data['Close'], window=period).iloc[-1]
            
            # Exponential Moving Averages
            for period in self.indicators_config['ema_periods']:
                indicators[f'ema_{period}'] = ta.trend.ema_indicator(data['Close'], window=period).iloc[-1]
            
            # RSI
            indicators['rsi'] = ta.momentum.rsi(data['Close'], window=self.indicators_config['rsi_period']).iloc[-1]
            
            # MACD
            macd_line = ta.trend.macd(data['Close'], 
                                    window_slow=self.indicators_config['macd_slow'],
                                    window_fast=self.indicators_config['macd_fast']).iloc[-1]
            macd_signal = ta.trend.macd_signal(data['Close'],
                                             window_slow=self.indicators_config['macd_slow'],
                                             window_fast=self.indicators_config['macd_fast'],
                                             window_sign=self.indicators_config['macd_signal']).iloc[-1]
            
            indicators['macd'] = macd_line
            indicators['macd_signal'] = macd_signal
            indicators['macd_histogram'] = macd_line - macd_signal
            
            # Bollinger Bands
            bb_high = ta.volatility.bollinger_hband(data['Close'], 
                                                   window=self.indicators_config['bb_period'],
                                                   window_dev=self.indicators_config['bb_std']).iloc[-1]
            bb_low = ta.volatility.bollinger_lband(data['Close'],
                                                  window=self.indicators_config['bb_period'],
                                                  window_dev=self.indicators_config['bb_std']).iloc[-1]
            bb_mid = ta.volatility.bollinger_mavg(data['Close'],
                                                 window=self.indicators_config['bb_period']).iloc[-1]
            
            indicators['bb_upper'] = bb_high
            indicators['bb_lower'] = bb_low
            indicators['bb_middle'] = bb_mid
            
            # Current price
            indicators['current_price'] = data['Close'].iloc[-1]
            
            # Volume indicators
            indicators['volume_sma'] = ta.volume.volume_sma(data['Close'], data['Volume'], window=20).iloc[-1]
            
        except Exception as e:
            print(f"Error calculating indicators: {e}")
        
        return indicators
    
    def _generate_technical_signals(self, indicators: Dict[str, Any]) -> Dict[str, str]:
        """Generate trading signals from technical indicators."""
        signals = {}
        
        try:
            current_price = indicators.get('current_price', 0)
            
            # RSI signals
            rsi = indicators.get('rsi', 50)
            if rsi > 70:
                signals['rsi'] = 'bearish'  # Overbought
            elif rsi < 30:
                signals['rsi'] = 'bullish'  # Oversold
            else:
                signals['rsi'] = 'neutral'
            
            # MACD signals
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            if macd > macd_signal:
                signals['macd'] = 'bullish'
            else:
                signals['macd'] = 'bearish'
            
            # Moving average signals
            sma_20 = indicators.get('sma_20', current_price)
            sma_50 = indicators.get('sma_50', current_price)
            
            if current_price > sma_20 > sma_50:
                signals['ma_trend'] = 'bullish'
            elif current_price < sma_20 < sma_50:
                signals['ma_trend'] = 'bearish'
            else:
                signals['ma_trend'] = 'neutral'
            
            # Bollinger Bands signals
            bb_upper = indicators.get('bb_upper', current_price)
            bb_lower = indicators.get('bb_lower', current_price)
            
            if current_price > bb_upper:
                signals['bollinger'] = 'bearish'  # Overbought
            elif current_price < bb_lower:
                signals['bollinger'] = 'bullish'  # Oversold
            else:
                signals['bollinger'] = 'neutral'
            
        except Exception as e:
            print(f"Error generating signals: {e}")
        
        return signals
    
    def _format_technical_analysis(self, symbol: str, indicators: Dict[str, Any], signals: Dict[str, str]) -> str:
        """Format technical analysis into readable text."""
        try:
            current_price = indicators.get('current_price', 0)
            rsi = indicators.get('rsi', 50)
            
            analysis = f"Technical Analysis for {symbol}:\n\n"
            analysis += f"Current Price: ${current_price:.2f}\n\n"
            
            analysis += "Key Indicators:\n"
            analysis += f"• RSI (14): {rsi:.1f} - {signals.get('rsi', 'neutral').title()}\n"
            analysis += f"• MACD: {signals.get('macd', 'neutral').title()} signal\n"
            analysis += f"• Moving Averages: {signals.get('ma_trend', 'neutral').title()} trend\n"
            analysis += f"• Bollinger Bands: {signals.get('bollinger', 'neutral').title()}\n\n"
            
            # Overall assessment
            bullish_count = sum(1 for signal in signals.values() if signal == 'bullish')
            bearish_count = sum(1 for signal in signals.values() if signal == 'bearish')
            
            if bullish_count > bearish_count:
                analysis += "Overall Assessment: The technical indicators suggest a bullish outlook."
            elif bearish_count > bullish_count:
                analysis += "Overall Assessment: The technical indicators suggest a bearish outlook."
            else:
                analysis += "Overall Assessment: The technical indicators are mixed, suggesting a neutral outlook."
            
            return analysis
            
        except Exception as e:
            print(f"Error formatting technical analysis: {e}")
            return f"Technical analysis for {symbol} is currently unavailable."
    
    async def _openai_explanation(self, data: Dict[str, Any], context: str) -> str:
        """Generate explanation using OpenAI API."""
        try:
            prompt = f"""
            Explain the following financial data in simple, easy-to-understand terms:
            
            Context: {context}
            Data: {json.dumps(data, indent=2)}
            
            Please provide a clear, concise explanation that a beginner investor could understand.
            Focus on what this data means and why it's important.
            """
            
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful financial advisor who explains complex financial concepts in simple terms."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error with OpenAI explanation: {e}")
            return self._rule_based_explanation(data, context)
    
    def _rule_based_explanation(self, data: Dict[str, Any], context: str) -> str:
        """Generate explanation using rule-based approach."""
        try:
            explanation = f"Analysis of {context}:\n\n"
            
            # Handle different data types
            if 'price' in data:
                explanation += f"The current price is ${data['price']:.2f}. "
            
            if 'change' in data:
                change = data['change']
                if change > 0:
                    explanation += f"This represents a positive change of {change:.2f}%, indicating upward momentum. "
                elif change < 0:
                    explanation += f"This represents a negative change of {change:.2f}%, indicating downward pressure. "
                else:
                    explanation += "The price has remained stable with no significant change. "
            
            if 'volume' in data:
                explanation += f"Trading volume is {data['volume']:,}, which indicates the level of market activity. "
            
            if 'rsi' in data:
                rsi = data['rsi']
                if rsi > 70:
                    explanation += "The RSI indicates the asset may be overbought. "
                elif rsi < 30:
                    explanation += "The RSI indicates the asset may be oversold. "
                else:
                    explanation += "The RSI is in a neutral range. "
            
            return explanation
            
        except Exception as e:
            print(f"Error in rule-based explanation: {e}")
            return "Unable to generate explanation for the provided data."
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid."""
        if cache_key not in self.cache:
            return False
        
        cache_time = self.cache[cache_key]['timestamp']
        return (datetime.now() - cache_time).total_seconds() < self.cache_ttl
    
    def _get_mock_analysis(self, request: AnalysisRequest) -> AIInsightResponse:
        """Return mock analysis when real analysis fails."""
        return AIInsightResponse(
            id=f"mock_{request.symbol}_{int(datetime.now().timestamp())}",
            title=f"{request.analysis_type.title()} Analysis: {request.symbol}",
            content=f"Mock {request.analysis_type} analysis for {request.symbol}. This is placeholder content.",
            summary=f"Mock analysis shows neutral outlook for {request.symbol}",
            insight_type="analysis",
            category=request.analysis_type,
            related_symbols=[request.symbol],
            confidence=0.5,
            sentiment="neutral",
            tags=[request.analysis_type, "mock"],
            metadata={"mock": True},
            created_at=datetime.now()
        )
    
    def _get_mock_analysis_response(self, symbol: str, analysis_type: str) -> AIInsightResponse:
        """Return mock analysis response."""
        return AIInsightResponse(
            id=f"mock_{symbol}_{analysis_type}_{int(datetime.now().timestamp())}",
            title=f"{analysis_type.title()} Analysis: {symbol}",
            content=f"Mock {analysis_type} analysis for {symbol}.",
            summary=f"Mock {analysis_type} analysis shows neutral outlook",
            insight_type="analysis",
            category=analysis_type,
            related_symbols=[symbol],
            confidence=0.5,
            sentiment="neutral",
            tags=[analysis_type, "mock"],
            metadata={"mock": True},
            created_at=datetime.now()
        )
    
    def _get_mock_insights(self, category: Optional[str], limit: int) -> List[AIInsightResponse]:
        """Return mock insights when real insights fail."""
        mock_insights = [
            AIInsightResponse(
                id="mock_insight_1",
                title="Market Volatility Alert",
                content="Current market conditions show increased volatility across major indices.",
                summary="Volatility spike detected in major markets",
                insight_type="alert",
                category="market",
                related_symbols=["SPY", "QQQ"],
                confidence=0.8,
                sentiment="neutral",
                tags=["volatility", "market", "alert"],
                metadata={"mock": True},
                created_at=datetime.now()
            ),
            AIInsightResponse(
                id="mock_insight_2",
                title="Bitcoin Technical Breakout",
                content="Bitcoin shows signs of technical breakout above key resistance levels.",
                summary="BTC technical analysis suggests potential upward movement",
                insight_type="analysis",
                category="crypto",
                related_symbols=["BTC-USD"],
                confidence=0.7,
                sentiment="bullish",
                tags=["bitcoin", "technical", "breakout"],
                metadata={"mock": True},
                created_at=datetime.now() - timedelta(hours=1)
            )
        ]
        
        if category:
            mock_insights = [insight for insight in mock_insights if insight.category == category]
        
        return mock_insights[:limit]
    
    async def _get_fundamental_data(self, symbol: str) -> Dict[str, Any]:
        """Get fundamental data for analysis (mock implementation)."""
        # In a real implementation, this would fetch from financial APIs
        return {
            "pe_ratio": 25.5,
            "market_cap": 2500000000,
            "revenue_growth": 0.15,
            "profit_margin": 0.12,
            "debt_to_equity": 0.3,
            "return_on_equity": 0.18
        }
    
    def _analyze_fundamentals(self, fundamentals: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze fundamental data."""
        score = 0
        max_score = 6
        
        # PE Ratio (lower is better, but not too low)
        pe = fundamentals.get('pe_ratio', 20)
        if 10 <= pe <= 25:
            score += 1
        
        # Revenue Growth (higher is better)
        revenue_growth = fundamentals.get('revenue_growth', 0)
        if revenue_growth > 0.1:
            score += 1
        
        # Profit Margin (higher is better)
        profit_margin = fundamentals.get('profit_margin', 0)
        if profit_margin > 0.1:
            score += 1
        
        # Debt to Equity (lower is better)
        debt_to_equity = fundamentals.get('debt_to_equity', 1)
        if debt_to_equity < 0.5:
            score += 1
        
        # Return on Equity (higher is better)
        roe = fundamentals.get('return_on_equity', 0)
        if roe > 0.15:
            score += 1
        
        # Market Cap (stability indicator)
        market_cap = fundamentals.get('market_cap', 0)
        if market_cap > 1000000000:  # > 1B
            score += 1
        
        # Determine rating
        score_ratio = score / max_score
        if score_ratio >= 0.8:
            rating = "Strong Buy"
            sentiment = "bullish"
        elif score_ratio >= 0.6:
            rating = "Buy"
            sentiment = "bullish"
        elif score_ratio >= 0.4:
            rating = "Hold"
            sentiment = "neutral"
        elif score_ratio >= 0.2:
            rating = "Sell"
            sentiment = "bearish"
        else:
            rating = "Strong Sell"
            sentiment = "bearish"
        
        return {
            "overall_rating": rating,
            "sentiment": sentiment,
            "confidence": min(0.9, 0.5 + score_ratio * 0.4),
            "score": score,
            "max_score": max_score
        }
    
    def _format_fundamental_analysis(self, symbol: str, fundamentals: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        """Format fundamental analysis into readable text."""
        text = f"Fundamental Analysis for {symbol}:\n\n"
        text += f"Overall Rating: {analysis['overall_rating']}\n\n"
        text += "Key Metrics:\n"
        text += f"• P/E Ratio: {fundamentals.get('pe_ratio', 'N/A')}\n"
        text += f"• Market Cap: ${fundamentals.get('market_cap', 0):,.0f}\n"
        text += f"• Revenue Growth: {fundamentals.get('revenue_growth', 0):.1%}\n"
        text += f"• Profit Margin: {fundamentals.get('profit_margin', 0):.1%}\n"
        text += f"• Debt-to-Equity: {fundamentals.get('debt_to_equity', 0):.2f}\n"
        text += f"• Return on Equity: {fundamentals.get('return_on_equity', 0):.1%}\n\n"
        text += f"Score: {analysis['score']}/{analysis['max_score']}\n"
        return text
    
    async def _get_sentiment_data(self, symbol: str) -> Dict[str, Any]:
        """Get sentiment data (mock implementation)."""
        return {
            "news_sentiment": 0.3,
            "social_sentiment": 0.1,
            "analyst_sentiment": 0.5,
            "news_count": 25,
            "social_mentions": 150,
            "analyst_ratings": {"buy": 8, "hold": 5, "sell": 2}
        }
    
    def _analyze_sentiment_data(self, sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sentiment data."""
        news_sentiment = sentiment_data.get('news_sentiment', 0)
        social_sentiment = sentiment_data.get('social_sentiment', 0)
        analyst_sentiment = sentiment_data.get('analyst_sentiment', 0)
        
        # Weighted average
        overall_sentiment_score = (news_sentiment * 0.4 + social_sentiment * 0.3 + analyst_sentiment * 0.3)
        
        if overall_sentiment_score > 0.2:
            overall_sentiment = "bullish"
        elif overall_sentiment_score < -0.2:
            overall_sentiment = "bearish"
        else:
            overall_sentiment = "neutral"
        
        confidence = min(0.9, 0.5 + abs(overall_sentiment_score) * 0.5)
        
        return {
            "overall_sentiment": overall_sentiment,
            "sentiment_score": overall_sentiment_score,
            "confidence": confidence
        }
    
    def _format_sentiment_analysis(self, symbol: str, sentiment_data: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        """Format sentiment analysis into readable text."""
        text = f"Sentiment Analysis for {symbol}:\n\n"
        text += f"Overall Sentiment: {analysis['overall_sentiment'].title()}\n"
        text += f"Sentiment Score: {analysis['sentiment_score']:.2f}\n\n"
        text += "Component Sentiments:\n"
        text += f"• News Sentiment: {sentiment_data.get('news_sentiment', 0):.2f}\n"
        text += f"• Social Media Sentiment: {sentiment_data.get('social_sentiment', 0):.2f}\n"
        text += f"• Analyst Sentiment: {sentiment_data.get('analyst_sentiment', 0):.2f}\n\n"
        text += f"Data Points: {sentiment_data.get('news_count', 0)} news articles, {sentiment_data.get('social_mentions', 0)} social mentions\n"
        return text
    
    def _prepare_ml_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML models."""
        features = data.copy()
        
        # Technical indicators as features
        features['sma_20'] = ta.trend.sma_indicator(features['Close'], window=20)
        features['rsi'] = ta.momentum.rsi(features['Close'], window=14)
        features['macd'] = ta.trend.macd(features['Close'])
        
        # Price-based features
        features['price_change'] = features['Close'].pct_change()
        features['volume_change'] = features['Volume'].pct_change()
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            features[f'close_lag_{lag}'] = features['Close'].shift(lag)
            features[f'volume_lag_{lag}'] = features['Volume'].shift(lag)
        
        # Drop NaN values
        features = features.dropna()
        
        return features
    
    async def _get_ml_model(self, symbol: str, features: pd.DataFrame) -> Any:
        """Get or train ML model for predictions."""
        model_key = f"model_{symbol}"
        
        if model_key in self.models_cache:
            return self.models_cache[model_key]
        
        # Prepare training data
        X = features[['sma_20', 'rsi', 'macd', 'price_change', 'volume_change']].fillna(0)
        y = features['Close'].shift(-1).fillna(features['Close'].iloc[-1])  # Next day's price
        
        # Train simple model
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X[:-1], y[:-1])  # Exclude last row as it has no target
        
        # Cache the model
        self.models_cache[model_key] = model
        
        return model
    
    def _make_predictions(self, model: Any, features: pd.DataFrame) -> Dict[str, Any]:
        """Make predictions using the ML model."""
        try:
            # Prepare last row for prediction
            X_last = features[['sma_20', 'rsi', 'macd', 'price_change', 'volume_change']].fillna(0).iloc[-1:]
            
            # Make prediction
            predicted_price = model.predict(X_last)[0]
            current_price = features['Close'].iloc[-1]
            
            # Calculate direction and confidence
            price_change = (predicted_price - current_price) / current_price
            
            if price_change > 0.02:
                direction = "upward"
                sentiment = "bullish"
            elif price_change < -0.02:
                direction = "downward"
                sentiment = "bearish"
            else:
                direction = "sideways"
                sentiment = "neutral"
            
            confidence = min(0.8, 0.5 + abs(price_change) * 2)
            
            return {
                "predicted_price": predicted_price,
                "current_price": current_price,
                "price_change": price_change,
                "direction": direction,
                "sentiment": sentiment,
                "confidence": confidence
            }
            
        except Exception as e:
            print(f"Error making predictions: {e}")
            return {
                "predicted_price": features['Close'].iloc[-1],
                "current_price": features['Close'].iloc[-1],
                "price_change": 0,
                "direction": "sideways",
                "sentiment": "neutral",
                "confidence": 0.5
            }
    
    def _format_prediction_analysis(self, symbol: str, predictions: Dict[str, Any]) -> str:
        """Format prediction analysis into readable text."""
        text = f"Price Prediction for {symbol}:\n\n"
        text += f"Current Price: ${predictions['current_price']:.2f}\n"
        text += f"Predicted Price: ${predictions['predicted_price']:.2f}\n"
        text += f"Expected Change: {predictions['price_change']:.2%}\n"
        text += f"Direction: {predictions['direction'].title()}\n"
        text += f"Confidence: {predictions['confidence']:.1%}\n\n"
        text += "This prediction is based on machine learning analysis of historical price patterns and technical indicators."
        return text
    
    def _combine_analyses(self, analyses: List[AIInsightResponse]) -> Dict[str, Any]:
        """Combine multiple analyses into a comprehensive view."""
        # Calculate weighted sentiment
        sentiments = [analysis.sentiment for analysis in analyses]
        confidences = [analysis.confidence for analysis in analyses]
        
        # Weight sentiments by confidence
        weighted_sentiment_score = 0
        total_weight = 0
        
        for sentiment, confidence in zip(sentiments, confidences):
            if sentiment == "bullish":
                score = 1
            elif sentiment == "bearish":
                score = -1
            else:
                score = 0
            
            weighted_sentiment_score += score * confidence
            total_weight += confidence
        
        if total_weight > 0:
            avg_sentiment_score = weighted_sentiment_score / total_weight
        else:
            avg_sentiment_score = 0
        
        # Determine overall sentiment
        if avg_sentiment_score > 0.2:
            overall_sentiment = "bullish"
        elif avg_sentiment_score < -0.2:
            overall_sentiment = "bearish"
        else:
            overall_sentiment = "neutral"
        
        # Calculate overall confidence
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        # Create combined content
        content = "Comprehensive Analysis Summary:\n\n"
        for analysis in analyses:
            content += f"{analysis.category.title()} Analysis: {analysis.sentiment.title()} (Confidence: {analysis.confidence:.1%})\n"
        
        content += f"\nOverall Assessment: {overall_sentiment.title()} with {overall_confidence:.1%} confidence"
        
        return {
            "content": content,
            "summary": f"Comprehensive analysis shows {overall_sentiment} outlook",
            "sentiment": overall_sentiment,
            "confidence": overall_confidence
        }
    
    async def _generate_market_insights(self) -> List[AIInsightResponse]:
        """Generate market-wide insights."""
        insights = []
        
        # Mock market insight
        market_insight = AIInsightResponse(
            id=f"market_{int(datetime.now().timestamp())}",
            title="Market Overview: Mixed Signals Across Sectors",
            content="Current market conditions show mixed signals with technology stocks showing strength while energy sector faces headwinds.",
            summary="Mixed market conditions with sector rotation evident",
            insight_type="analysis",
            category="market",
            related_symbols=["SPY", "QQQ", "XLE"],
            confidence=0.7,
            sentiment="neutral",
            tags=["market", "sectors", "rotation"],
            metadata={"market_wide": True},
            created_at=datetime.now()
        )
        
        insights.append(market_insight)
        
        return insights