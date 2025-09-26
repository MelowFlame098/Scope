"""NFT Investment Analyzer

This module provides comprehensive NFT investment analysis capabilities,
including market trend analysis, rarity scoring, price predictions,
portfolio optimization, and risk assessment for NFT investments.

Features:
- NFT collection analysis
- Rarity scoring and ranking
- Price prediction models
- Market trend analysis
- Portfolio optimization
- Risk assessment
- Liquidity analysis
- Cross-marketplace data
- Historical performance tracking
- Investment recommendations

Supported Marketplaces:
- OpenSea
- LooksRare
- X2Y2
- Foundation
- SuperRare
- Async Art
- KnownOrigin
- MakersPlace

Author: FinScope AI Team
Version: 1.0.0
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime, timedelta
from collections import defaultdict
import statistics
import math
from abc import ABC, abstractmethod

class NFTCategory(Enum):
    """NFT categories"""
    ART = "art"
    COLLECTIBLES = "collectibles"
    GAMING = "gaming"
    METAVERSE = "metaverse"
    MUSIC = "music"
    SPORTS = "sports"
    UTILITY = "utility"
    PHOTOGRAPHY = "photography"
    DOMAIN_NAMES = "domain_names"
    VIRTUAL_WORLDS = "virtual_worlds"
    TRADING_CARDS = "trading_cards"
    MEMES = "memes"

class RarityTier(Enum):
    """NFT rarity tiers"""
    COMMON = "common"
    UNCOMMON = "uncommon"
    RARE = "rare"
    EPIC = "epic"
    LEGENDARY = "legendary"
    MYTHIC = "mythic"

class MarketTrend(Enum):
    """Market trend directions"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"

class InvestmentRisk(Enum):
    """Investment risk levels"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class NFTMetadata:
    """NFT metadata information"""
    token_id: str
    name: str
    description: str
    image_url: str
    attributes: List[Dict[str, Any]]
    collection_name: str
    creator: str
    owner: str
    contract_address: str
    blockchain: str = "ethereum"
    metadata_url: Optional[str] = None
    animation_url: Optional[str] = None
    external_url: Optional[str] = None

@dataclass
class NFTMarketData:
    """NFT market data"""
    token_id: str
    contract_address: str
    current_price_eth: Optional[Decimal]
    current_price_usd: Optional[Decimal]
    last_sale_price_eth: Optional[Decimal]
    last_sale_price_usd: Optional[Decimal]
    floor_price_eth: Decimal
    floor_price_usd: Decimal
    volume_24h_eth: Decimal
    volume_24h_usd: Decimal
    sales_count_24h: int
    listing_count: int
    highest_offer_eth: Optional[Decimal]
    highest_offer_usd: Optional[Decimal]
    last_updated: datetime
    marketplace: str
    is_listed: bool = False
    listing_expiry: Optional[datetime] = None

@dataclass
class RarityScore:
    """NFT rarity scoring"""
    token_id: str
    overall_rarity_score: float
    rarity_rank: int
    rarity_tier: RarityTier
    trait_scores: Dict[str, float]
    trait_rarities: Dict[str, float]  # Percentage rarity for each trait
    statistical_rarity: float  # Based on trait frequency
    market_rarity: float  # Based on market performance
    collection_size: int
    percentile: float  # Top X% of collection
    calculated_at: datetime = field(default_factory=datetime.now)

@dataclass
class PricePrediction:
    """NFT price prediction"""
    token_id: str
    predicted_price_eth: Decimal
    predicted_price_usd: Decimal
    confidence_score: float  # 0-1
    prediction_timeframe: str  # "7d", "30d", "90d"
    factors: Dict[str, float]  # Contributing factors and weights
    price_range_low_eth: Decimal
    price_range_high_eth: Decimal
    price_range_low_usd: Decimal
    price_range_high_usd: Decimal
    model_used: str
    predicted_at: datetime = field(default_factory=datetime.now)

@dataclass
class CollectionAnalysis:
    """NFT collection analysis"""
    contract_address: str
    collection_name: str
    total_supply: int
    unique_holders: int
    floor_price_eth: Decimal
    floor_price_usd: Decimal
    market_cap_eth: Decimal
    market_cap_usd: Decimal
    volume_24h_eth: Decimal
    volume_24h_usd: Decimal
    volume_7d_eth: Decimal
    volume_7d_usd: Decimal
    volume_30d_eth: Decimal
    volume_30d_usd: Decimal
    sales_count_24h: int
    sales_count_7d: int
    sales_count_30d: int
    average_price_eth: Decimal
    average_price_usd: Decimal
    price_volatility: float
    liquidity_score: float  # 0-10
    trend: MarketTrend
    category: NFTCategory
    risk_level: InvestmentRisk
    social_metrics: Dict[str, Any]
    technical_metrics: Dict[str, Any]
    analyzed_at: datetime = field(default_factory=datetime.now)

@dataclass
class NFTPortfolioItem:
    """NFT portfolio item"""
    token_id: str
    contract_address: str
    collection_name: str
    name: str
    purchase_price_eth: Decimal
    purchase_price_usd: Decimal
    current_value_eth: Decimal
    current_value_usd: Decimal
    unrealized_pnl_eth: Decimal
    unrealized_pnl_usd: Decimal
    unrealized_pnl_percentage: float
    rarity_score: Optional[RarityScore]
    market_data: Optional[NFTMarketData]
    purchase_date: datetime
    holding_period_days: int
    estimated_liquidity_days: int  # Days to sell at current price
    risk_score: float  # 1-10
    recommendation: str  # "hold", "sell", "buy_more"

@dataclass
class InvestmentRecommendation:
    """NFT investment recommendation"""
    token_id: str
    contract_address: str
    collection_name: str
    name: str
    recommendation_type: str  # "buy", "sell", "hold", "avoid"
    confidence_score: float  # 0-1
    target_price_eth: Optional[Decimal]
    target_price_usd: Optional[Decimal]
    expected_return_percentage: float
    risk_level: InvestmentRisk
    investment_horizon: str  # "short", "medium", "long"
    reasoning: List[str]
    supporting_factors: Dict[str, Any]
    risk_factors: List[str]
    market_conditions: Dict[str, Any]
    generated_at: datetime = field(default_factory=datetime.now)

class NFTAnalyzer:
    """NFT Investment Analyzer
    
    Provides comprehensive NFT investment analysis including rarity scoring,
    price predictions, market analysis, and portfolio optimization.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize NFT Analyzer
        
        Args:
            config: Configuration dictionary containing:
                - api_keys: API keys for various NFT marketplaces
                - analysis_settings: Analysis parameters
                - prediction_models: ML model configurations
                - risk_settings: Risk assessment parameters
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.NFTAnalyzer")
        
        # API configurations
        self.api_keys = config.get('api_keys', {})
        self.marketplace_endpoints = {
            'opensea': 'https://api.opensea.io/api/v1',
            'looksrare': 'https://api.looksrare.org/api/v1',
            'x2y2': 'https://api.x2y2.org/api/v1',
            'foundation': 'https://api.foundation.app/v1',
            'superrare': 'https://superrare.com/api/v1'
        }
        
        # Analysis settings
        self.analysis_settings = config.get('analysis_settings', {
            'rarity_calculation_method': 'statistical',  # 'statistical', 'market_based', 'hybrid'
            'price_prediction_timeframes': ['7d', '30d', '90d'],
            'min_collection_size': 100,
            'min_trading_volume': 1.0,  # ETH
            'liquidity_threshold': 0.1,  # 10% of floor price
            'volatility_window_days': 30
        })
        
        # Prediction models
        self.prediction_models = config.get('prediction_models', {
            'price_model': 'ensemble',  # 'linear', 'random_forest', 'neural_network', 'ensemble'
            'trend_model': 'technical_analysis',
            'rarity_model': 'trait_frequency'
        })
        
        # Risk settings
        self.risk_settings = config.get('risk_settings', {
            'max_position_size_percentage': 10,  # 10% of portfolio
            'max_collection_exposure': 25,  # 25% in single collection
            'min_liquidity_score': 5,
            'max_volatility': 50  # 50%
        })
        
        # Data caches
        self.collection_cache: Dict[str, CollectionAnalysis] = {}
        self.rarity_cache: Dict[str, RarityScore] = {}
        self.price_cache: Dict[str, PricePrediction] = {}
        self.market_data_cache: Dict[str, NFTMarketData] = {}
        
        # Cache TTL
        self.cache_ttl = timedelta(minutes=15)
        
        self.logger.info("NFT Analyzer initialized")
    
    async def analyze_collection(self, contract_address: str) -> CollectionAnalysis:
        """Analyze an NFT collection
        
        Args:
            contract_address: Contract address of the NFT collection
            
        Returns:
            Collection analysis data
        """
        try:
            # Check cache first
            if contract_address in self.collection_cache:
                cached_analysis = self.collection_cache[contract_address]
                if datetime.now() - cached_analysis.analyzed_at < self.cache_ttl:
                    return cached_analysis
            
            # Mock collection analysis (in real implementation, this would fetch from APIs)
            analysis = CollectionAnalysis(
                contract_address=contract_address,
                collection_name="Bored Ape Yacht Club",
                total_supply=10000,
                unique_holders=6500,
                floor_price_eth=Decimal('45.5'),
                floor_price_usd=Decimal('91000'),
                market_cap_eth=Decimal('455000'),
                market_cap_usd=Decimal('910000000'),
                volume_24h_eth=Decimal('125.5'),
                volume_24h_usd=Decimal('251000'),
                volume_7d_eth=Decimal('890.2'),
                volume_7d_usd=Decimal('1780400'),
                volume_30d_eth=Decimal('3456.8'),
                volume_30d_usd=Decimal('6913600'),
                sales_count_24h=15,
                sales_count_7d=95,
                sales_count_30d=380,
                average_price_eth=Decimal('52.3'),
                average_price_usd=Decimal('104600'),
                price_volatility=0.25,  # 25%
                liquidity_score=8.5,
                trend=MarketTrend.SIDEWAYS,
                category=NFTCategory.COLLECTIBLES,
                risk_level=InvestmentRisk.MEDIUM,
                social_metrics={
                    'twitter_followers': 450000,
                    'discord_members': 180000,
                    'social_sentiment': 0.65,  # 0-1 scale
                    'influencer_mentions': 25
                },
                technical_metrics={
                    'holder_distribution_gini': 0.45,  # Wealth distribution
                    'whale_concentration': 0.15,  # Top 10 holders percentage
                    'trading_velocity': 0.08,  # Daily volume / market cap
                    'price_support_levels': [40.0, 35.0, 30.0],
                    'price_resistance_levels': [55.0, 60.0, 70.0]
                }
            )
            
            # Cache the analysis
            self.collection_cache[contract_address] = analysis
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing collection {contract_address}: {e}")
            raise
    
    async def calculate_rarity_score(self, token_id: str, contract_address: str) -> RarityScore:
        """Calculate rarity score for an NFT
        
        Args:
            token_id: Token ID of the NFT
            contract_address: Contract address of the collection
            
        Returns:
            Rarity score data
        """
        try:
            cache_key = f"{contract_address}:{token_id}"
            
            # Check cache first
            if cache_key in self.rarity_cache:
                cached_score = self.rarity_cache[cache_key]
                if datetime.now() - cached_score.calculated_at < self.cache_ttl:
                    return cached_score
            
            # Mock rarity calculation (in real implementation, this would analyze traits)
            trait_scores = {
                'background': 0.15,  # 15% have this background
                'fur': 0.08,  # 8% have this fur type
                'eyes': 0.25,  # 25% have these eyes
                'mouth': 0.12,  # 12% have this mouth
                'clothes': 0.05,  # 5% have these clothes (rare)
                'hat': 0.03  # 3% have this hat (very rare)
            }
            
            trait_rarities = {trait: score * 100 for trait, score in trait_scores.items()}
            
            # Calculate overall rarity score using harmonic mean
            harmonic_mean = len(trait_scores) / sum(1/score for score in trait_scores.values())
            statistical_rarity = 1 / harmonic_mean
            
            # Mock market-based rarity (based on trading patterns)
            market_rarity = 0.85  # High market demand
            
            # Combined rarity score
            overall_rarity_score = (statistical_rarity * 0.7) + (market_rarity * 0.3)
            
            # Determine rarity tier
            if overall_rarity_score >= 0.95:
                rarity_tier = RarityTier.MYTHIC
            elif overall_rarity_score >= 0.85:
                rarity_tier = RarityTier.LEGENDARY
            elif overall_rarity_score >= 0.70:
                rarity_tier = RarityTier.EPIC
            elif overall_rarity_score >= 0.50:
                rarity_tier = RarityTier.RARE
            elif overall_rarity_score >= 0.25:
                rarity_tier = RarityTier.UNCOMMON
            else:
                rarity_tier = RarityTier.COMMON
            
            rarity_score = RarityScore(
                token_id=token_id,
                overall_rarity_score=overall_rarity_score,
                rarity_rank=250,  # Rank 250 out of 10000
                rarity_tier=rarity_tier,
                trait_scores=trait_scores,
                trait_rarities=trait_rarities,
                statistical_rarity=statistical_rarity,
                market_rarity=market_rarity,
                collection_size=10000,
                percentile=97.5  # Top 2.5%
            )
            
            # Cache the score
            self.rarity_cache[cache_key] = rarity_score
            
            return rarity_score
            
        except Exception as e:
            self.logger.error(f"Error calculating rarity score for {contract_address}:{token_id}: {e}")
            raise
    
    async def predict_price(self, token_id: str, contract_address: str, timeframe: str = "30d") -> PricePrediction:
        """Predict NFT price
        
        Args:
            token_id: Token ID of the NFT
            contract_address: Contract address of the collection
            timeframe: Prediction timeframe ("7d", "30d", "90d")
            
        Returns:
            Price prediction data
        """
        try:
            cache_key = f"{contract_address}:{token_id}:{timeframe}"
            
            # Check cache first
            if cache_key in self.price_cache:
                cached_prediction = self.price_cache[cache_key]
                if datetime.now() - cached_prediction.predicted_at < self.cache_ttl:
                    return cached_prediction
            
            # Get collection analysis and rarity score
            collection_analysis = await self.analyze_collection(contract_address)
            rarity_score = await self.calculate_rarity_score(token_id, contract_address)
            
            # Mock price prediction model
            base_price = collection_analysis.floor_price_eth
            rarity_multiplier = 1 + (rarity_score.overall_rarity_score * 2)  # Up to 3x for mythic
            
            # Market trend adjustment
            trend_multiplier = {
                MarketTrend.BULLISH: 1.15,
                MarketTrend.SIDEWAYS: 1.0,
                MarketTrend.BEARISH: 0.85,
                MarketTrend.VOLATILE: 0.95
            }.get(collection_analysis.trend, 1.0)
            
            # Timeframe adjustment
            timeframe_multiplier = {
                "7d": 1.02,
                "30d": 1.08,
                "90d": 1.15
            }.get(timeframe, 1.0)
            
            predicted_price_eth = base_price * rarity_multiplier * trend_multiplier * timeframe_multiplier
            predicted_price_usd = predicted_price_eth * Decimal('2000')  # Mock ETH price
            
            # Calculate confidence score
            confidence_factors = {
                'collection_volume': min(1.0, float(collection_analysis.volume_30d_eth) / 1000),
                'rarity_confidence': rarity_score.overall_rarity_score,
                'market_stability': 1 - collection_analysis.price_volatility,
                'liquidity': collection_analysis.liquidity_score / 10
            }
            confidence_score = sum(confidence_factors.values()) / len(confidence_factors)
            
            # Price range (±20% based on volatility)
            volatility_factor = Decimal(str(collection_analysis.price_volatility))
            price_range_low_eth = predicted_price_eth * (1 - volatility_factor)
            price_range_high_eth = predicted_price_eth * (1 + volatility_factor)
            price_range_low_usd = price_range_low_eth * Decimal('2000')
            price_range_high_usd = price_range_high_eth * Decimal('2000')
            
            prediction = PricePrediction(
                token_id=token_id,
                predicted_price_eth=predicted_price_eth,
                predicted_price_usd=predicted_price_usd,
                confidence_score=confidence_score,
                prediction_timeframe=timeframe,
                factors={
                    'rarity_impact': float(rarity_multiplier - 1),
                    'market_trend_impact': float(trend_multiplier - 1),
                    'time_appreciation': float(timeframe_multiplier - 1),
                    'collection_strength': collection_analysis.liquidity_score / 10
                },
                price_range_low_eth=price_range_low_eth,
                price_range_high_eth=price_range_high_eth,
                price_range_low_usd=price_range_low_usd,
                price_range_high_usd=price_range_high_usd,
                model_used="ensemble_v1"
            )
            
            # Cache the prediction
            self.price_cache[cache_key] = prediction
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error predicting price for {contract_address}:{token_id}: {e}")
            raise
    
    async def analyze_portfolio(self, user_address: str, nft_holdings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze NFT portfolio
        
        Args:
            user_address: User's wallet address
            nft_holdings: List of NFT holdings with basic info
            
        Returns:
            Portfolio analysis data
        """
        try:
            portfolio_items = []
            total_value_eth = Decimal('0')
            total_value_usd = Decimal('0')
            total_cost_eth = Decimal('0')
            total_cost_usd = Decimal('0')
            
            collection_exposure = defaultdict(Decimal)
            category_exposure = defaultdict(Decimal)
            risk_distribution = defaultdict(int)
            
            for holding in nft_holdings:
                try:
                    token_id = holding['token_id']
                    contract_address = holding['contract_address']
                    purchase_price_eth = Decimal(str(holding.get('purchase_price_eth', 0)))
                    purchase_price_usd = Decimal(str(holding.get('purchase_price_usd', 0)))
                    purchase_date = datetime.fromisoformat(holding.get('purchase_date', datetime.now().isoformat()))
                    
                    # Get current market data and analysis
                    collection_analysis = await self.analyze_collection(contract_address)
                    rarity_score = await self.calculate_rarity_score(token_id, contract_address)
                    
                    # Calculate current value (simplified)
                    base_value = collection_analysis.floor_price_eth
                    rarity_multiplier = 1 + (rarity_score.overall_rarity_score * 1.5)
                    current_value_eth = base_value * Decimal(str(rarity_multiplier))
                    current_value_usd = current_value_eth * Decimal('2000')
                    
                    # Calculate P&L
                    unrealized_pnl_eth = current_value_eth - purchase_price_eth
                    unrealized_pnl_usd = current_value_usd - purchase_price_usd
                    unrealized_pnl_percentage = float((unrealized_pnl_eth / purchase_price_eth) * 100) if purchase_price_eth > 0 else 0
                    
                    # Calculate holding period
                    holding_period_days = (datetime.now() - purchase_date).days
                    
                    # Estimate liquidity (days to sell)
                    daily_sales = collection_analysis.sales_count_24h
                    estimated_liquidity_days = max(1, int(collection_analysis.total_supply / (daily_sales * 365)) if daily_sales > 0 else 30)
                    
                    # Risk score calculation
                    risk_factors = {
                        'collection_volatility': collection_analysis.price_volatility,
                        'liquidity_risk': 1 - (collection_analysis.liquidity_score / 10),
                        'market_trend_risk': 0.3 if collection_analysis.trend == MarketTrend.BEARISH else 0.1,
                        'concentration_risk': min(0.5, float(current_value_eth) / float(total_value_eth + current_value_eth))
                    }
                    risk_score = sum(risk_factors.values()) / len(risk_factors) * 10
                    
                    # Generate recommendation
                    if unrealized_pnl_percentage > 50 and risk_score > 7:
                        recommendation = "sell"
                    elif unrealized_pnl_percentage < -30 and rarity_score.overall_rarity_score > 0.8:
                        recommendation = "hold"
                    elif collection_analysis.trend == MarketTrend.BULLISH and rarity_score.overall_rarity_score > 0.6:
                        recommendation = "buy_more"
                    else:
                        recommendation = "hold"
                    
                    portfolio_item = NFTPortfolioItem(
                        token_id=token_id,
                        contract_address=contract_address,
                        collection_name=collection_analysis.collection_name,
                        name=holding.get('name', f"#{token_id}"),
                        purchase_price_eth=purchase_price_eth,
                        purchase_price_usd=purchase_price_usd,
                        current_value_eth=current_value_eth,
                        current_value_usd=current_value_usd,
                        unrealized_pnl_eth=unrealized_pnl_eth,
                        unrealized_pnl_usd=unrealized_pnl_usd,
                        unrealized_pnl_percentage=unrealized_pnl_percentage,
                        rarity_score=rarity_score,
                        market_data=None,  # Would be populated with real market data
                        purchase_date=purchase_date,
                        holding_period_days=holding_period_days,
                        estimated_liquidity_days=estimated_liquidity_days,
                        risk_score=risk_score,
                        recommendation=recommendation
                    )
                    
                    portfolio_items.append(portfolio_item)
                    
                    # Update totals
                    total_value_eth += current_value_eth
                    total_value_usd += current_value_usd
                    total_cost_eth += purchase_price_eth
                    total_cost_usd += purchase_price_usd
                    
                    # Update exposures
                    collection_exposure[collection_analysis.collection_name] += current_value_eth
                    category_exposure[collection_analysis.category.value] += current_value_eth
                    risk_distribution[collection_analysis.risk_level.value] += 1
                    
                except Exception as e:
                    self.logger.warning(f"Error analyzing portfolio item {holding}: {e}")
                    continue
            
            # Calculate portfolio metrics
            total_pnl_eth = total_value_eth - total_cost_eth
            total_pnl_usd = total_value_usd - total_cost_usd
            total_pnl_percentage = float((total_pnl_eth / total_cost_eth) * 100) if total_cost_eth > 0 else 0
            
            # Portfolio diversification metrics
            collection_count = len(collection_exposure)
            max_collection_exposure = max(collection_exposure.values()) if collection_exposure else Decimal('0')
            max_collection_percentage = float((max_collection_exposure / total_value_eth) * 100) if total_value_eth > 0 else 0
            
            # Risk metrics
            avg_risk_score = statistics.mean([item.risk_score for item in portfolio_items]) if portfolio_items else 0
            portfolio_volatility = statistics.stdev([item.unrealized_pnl_percentage for item in portfolio_items]) if len(portfolio_items) > 1 else 0
            
            # Liquidity metrics
            avg_liquidity_days = statistics.mean([item.estimated_liquidity_days for item in portfolio_items]) if portfolio_items else 0
            
            portfolio_analysis = {
                'user_address': user_address,
                'total_items': len(portfolio_items),
                'total_value_eth': float(total_value_eth),
                'total_value_usd': float(total_value_usd),
                'total_cost_eth': float(total_cost_eth),
                'total_cost_usd': float(total_cost_usd),
                'total_pnl_eth': float(total_pnl_eth),
                'total_pnl_usd': float(total_pnl_usd),
                'total_pnl_percentage': total_pnl_percentage,
                'portfolio_items': [
                    {
                        'token_id': item.token_id,
                        'contract_address': item.contract_address,
                        'collection_name': item.collection_name,
                        'name': item.name,
                        'current_value_eth': float(item.current_value_eth),
                        'current_value_usd': float(item.current_value_usd),
                        'unrealized_pnl_percentage': item.unrealized_pnl_percentage,
                        'rarity_tier': item.rarity_score.rarity_tier.value if item.rarity_score else 'unknown',
                        'risk_score': item.risk_score,
                        'recommendation': item.recommendation,
                        'holding_period_days': item.holding_period_days
                    } for item in portfolio_items
                ],
                'diversification_metrics': {
                    'collection_count': collection_count,
                    'max_collection_exposure_percentage': max_collection_percentage,
                    'collection_exposure': {k: float(v) for k, v in collection_exposure.items()},
                    'category_exposure': {k: float(v) for k, v in category_exposure.items()}
                },
                'risk_metrics': {
                    'average_risk_score': avg_risk_score,
                    'portfolio_volatility': portfolio_volatility,
                    'risk_distribution': dict(risk_distribution)
                },
                'liquidity_metrics': {
                    'average_liquidity_days': avg_liquidity_days,
                    'highly_liquid_items': len([item for item in portfolio_items if item.estimated_liquidity_days <= 7]),
                    'illiquid_items': len([item for item in portfolio_items if item.estimated_liquidity_days > 30])
                },
                'recommendations': {
                    'sell_count': len([item for item in portfolio_items if item.recommendation == 'sell']),
                    'hold_count': len([item for item in portfolio_items if item.recommendation == 'hold']),
                    'buy_more_count': len([item for item in portfolio_items if item.recommendation == 'buy_more'])
                },
                'analyzed_at': datetime.now().isoformat()
            }
            
            return portfolio_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing NFT portfolio: {e}")
            return {'error': str(e)}
    
    async def get_investment_recommendations(self, 
                                           user_preferences: Dict[str, Any],
                                           budget_eth: float,
                                           risk_tolerance: str = "medium") -> List[InvestmentRecommendation]:
        """Get NFT investment recommendations
        
        Args:
            user_preferences: User preferences (categories, price range, etc.)
            budget_eth: Investment budget in ETH
            risk_tolerance: Risk tolerance level
            
        Returns:
            List of investment recommendations
        """
        try:
            recommendations = []
            
            # Mock collections to analyze
            collections_to_analyze = [
                {
                    'contract_address': '0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13D',
                    'name': 'Bored Ape Yacht Club',
                    'sample_tokens': ['1234', '5678', '9012']
                },
                {
                    'contract_address': '0x60E4d786628Fea6478F785A6d7e704777c86a7c6',
                    'name': 'Mutant Ape Yacht Club',
                    'sample_tokens': ['3456', '7890', '1357']
                },
                {
                    'contract_address': '0xb47e3cd837dDF8e4c57F05d70Ab865de6e193BBB',
                    'name': 'CryptoPunks',
                    'sample_tokens': ['2468', '8024', '4680']
                }
            ]
            
            risk_multipliers = {
                'low': 0.7,
                'medium': 1.0,
                'high': 1.5
            }
            risk_multiplier = risk_multipliers.get(risk_tolerance, 1.0)
            
            for collection in collections_to_analyze:
                try:
                    collection_analysis = await self.analyze_collection(collection['contract_address'])
                    
                    # Filter by user preferences
                    if 'categories' in user_preferences:
                        if collection_analysis.category.value not in user_preferences['categories']:
                            continue
                    
                    # Filter by budget
                    if collection_analysis.floor_price_eth > Decimal(str(budget_eth * risk_multiplier)):
                        continue
                    
                    # Analyze sample tokens from the collection
                    for token_id in collection['sample_tokens'][:2]:  # Analyze 2 tokens per collection
                        try:
                            rarity_score = await self.calculate_rarity_score(token_id, collection['contract_address'])
                            price_prediction = await self.predict_price(token_id, collection['contract_address'], '30d')
                            
                            # Calculate expected return
                            current_price = collection_analysis.floor_price_eth * Decimal(str(1 + rarity_score.overall_rarity_score))
                            expected_return_percentage = float(((price_prediction.predicted_price_eth - current_price) / current_price) * 100)
                            
                            # Determine recommendation type
                            if (expected_return_percentage > 20 and 
                                collection_analysis.risk_level.value in ['low', 'medium'] and
                                rarity_score.overall_rarity_score > 0.6):
                                recommendation_type = "buy"
                            elif expected_return_percentage > 10:
                                recommendation_type = "hold"
                            elif expected_return_percentage < -10:
                                recommendation_type = "avoid"
                            else:
                                recommendation_type = "hold"
                            
                            # Generate reasoning
                            reasoning = []
                            if rarity_score.rarity_tier in [RarityTier.LEGENDARY, RarityTier.MYTHIC]:
                                reasoning.append(f"High rarity tier: {rarity_score.rarity_tier.value}")
                            if collection_analysis.trend == MarketTrend.BULLISH:
                                reasoning.append("Collection showing bullish trend")
                            if collection_analysis.liquidity_score > 7:
                                reasoning.append("High liquidity for easy exit")
                            if expected_return_percentage > 15:
                                reasoning.append(f"Strong expected returns: {expected_return_percentage:.1f}%")
                            
                            # Risk factors
                            risk_factors = []
                            if collection_analysis.price_volatility > 0.3:
                                risk_factors.append("High price volatility")
                            if collection_analysis.liquidity_score < 5:
                                risk_factors.append("Limited liquidity")
                            if collection_analysis.trend == MarketTrend.BEARISH:
                                risk_factors.append("Bearish market trend")
                            
                            recommendation = InvestmentRecommendation(
                                token_id=token_id,
                                contract_address=collection['contract_address'],
                                collection_name=collection_analysis.collection_name,
                                name=f"{collection['name']} #{token_id}",
                                recommendation_type=recommendation_type,
                                confidence_score=price_prediction.confidence_score,
                                target_price_eth=price_prediction.predicted_price_eth,
                                target_price_usd=price_prediction.predicted_price_usd,
                                expected_return_percentage=expected_return_percentage,
                                risk_level=collection_analysis.risk_level,
                                investment_horizon="medium",
                                reasoning=reasoning,
                                supporting_factors={
                                    'rarity_score': rarity_score.overall_rarity_score,
                                    'collection_volume_30d': float(collection_analysis.volume_30d_eth),
                                    'liquidity_score': collection_analysis.liquidity_score,
                                    'social_sentiment': collection_analysis.social_metrics.get('social_sentiment', 0.5)
                                },
                                risk_factors=risk_factors,
                                market_conditions={
                                    'trend': collection_analysis.trend.value,
                                    'volatility': collection_analysis.price_volatility,
                                    'market_cap_eth': float(collection_analysis.market_cap_eth)
                                }
                            )
                            
                            recommendations.append(recommendation)
                            
                        except Exception as e:
                            self.logger.warning(f"Error analyzing token {token_id}: {e}")
                            continue
                
                except Exception as e:
                    self.logger.warning(f"Error analyzing collection {collection['contract_address']}: {e}")
                    continue
            
            # Sort recommendations by expected return and confidence
            recommendations.sort(
                key=lambda x: (x.expected_return_percentage * x.confidence_score), 
                reverse=True
            )
            
            return recommendations[:10]  # Top 10 recommendations
            
        except Exception as e:
            self.logger.error(f"Error getting investment recommendations: {e}")
            return []
    
    async def get_market_trends(self, category: Optional[NFTCategory] = None) -> Dict[str, Any]:
        """Get NFT market trends
        
        Args:
            category: Specific NFT category to analyze (optional)
            
        Returns:
            Market trends data
        """
        try:
            # Mock market trends data
            trends = {
                'overall_market': {
                    'total_volume_24h_eth': 1250.5,
                    'total_volume_24h_usd': 2501000,
                    'total_sales_24h': 3420,
                    'average_sale_price_eth': 0.365,
                    'average_sale_price_usd': 730,
                    'market_trend': MarketTrend.SIDEWAYS.value,
                    'sentiment_score': 0.62,  # 0-1 scale
                    'volatility_index': 0.28
                },
                'category_trends': {
                    'art': {
                        'volume_change_24h': 5.2,
                        'price_change_24h': -2.1,
                        'trend': MarketTrend.BULLISH.value
                    },
                    'collectibles': {
                        'volume_change_24h': -8.5,
                        'price_change_24h': -5.3,
                        'trend': MarketTrend.BEARISH.value
                    },
                    'gaming': {
                        'volume_change_24h': 15.8,
                        'price_change_24h': 12.4,
                        'trend': MarketTrend.BULLISH.value
                    },
                    'metaverse': {
                        'volume_change_24h': 3.2,
                        'price_change_24h': 1.8,
                        'trend': MarketTrend.SIDEWAYS.value
                    }
                },
                'top_collections_24h': [
                    {
                        'name': 'Bored Ape Yacht Club',
                        'volume_eth': 125.5,
                        'volume_change': 8.2,
                        'floor_price_eth': 45.5,
                        'floor_change': -2.1
                    },
                    {
                        'name': 'CryptoPunks',
                        'volume_eth': 89.3,
                        'volume_change': -5.4,
                        'floor_price_eth': 65.2,
                        'floor_change': 1.8
                    },
                    {
                        'name': 'Azuki',
                        'volume_eth': 67.8,
                        'volume_change': 22.1,
                        'floor_price_eth': 8.9,
                        'floor_change': 15.3
                    }
                ],
                'emerging_collections': [
                    {
                        'name': 'New Art Collection',
                        'volume_growth_7d': 450.2,
                        'floor_price_eth': 0.5,
                        'unique_holders': 1250,
                        'risk_level': InvestmentRisk.HIGH.value
                    }
                ],
                'market_indicators': {
                    'whale_activity': 'moderate',  # low, moderate, high
                    'new_collection_launches': 15,
                    'celebrity_endorsements': 3,
                    'institutional_interest': 'growing',
                    'regulatory_sentiment': 'neutral'
                },
                'price_predictions': {
                    'next_7_days': {
                        'direction': 'sideways',
                        'confidence': 0.65,
                        'expected_change': 2.1
                    },
                    'next_30_days': {
                        'direction': 'bullish',
                        'confidence': 0.58,
                        'expected_change': 8.5
                    }
                },
                'last_updated': datetime.now().isoformat()
            }
            
            # Filter by category if specified
            if category:
                category_data = trends['category_trends'].get(category.value, {})
                trends['filtered_category'] = {
                    'category': category.value,
                    'data': category_data
                }
            
            return trends
            
        except Exception as e:
            self.logger.error(f"Error getting market trends: {e}")
            return {}
    
    async def shutdown(self):
        """Gracefully shutdown NFT analyzer"""
        self.logger.info("Shutting down NFT analyzer...")
        
        # Clear caches
        self.collection_cache.clear()
        self.rarity_cache.clear()
        self.price_cache.clear()
        self.market_data_cache.clear()
        
        self.logger.info("NFT analyzer shutdown complete")

# Export main classes
__all__ = [
    'NFTAnalyzer',
    'NFTMetadata', 'NFTMarketData', 'RarityScore', 'PricePrediction',
    'CollectionAnalysis', 'NFTPortfolioItem', 'InvestmentRecommendation',
    'NFTCategory', 'RarityTier', 'MarketTrend', 'InvestmentRisk'
]