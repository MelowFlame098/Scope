from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from database import get_db
from auth import get_current_user
import json
import numpy as np

router = APIRouter(prefix="/institutional-analytics", tags=["institutional-analytics"])

# Analytics models
class PerformanceAttributionRequest(BaseModel):
    portfolio_id: str
    benchmark_id: str
    period_start: datetime
    period_end: datetime
    attribution_type: str  # sector, security, factor

class ESGRequest(BaseModel):
    portfolio_id: str
    analysis_type: str  # score, breakdown, comparison
    benchmark_id: Optional[str] = None

class PerformanceAttribution(BaseModel):
    portfolio_id: str
    benchmark_id: str
    period_start: datetime
    period_end: datetime
    total_return: float
    benchmark_return: float
    active_return: float
    attribution_breakdown: List[Dict[str, Any]]
    risk_metrics: Dict[str, float]
    generated_at: datetime

class ESGMetrics(BaseModel):
    portfolio_id: str
    overall_score: float
    environmental_score: float
    social_score: float
    governance_score: float
    controversy_score: float
    carbon_intensity: float
    water_usage: float
    waste_generation: float
    board_diversity: float
    employee_satisfaction: float
    data_privacy_score: float
    last_updated: datetime

class RiskAnalytics(BaseModel):
    portfolio_id: str
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    beta: float
    alpha: float
    tracking_error: float
    information_ratio: float
    calculated_at: datetime

class FactorExposure(BaseModel):
    factor_name: str
    exposure: float
    contribution: float
    risk_contribution: float

class StyleAnalysis(BaseModel):
    portfolio_id: str
    style_factors: List[FactorExposure]
    r_squared: float
    active_share: float
    style_drift: float
    analysis_date: datetime

class PerformanceDecomposition(BaseModel):
    portfolio_id: str
    asset_allocation: float
    security_selection: float
    interaction: float
    currency_effect: float
    total_active_return: float
    period: str

class LiquidityAnalysis(BaseModel):
    portfolio_id: str
    liquidity_score: float
    days_to_liquidate: int
    liquidity_buckets: Dict[str, float]  # immediate, 1-day, 1-week, 1-month, >1-month
    market_impact: float
    bid_ask_spread: float
    analysis_date: datetime

class StressTestResult(BaseModel):
    scenario_name: str
    portfolio_impact: float
    var_impact: float
    duration: int  # days
    probability: float
    description: str

class InstitutionalAnalytics:
    def __init__(self):
        self.performance_cache = {}
        self.esg_cache = {}
        self.risk_cache = {}
        
    def calculate_performance_attribution(self, request: PerformanceAttributionRequest) -> PerformanceAttribution:
        """Calculate performance attribution analysis"""
        # Mock performance attribution calculation
        portfolio_return = 8.5  # Mock portfolio return
        benchmark_return = 7.2  # Mock benchmark return
        active_return = portfolio_return - benchmark_return
        
        # Mock attribution breakdown by sector
        attribution_breakdown = [
            {
                "sector": "Technology",
                "portfolio_weight": 25.0,
                "benchmark_weight": 22.0,
                "portfolio_return": 12.5,
                "benchmark_return": 10.2,
                "allocation_effect": 0.15,
                "selection_effect": 0.58,
                "total_effect": 0.73
            },
            {
                "sector": "Healthcare",
                "portfolio_weight": 18.0,
                "benchmark_weight": 15.0,
                "portfolio_return": 6.8,
                "benchmark_return": 7.5,
                "allocation_effect": 0.23,
                "selection_effect": -0.13,
                "total_effect": 0.10
            },
            {
                "sector": "Financial Services",
                "portfolio_weight": 15.0,
                "benchmark_weight": 18.0,
                "portfolio_return": 9.2,
                "benchmark_return": 8.1,
                "allocation_effect": -0.24,
                "selection_effect": 0.17,
                "total_effect": -0.07
            }
        ]
        
        risk_metrics = {
            "tracking_error": 2.8,
            "information_ratio": 0.46,
            "active_share": 65.2,
            "beta": 1.05
        }
        
        return PerformanceAttribution(
            portfolio_id=request.portfolio_id,
            benchmark_id=request.benchmark_id,
            period_start=request.period_start,
            period_end=request.period_end,
            total_return=portfolio_return,
            benchmark_return=benchmark_return,
            active_return=active_return,
            attribution_breakdown=attribution_breakdown,
            risk_metrics=risk_metrics,
            generated_at=datetime.now()
        )
    
    def calculate_esg_metrics(self, request: ESGRequest) -> ESGMetrics:
        """Calculate ESG metrics for a portfolio"""
        # Mock ESG calculation
        return ESGMetrics(
            portfolio_id=request.portfolio_id,
            overall_score=7.8,
            environmental_score=8.2,
            social_score=7.5,
            governance_score=7.7,
            controversy_score=2.1,
            carbon_intensity=125.5,  # tons CO2e per $M invested
            water_usage=45.2,  # cubic meters per $M invested
            waste_generation=12.8,  # tons per $M invested
            board_diversity=42.5,  # percentage
            employee_satisfaction=78.3,  # percentage
            data_privacy_score=8.1,
            last_updated=datetime.now()
        )
    
    def calculate_risk_analytics(self, portfolio_id: str) -> RiskAnalytics:
        """Calculate comprehensive risk analytics"""
        # Mock risk calculations
        return RiskAnalytics(
            portfolio_id=portfolio_id,
            var_95=2.1,  # 95% VaR
            var_99=3.8,  # 99% VaR
            cvar_95=3.2,  # 95% Conditional VaR
            cvar_99=5.1,  # 99% Conditional VaR
            volatility=15.8,  # Annualized volatility
            sharpe_ratio=1.25,
            sortino_ratio=1.68,
            max_drawdown=8.5,
            beta=1.08,
            alpha=1.2,
            tracking_error=2.8,
            information_ratio=0.46,
            calculated_at=datetime.now()
        )
    
    def perform_style_analysis(self, portfolio_id: str) -> StyleAnalysis:
        """Perform style analysis using factor exposures"""
        style_factors = [
            FactorExposure(factor_name="Market", exposure=1.05, contribution=7.2, risk_contribution=65.2),
            FactorExposure(factor_name="Size", exposure=-0.15, contribution=-0.3, risk_contribution=8.1),
            FactorExposure(factor_name="Value", exposure=0.22, contribution=0.8, risk_contribution=12.5),
            FactorExposure(factor_name="Momentum", exposure=0.18, contribution=0.6, risk_contribution=9.8),
            FactorExposure(factor_name="Quality", exposure=0.35, contribution=1.2, risk_contribution=15.3),
            FactorExposure(factor_name="Low Volatility", exposure=-0.08, contribution=-0.2, risk_contribution=4.2)
        ]
        
        return StyleAnalysis(
            portfolio_id=portfolio_id,
            style_factors=style_factors,
            r_squared=0.87,
            active_share=65.2,
            style_drift=2.3,
            analysis_date=datetime.now()
        )
    
    def decompose_performance(self, portfolio_id: str, period: str) -> PerformanceDecomposition:
        """Decompose portfolio performance into components"""
        return PerformanceDecomposition(
            portfolio_id=portfolio_id,
            asset_allocation=0.45,  # Asset allocation effect
            security_selection=0.85,  # Security selection effect
            interaction=0.12,  # Interaction effect
            currency_effect=-0.08,  # Currency effect
            total_active_return=1.34,  # Total active return
            period=period
        )
    
    def analyze_liquidity(self, portfolio_id: str) -> LiquidityAnalysis:
        """Analyze portfolio liquidity"""
        liquidity_buckets = {
            "immediate": 15.2,  # Percentage that can be liquidated immediately
            "1_day": 35.8,     # Within 1 day
            "1_week": 25.5,    # Within 1 week
            "1_month": 18.3,   # Within 1 month
            "over_1_month": 5.2  # Over 1 month
        }
        
        return LiquidityAnalysis(
            portfolio_id=portfolio_id,
            liquidity_score=7.8,
            days_to_liquidate=12,
            liquidity_buckets=liquidity_buckets,
            market_impact=0.85,  # Percentage
            bid_ask_spread=0.12,  # Percentage
            analysis_date=datetime.now()
        )
    
    def run_stress_tests(self, portfolio_id: str) -> List[StressTestResult]:
        """Run stress tests on portfolio"""
        stress_scenarios = [
            StressTestResult(
                scenario_name="2008 Financial Crisis",
                portfolio_impact=-28.5,
                var_impact=4.2,
                duration=365,
                probability=0.02,
                description="Severe market downturn similar to 2008 financial crisis"
            ),
            StressTestResult(
                scenario_name="COVID-19 Market Shock",
                portfolio_impact=-18.2,
                var_impact=3.1,
                duration=90,
                probability=0.05,
                description="Pandemic-induced market volatility and economic uncertainty"
            ),
            StressTestResult(
                scenario_name="Interest Rate Shock",
                portfolio_impact=-12.8,
                var_impact=2.5,
                duration=180,
                probability=0.15,
                description="Rapid increase in interest rates by 300 basis points"
            ),
            StressTestResult(
                scenario_name="Geopolitical Crisis",
                portfolio_impact=-15.6,
                var_impact=2.8,
                duration=120,
                probability=0.08,
                description="Major geopolitical event affecting global markets"
            )
        ]
        
        return stress_scenarios
    
    def generate_institutional_report(self, portfolio_id: str, report_type: str) -> Dict[str, Any]:
        """Generate comprehensive institutional analytics report"""
        performance_attr = self.calculate_performance_attribution(
            PerformanceAttributionRequest(
                portfolio_id=portfolio_id,
                benchmark_id="benchmark_001",
                period_start=datetime.now() - timedelta(days=365),
                period_end=datetime.now(),
                attribution_type="sector"
            )
        )
        
        esg_metrics = self.calculate_esg_metrics(
            ESGRequest(
                portfolio_id=portfolio_id,
                analysis_type="score"
            )
        )
        
        risk_analytics = self.calculate_risk_analytics(portfolio_id)
        style_analysis = self.perform_style_analysis(portfolio_id)
        liquidity_analysis = self.analyze_liquidity(portfolio_id)
        stress_tests = self.run_stress_tests(portfolio_id)
        
        return {
            "report_id": f"report_{portfolio_id}_{datetime.now().strftime('%Y%m%d')}",
            "portfolio_id": portfolio_id,
            "report_type": report_type,
            "generated_at": datetime.now(),
            "performance_attribution": performance_attr.dict(),
            "esg_metrics": esg_metrics.dict(),
            "risk_analytics": risk_analytics.dict(),
            "style_analysis": style_analysis.dict(),
            "liquidity_analysis": liquidity_analysis.dict(),
            "stress_tests": [test.dict() for test in stress_tests],
            "summary": {
                "total_return": performance_attr.total_return,
                "active_return": performance_attr.active_return,
                "sharpe_ratio": risk_analytics.sharpe_ratio,
                "max_drawdown": risk_analytics.max_drawdown,
                "esg_score": esg_metrics.overall_score,
                "liquidity_score": liquidity_analysis.liquidity_score
            }
        }

# Initialize service
institutional_analytics = InstitutionalAnalytics()

# API Endpoints
@router.post("/performance-attribution")
async def calculate_performance_attribution(
    request: PerformanceAttributionRequest,
    current_user = Depends(get_current_user)
):
    """Calculate performance attribution"""
    attribution = institutional_analytics.calculate_performance_attribution(request)
    return attribution

@router.post("/esg-analysis")
async def calculate_esg_metrics(
    request: ESGRequest,
    current_user = Depends(get_current_user)
):
    """Calculate ESG metrics"""
    esg_metrics = institutional_analytics.calculate_esg_metrics(request)
    return esg_metrics

@router.get("/risk-analytics/{portfolio_id}")
async def get_risk_analytics(
    portfolio_id: str,
    current_user = Depends(get_current_user)
):
    """Get comprehensive risk analytics"""
    risk_analytics = institutional_analytics.calculate_risk_analytics(portfolio_id)
    return risk_analytics

@router.get("/style-analysis/{portfolio_id}")
async def get_style_analysis(
    portfolio_id: str,
    current_user = Depends(get_current_user)
):
    """Get style analysis"""
    style_analysis = institutional_analytics.perform_style_analysis(portfolio_id)
    return style_analysis

@router.get("/performance-decomposition/{portfolio_id}")
async def get_performance_decomposition(
    portfolio_id: str,
    period: str = "1Y",
    current_user = Depends(get_current_user)
):
    """Get performance decomposition"""
    decomposition = institutional_analytics.decompose_performance(portfolio_id, period)
    return decomposition

@router.get("/liquidity-analysis/{portfolio_id}")
async def get_liquidity_analysis(
    portfolio_id: str,
    current_user = Depends(get_current_user)
):
    """Get liquidity analysis"""
    liquidity = institutional_analytics.analyze_liquidity(portfolio_id)
    return liquidity

@router.get("/stress-tests/{portfolio_id}")
async def get_stress_tests(
    portfolio_id: str,
    current_user = Depends(get_current_user)
):
    """Run stress tests"""
    stress_tests = institutional_analytics.run_stress_tests(portfolio_id)
    return stress_tests

@router.get("/institutional-report/{portfolio_id}")
async def generate_institutional_report(
    portfolio_id: str,
    report_type: str = "comprehensive",
    current_user = Depends(get_current_user)
):
    """Generate institutional analytics report"""
    report = institutional_analytics.generate_institutional_report(portfolio_id, report_type)
    return report

@router.get("/health")
async def analytics_health_check():
    """Institutional analytics health check"""
    return {
        "status": "healthy",
        "service": "institutional_analytics",
        "timestamp": datetime.now(),
        "features": [
            "performance_attribution",
            "esg_analysis",
            "risk_analytics",
            "style_analysis",
            "liquidity_analysis",
            "stress_testing"
        ]
    }

# Export service and router
__all__ = ["InstitutionalAnalytics", "institutional_analytics", "router"]