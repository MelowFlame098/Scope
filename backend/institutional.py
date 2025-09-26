from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import json

router = APIRouter(prefix="/api/institutional", tags=["institutional"])

class ClientType(str, Enum):
    PENSION_FUND = "pension_fund"
    INSURANCE = "insurance"
    SOVEREIGN_WEALTH = "sovereign_wealth"
    ENDOWMENT = "endowment"
    FOUNDATION = "foundation"
    FAMILY_OFFICE = "family_office"
    HEDGE_FUND = "hedge_fund"
    MUTUAL_FUND = "mutual_fund"

class PortfolioStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"
    CLOSED = "closed"

class RiskLevel(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    VERY_AGGRESSIVE = "very_aggressive"

class ComplianceStatus(str, Enum):
    COMPLIANT = "compliant"
    WARNING = "warning"
    VIOLATION = "violation"
    PENDING_REVIEW = "pending_review"

# Pydantic Models
class InstitutionalClient(BaseModel):
    id: str
    name: str
    type: ClientType
    aum: float  # Assets Under Management
    contactPerson: str
    email: str
    phone: str
    address: str
    country: str
    onboardingDate: datetime
    lastReviewDate: datetime
    riskProfile: RiskLevel
    complianceStatus: ComplianceStatus
    kycStatus: str
    isActive: bool
    portfolios: List[str] = []  # Portfolio IDs
    documents: List[str] = []
    notes: str = ""

class InstitutionalPortfolio(BaseModel):
    id: str
    clientId: str
    name: str
    description: str
    totalValue: float
    cashBalance: float
    investedAmount: float
    unrealizedPnL: float
    realizedPnL: float
    totalReturn: float
    totalReturnPercent: float
    benchmark: str
    benchmarkReturn: float
    alpha: float
    beta: float
    sharpeRatio: float
    maxDrawdown: float
    volatility: float
    var95: float  # Value at Risk 95%
    expectedShortfall: float
    riskLevel: RiskLevel
    status: PortfolioStatus
    createdAt: datetime
    lastRebalanced: datetime
    targetAllocation: Dict[str, float]
    currentAllocation: Dict[str, float]
    holdings: List[Dict[str, Any]] = []
    constraints: Dict[str, Any] = {}

class RiskMetrics(BaseModel):
    portfolioId: str
    date: datetime
    var95: float
    var99: float
    expectedShortfall: float
    beta: float
    alpha: float
    sharpeRatio: float
    sortinoRatio: float
    informationRatio: float
    trackingError: float
    maxDrawdown: float
    volatility: float
    downside_volatility: float
    correlation_to_benchmark: float
    concentration_risk: float
    liquidity_risk: float
    credit_risk: float
    market_risk: float

class PerformanceAttribution(BaseModel):
    portfolioId: str
    period: str
    totalReturn: float
    benchmarkReturn: float
    activeReturn: float
    assetAllocation: float
    stockSelection: float
    interaction: float
    currency: float
    timing: float
    sectors: Dict[str, Dict[str, float]] = {}
    regions: Dict[str, Dict[str, float]] = {}

class ESGMetrics(BaseModel):
    portfolioId: str
    date: datetime
    esgScore: float
    environmentalScore: float
    socialScore: float
    governanceScore: float
    carbonFootprint: float
    waterFootprint: float
    wasteGeneration: float
    renewableEnergy: float
    boardDiversity: float
    executiveCompensation: float
    controversyScore: float
    sustainabilityRank: int
    esgRating: str
    msciEsgRating: str
    ungcCompliance: bool

class ComplianceCheck(BaseModel):
    id: str
    portfolioId: str
    ruleId: str
    ruleName: str
    description: str
    status: ComplianceStatus
    severity: str
    violationAmount: Optional[float] = None
    threshold: float
    currentValue: float
    checkDate: datetime
    resolvedDate: Optional[datetime] = None
    notes: str = ""
    actionRequired: str = ""

class TradingActivity(BaseModel):
    id: str
    portfolioId: str
    symbol: str
    side: str  # buy/sell
    quantity: float
    price: float
    value: float
    commission: float
    timestamp: datetime
    orderId: str
    executionVenue: str
    trader: str
    strategy: str
    reason: str
    status: str

class ClientRequest(BaseModel):
    name: str
    type: ClientType
    contactPerson: str
    email: str
    phone: str
    address: str
    country: str
    riskProfile: RiskLevel
    initialAum: float

class PortfolioRequest(BaseModel):
    clientId: str
    name: str
    description: str
    benchmark: str
    riskLevel: RiskLevel
    targetAllocation: Dict[str, float]
    constraints: Dict[str, Any] = {}

# Service Class
class InstitutionalService:
    def __init__(self):
        self.clients = {}
        self.portfolios = {}
        self.risk_metrics = {}
        self.performance_attribution = {}
        self.esg_metrics = {}
        self.compliance_checks = {}
        self.trading_activity = {}
        self._initialize_mock_data()
    
    def _initialize_mock_data(self):
        """Initialize with mock data for demonstration"""
        # Mock clients
        mock_clients = [
            {
                "id": "client-1",
                "name": "Global Pension Fund",
                "type": ClientType.PENSION_FUND,
                "aum": 2500000000,
                "contactPerson": "John Smith",
                "email": "john.smith@globalpension.com",
                "phone": "+1-555-0123",
                "address": "123 Wall Street, New York, NY 10005",
                "country": "United States",
                "onboardingDate": datetime.now() - timedelta(days=365),
                "lastReviewDate": datetime.now() - timedelta(days=30),
                "riskProfile": RiskLevel.MODERATE,
                "complianceStatus": ComplianceStatus.COMPLIANT,
                "kycStatus": "Approved",
                "isActive": True,
                "portfolios": ["portfolio-1", "portfolio-2"],
                "notes": "Large institutional client with conservative investment approach"
            },
            {
                "id": "client-2",
                "name": "University Endowment",
                "type": ClientType.ENDOWMENT,
                "aum": 850000000,
                "contactPerson": "Dr. Sarah Johnson",
                "email": "sarah.johnson@university.edu",
                "phone": "+1-555-0456",
                "address": "456 Academic Ave, Boston, MA 02138",
                "country": "United States",
                "onboardingDate": datetime.now() - timedelta(days=180),
                "lastReviewDate": datetime.now() - timedelta(days=15),
                "riskProfile": RiskLevel.AGGRESSIVE,
                "complianceStatus": ComplianceStatus.COMPLIANT,
                "kycStatus": "Approved",
                "isActive": True,
                "portfolios": ["portfolio-3"],
                "notes": "Long-term investment horizon with focus on growth"
            }
        ]
        
        for client_data in mock_clients:
            client = InstitutionalClient(**client_data)
            self.clients[client.id] = client
        
        # Mock portfolios
        mock_portfolios = [
            {
                "id": "portfolio-1",
                "clientId": "client-1",
                "name": "Conservative Growth Portfolio",
                "description": "Balanced portfolio with focus on capital preservation",
                "totalValue": 1500000000,
                "cashBalance": 75000000,
                "investedAmount": 1425000000,
                "unrealizedPnL": 45000000,
                "realizedPnL": 32000000,
                "totalReturn": 77000000,
                "totalReturnPercent": 5.4,
                "benchmark": "S&P 500",
                "benchmarkReturn": 4.8,
                "alpha": 0.6,
                "beta": 0.85,
                "sharpeRatio": 1.25,
                "maxDrawdown": -8.5,
                "volatility": 12.3,
                "var95": -15000000,
                "expectedShortfall": -22000000,
                "riskLevel": RiskLevel.MODERATE,
                "status": PortfolioStatus.ACTIVE,
                "createdAt": datetime.now() - timedelta(days=365),
                "lastRebalanced": datetime.now() - timedelta(days=30),
                "targetAllocation": {
                    "Equities": 60.0,
                    "Fixed Income": 30.0,
                    "Alternatives": 8.0,
                    "Cash": 2.0
                },
                "currentAllocation": {
                    "Equities": 62.5,
                    "Fixed Income": 28.0,
                    "Alternatives": 7.5,
                    "Cash": 2.0
                },
                "constraints": {
                    "maxSinglePosition": 5.0,
                    "maxSectorExposure": 15.0,
                    "minLiquidity": 10.0
                }
            },
            {
                "id": "portfolio-2",
                "clientId": "client-1",
                "name": "Fixed Income Portfolio",
                "description": "Conservative fixed income focused portfolio",
                "totalValue": 1000000000,
                "cashBalance": 50000000,
                "investedAmount": 950000000,
                "unrealizedPnL": 15000000,
                "realizedPnL": 18000000,
                "totalReturn": 33000000,
                "totalReturnPercent": 3.5,
                "benchmark": "Bloomberg Aggregate",
                "benchmarkReturn": 3.2,
                "alpha": 0.3,
                "beta": 0.95,
                "sharpeRatio": 0.85,
                "maxDrawdown": -3.2,
                "volatility": 4.8,
                "var95": -8000000,
                "expectedShortfall": -12000000,
                "riskLevel": RiskLevel.CONSERVATIVE,
                "status": PortfolioStatus.ACTIVE,
                "createdAt": datetime.now() - timedelta(days=300),
                "lastRebalanced": datetime.now() - timedelta(days=45),
                "targetAllocation": {
                    "Government Bonds": 40.0,
                    "Corporate Bonds": 35.0,
                    "Municipal Bonds": 15.0,
                    "TIPS": 5.0,
                    "Cash": 5.0
                },
                "currentAllocation": {
                    "Government Bonds": 38.5,
                    "Corporate Bonds": 36.0,
                    "Municipal Bonds": 15.5,
                    "TIPS": 5.0,
                    "Cash": 5.0
                },
                "constraints": {
                    "maxDuration": 7.0,
                    "minCreditRating": "BBB",
                    "maxConcentration": 10.0
                }
            }
        ]
        
        for portfolio_data in mock_portfolios:
            portfolio = InstitutionalPortfolio(**portfolio_data)
            self.portfolios[portfolio.id] = portfolio
        
        # Mock compliance checks
        mock_compliance = [
            {
                "id": "compliance-1",
                "portfolioId": "portfolio-1",
                "ruleId": "RULE-001",
                "ruleName": "Maximum Single Position",
                "description": "No single position should exceed 5% of portfolio value",
                "status": ComplianceStatus.WARNING,
                "severity": "Medium",
                "violationAmount": 2500000,
                "threshold": 5.0,
                "currentValue": 5.2,
                "checkDate": datetime.now() - timedelta(hours=2),
                "actionRequired": "Reduce position size or seek approval"
            }
        ]
        
        for compliance_data in mock_compliance:
            check = ComplianceCheck(**compliance_data)
            self.compliance_checks[check.id] = check
    
    async def get_clients(self, limit: int = 50, offset: int = 0) -> List[InstitutionalClient]:
        """Get institutional clients"""
        clients = list(self.clients.values())
        clients.sort(key=lambda x: x.aum, reverse=True)
        return clients[offset:offset + limit]
    
    async def get_client(self, client_id: str) -> InstitutionalClient:
        """Get specific client"""
        if client_id not in self.clients:
            raise HTTPException(status_code=404, detail="Client not found")
        return self.clients[client_id]
    
    async def create_client(self, client_data: ClientRequest) -> InstitutionalClient:
        """Create new institutional client"""
        client_id = f"client-{int(datetime.now().timestamp())}"
        client = InstitutionalClient(
            id=client_id,
            name=client_data.name,
            type=client_data.type,
            aum=client_data.initialAum,
            contactPerson=client_data.contactPerson,
            email=client_data.email,
            phone=client_data.phone,
            address=client_data.address,
            country=client_data.country,
            onboardingDate=datetime.now(),
            lastReviewDate=datetime.now(),
            riskProfile=client_data.riskProfile,
            complianceStatus=ComplianceStatus.PENDING_REVIEW,
            kycStatus="Pending",
            isActive=True
        )
        
        self.clients[client_id] = client
        return client
    
    async def get_portfolios(self, client_id: Optional[str] = None, limit: int = 50) -> List[InstitutionalPortfolio]:
        """Get portfolios, optionally filtered by client"""
        portfolios = list(self.portfolios.values())
        
        if client_id:
            portfolios = [p for p in portfolios if p.clientId == client_id]
        
        portfolios.sort(key=lambda x: x.totalValue, reverse=True)
        return portfolios[:limit]
    
    async def get_portfolio(self, portfolio_id: str) -> InstitutionalPortfolio:
        """Get specific portfolio"""
        if portfolio_id not in self.portfolios:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        return self.portfolios[portfolio_id]
    
    async def create_portfolio(self, portfolio_data: PortfolioRequest) -> InstitutionalPortfolio:
        """Create new portfolio"""
        if portfolio_data.clientId not in self.clients:
            raise HTTPException(status_code=404, detail="Client not found")
        
        portfolio_id = f"portfolio-{int(datetime.now().timestamp())}"
        portfolio = InstitutionalPortfolio(
            id=portfolio_id,
            clientId=portfolio_data.clientId,
            name=portfolio_data.name,
            description=portfolio_data.description,
            totalValue=0,
            cashBalance=0,
            investedAmount=0,
            unrealizedPnL=0,
            realizedPnL=0,
            totalReturn=0,
            totalReturnPercent=0,
            benchmark=portfolio_data.benchmark,
            benchmarkReturn=0,
            alpha=0,
            beta=1,
            sharpeRatio=0,
            maxDrawdown=0,
            volatility=0,
            var95=0,
            expectedShortfall=0,
            riskLevel=portfolio_data.riskLevel,
            status=PortfolioStatus.PENDING,
            createdAt=datetime.now(),
            lastRebalanced=datetime.now(),
            targetAllocation=portfolio_data.targetAllocation,
            currentAllocation={},
            constraints=portfolio_data.constraints
        )
        
        self.portfolios[portfolio_id] = portfolio
        
        # Add portfolio to client
        self.clients[portfolio_data.clientId].portfolios.append(portfolio_id)
        
        return portfolio
    
    async def get_risk_metrics(self, portfolio_id: str) -> RiskMetrics:
        """Get portfolio risk metrics"""
        if portfolio_id not in self.portfolios:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        
        # Mock risk metrics
        return RiskMetrics(
            portfolioId=portfolio_id,
            date=datetime.now(),
            var95=-15000000,
            var99=-25000000,
            expectedShortfall=-22000000,
            beta=0.85,
            alpha=0.6,
            sharpeRatio=1.25,
            sortinoRatio=1.45,
            informationRatio=0.35,
            trackingError=2.8,
            maxDrawdown=-8.5,
            volatility=12.3,
            downside_volatility=8.7,
            correlation_to_benchmark=0.92,
            concentration_risk=15.2,
            liquidity_risk=5.8,
            credit_risk=3.2,
            market_risk=12.1
        )
    
    async def get_performance_attribution(self, portfolio_id: str, period: str = "1M") -> PerformanceAttribution:
        """Get performance attribution analysis"""
        if portfolio_id not in self.portfolios:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        
        # Mock performance attribution
        return PerformanceAttribution(
            portfolioId=portfolio_id,
            period=period,
            totalReturn=5.4,
            benchmarkReturn=4.8,
            activeReturn=0.6,
            assetAllocation=0.2,
            stockSelection=0.3,
            interaction=0.05,
            currency=0.05,
            timing=0.0,
            sectors={
                "Technology": {"allocation": 0.1, "selection": 0.15, "total": 0.25},
                "Healthcare": {"allocation": 0.05, "selection": 0.08, "total": 0.13},
                "Financials": {"allocation": -0.02, "selection": 0.05, "total": 0.03}
            },
            regions={
                "North America": {"allocation": 0.15, "selection": 0.1, "total": 0.25},
                "Europe": {"allocation": 0.03, "selection": 0.12, "total": 0.15},
                "Asia Pacific": {"allocation": 0.02, "selection": 0.08, "total": 0.1}
            }
        )
    
    async def get_esg_metrics(self, portfolio_id: str) -> ESGMetrics:
        """Get ESG metrics for portfolio"""
        if portfolio_id not in self.portfolios:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        
        # Mock ESG metrics
        return ESGMetrics(
            portfolioId=portfolio_id,
            date=datetime.now(),
            esgScore=7.8,
            environmentalScore=8.2,
            socialScore=7.5,
            governanceScore=7.7,
            carbonFootprint=125.5,
            waterFootprint=89.2,
            wasteGeneration=45.8,
            renewableEnergy=65.3,
            boardDiversity=42.5,
            executiveCompensation=6.8,
            controversyScore=2.1,
            sustainabilityRank=85,
            esgRating="A-",
            msciEsgRating="AA",
            ungcCompliance=True
        )
    
    async def get_compliance_checks(self, portfolio_id: Optional[str] = None) -> List[ComplianceCheck]:
        """Get compliance checks"""
        checks = list(self.compliance_checks.values())
        
        if portfolio_id:
            checks = [c for c in checks if c.portfolioId == portfolio_id]
        
        checks.sort(key=lambda x: x.checkDate, reverse=True)
        return checks
    
    async def get_trading_activity(self, portfolio_id: str, limit: int = 100) -> List[TradingActivity]:
        """Get trading activity for portfolio"""
        if portfolio_id not in self.portfolios:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        
        # Mock trading activity
        mock_trades = [
            {
                "id": "trade-1",
                "portfolioId": portfolio_id,
                "symbol": "AAPL",
                "side": "buy",
                "quantity": 10000,
                "price": 175.50,
                "value": 1755000,
                "commission": 175.50,
                "timestamp": datetime.now() - timedelta(hours=2),
                "orderId": "ORD-12345",
                "executionVenue": "NYSE",
                "trader": "John Doe",
                "strategy": "Rebalancing",
                "reason": "Portfolio rebalancing to target allocation",
                "status": "Executed"
            },
            {
                "id": "trade-2",
                "portfolioId": portfolio_id,
                "symbol": "MSFT",
                "side": "sell",
                "quantity": 5000,
                "price": 380.25,
                "value": 1901250,
                "commission": 190.13,
                "timestamp": datetime.now() - timedelta(hours=4),
                "orderId": "ORD-12346",
                "executionVenue": "NASDAQ",
                "trader": "Jane Smith",
                "strategy": "Profit Taking",
                "reason": "Taking profits after 20% gain",
                "status": "Executed"
            }
        ]
        
        trades = [TradingActivity(**trade_data) for trade_data in mock_trades]
        return trades[:limit]
    
    async def get_dashboard_overview(self) -> Dict[str, Any]:
        """Get institutional dashboard overview"""
        total_aum = sum(client.aum for client in self.clients.values())
        total_portfolios = len(self.portfolios)
        active_clients = len([c for c in self.clients.values() if c.isActive])
        compliance_violations = len([c for c in self.compliance_checks.values() if c.status == ComplianceStatus.VIOLATION])
        
        return {
            "totalAUM": total_aum,
            "totalClients": len(self.clients),
            "activeClients": active_clients,
            "totalPortfolios": total_portfolios,
            "avgReturn": 5.7,
            "complianceScore": 94.5,
            "complianceViolations": compliance_violations,
            "riskScore": 6.8,
            "esgScore": 7.8,
            "recentActivity": {
                "newClients": 3,
                "newPortfolios": 2,
                "tradesExecuted": 156,
                "rebalancingEvents": 8
            }
        }

# Initialize service
institutional_service = InstitutionalService()

# API Endpoints
@router.get("/overview")
async def get_dashboard_overview():
    """Get institutional dashboard overview"""
    return await institutional_service.get_dashboard_overview()

@router.get("/clients", response_model=List[InstitutionalClient])
async def get_clients(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """Get institutional clients"""
    return await institutional_service.get_clients(limit, offset)

@router.get("/clients/{client_id}", response_model=InstitutionalClient)
async def get_client(client_id: str):
    """Get specific client"""
    return await institutional_service.get_client(client_id)

@router.post("/clients", response_model=InstitutionalClient)
async def create_client(client_data: ClientRequest):
    """Create new institutional client"""
    return await institutional_service.create_client(client_data)

@router.get("/portfolios", response_model=List[InstitutionalPortfolio])
async def get_portfolios(
    client_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=100)
):
    """Get portfolios"""
    return await institutional_service.get_portfolios(client_id, limit)

@router.get("/portfolios/{portfolio_id}", response_model=InstitutionalPortfolio)
async def get_portfolio(portfolio_id: str):
    """Get specific portfolio"""
    return await institutional_service.get_portfolio(portfolio_id)

@router.post("/portfolios", response_model=InstitutionalPortfolio)
async def create_portfolio(portfolio_data: PortfolioRequest):
    """Create new portfolio"""
    return await institutional_service.create_portfolio(portfolio_data)

@router.get("/portfolios/{portfolio_id}/risk", response_model=RiskMetrics)
async def get_risk_metrics(portfolio_id: str):
    """Get portfolio risk metrics"""
    return await institutional_service.get_risk_metrics(portfolio_id)

@router.get("/portfolios/{portfolio_id}/attribution", response_model=PerformanceAttribution)
async def get_performance_attribution(
    portfolio_id: str,
    period: str = Query("1M", regex="^(1D|1W|1M|3M|6M|1Y)$")
):
    """Get performance attribution"""
    return await institutional_service.get_performance_attribution(portfolio_id, period)

@router.get("/portfolios/{portfolio_id}/esg", response_model=ESGMetrics)
async def get_esg_metrics(portfolio_id: str):
    """Get ESG metrics"""
    return await institutional_service.get_esg_metrics(portfolio_id)

@router.get("/compliance", response_model=List[ComplianceCheck])
async def get_compliance_checks(
    portfolio_id: Optional[str] = Query(None)
):
    """Get compliance checks"""
    return await institutional_service.get_compliance_checks(portfolio_id)

@router.get("/portfolios/{portfolio_id}/trades", response_model=List[TradingActivity])
async def get_trading_activity(
    portfolio_id: str,
    limit: int = Query(100, ge=1, le=500)
):
    """Get trading activity"""
    return await institutional_service.get_trading_activity(portfolio_id, limit)

@router.get("/stats")
async def get_institutional_stats():
    """Get institutional platform statistics"""
    return {
        "totalClients": len(institutional_service.clients),
        "totalPortfolios": len(institutional_service.portfolios),
        "totalAUM": sum(client.aum for client in institutional_service.clients.values()),
        "avgReturn": 5.7,
        "complianceScore": 94.5,
        "esgScore": 7.8
    }