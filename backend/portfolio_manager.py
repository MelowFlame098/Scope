"""Portfolio Management System for FinScope - Phase 7 Implementation

Provides comprehensive portfolio tracking, performance analysis, and optimization
capabilities for managing investment portfolios across multiple asset classes.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database import get_db
from db_models import User, Portfolio, PortfolioHolding
from market_data import MarketDataService
from portfolio_analytics import PortfolioAnalytics
from risk_engine import RiskEngine

logger = logging.getLogger(__name__)

class PortfolioType(str, Enum):
    """Types of portfolios"""
    INVESTMENT = "investment"
    TRADING = "trading"
    RETIREMENT = "retirement"
    SAVINGS = "savings"
    CRYPTO = "crypto"
    CUSTOM = "custom"

class TransactionType(str, Enum):
    """Types of portfolio transactions"""
    BUY = "buy"
    SELL = "sell"
    DIVIDEND = "dividend"
    SPLIT = "split"
    TRANSFER_IN = "transfer_in"
    TRANSFER_OUT = "transfer_out"
    FEE = "fee"
    INTEREST = "interest"

class RebalanceStrategy(str, Enum):
    """Portfolio rebalancing strategies"""
    EQUAL_WEIGHT = "equal_weight"
    MARKET_CAP = "market_cap"
    RISK_PARITY = "risk_parity"
    TARGET_ALLOCATION = "target_allocation"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"

@dataclass
class PortfolioHolding:
    """Individual portfolio holding"""
    symbol: str
    quantity: float
    average_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_percent: float
    weight: float
    last_updated: datetime

@dataclass
class PortfolioSummary:
    """Portfolio summary statistics"""
    total_value: float
    total_cost: float
    total_pnl: float
    total_pnl_percent: float
    day_change: float
    day_change_percent: float
    cash_balance: float
    invested_amount: float
    number_of_holdings: int
    last_updated: datetime

class PortfolioRequest(BaseModel):
    """Request for portfolio operations"""
    user_id: str
    portfolio_id: Optional[str] = None
    name: str
    description: Optional[str] = None
    portfolio_type: PortfolioType = PortfolioType.INVESTMENT
    initial_cash: float = 10000.0
    benchmark_symbol: Optional[str] = "SPY"
    target_allocation: Optional[Dict[str, float]] = None

class TransactionRequest(BaseModel):
    """Request for portfolio transactions"""
    portfolio_id: str
    symbol: str
    transaction_type: TransactionType
    quantity: float
    price: float
    fee: float = 0.0
    timestamp: Optional[datetime] = None
    notes: Optional[str] = None

class RebalanceRequest(BaseModel):
    """Request for portfolio rebalancing"""
    portfolio_id: str
    strategy: RebalanceStrategy
    target_allocation: Optional[Dict[str, float]] = None
    min_trade_amount: float = 100.0
    max_deviation: float = 0.05  # 5% deviation threshold
    dry_run: bool = True

class PortfolioResponse(BaseModel):
    """Response for portfolio operations"""
    portfolio_id: str
    name: str
    summary: PortfolioSummary
    holdings: List[PortfolioHolding]
    performance_metrics: Dict[str, Any]
    risk_metrics: Dict[str, Any]
    allocation: Dict[str, float]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class PortfolioManager:
    """Advanced portfolio management system"""
    
    def __init__(self):
        self.market_service = MarketDataService()
        self.analytics = PortfolioAnalytics()
        self.risk_engine = RiskEngine()
        
        # Portfolio configurations
        self.default_allocations = {
            PortfolioType.INVESTMENT: {
                "stocks": 0.60,
                "bonds": 0.30,
                "cash": 0.10
            },
            PortfolioType.TRADING: {
                "stocks": 0.80,
                "cash": 0.20
            },
            PortfolioType.RETIREMENT: {
                "stocks": 0.50,
                "bonds": 0.40,
                "cash": 0.10
            },
            PortfolioType.CRYPTO: {
                "crypto": 0.90,
                "cash": 0.10
            }
        }
    
    async def create_portfolio(
        self,
        request: PortfolioRequest,
        db: Session
    ) -> PortfolioResponse:
        """Create a new portfolio"""
        try:
            # Create portfolio record
            portfolio = Portfolio(
                user_id=request.user_id,
                name=request.name,
                description=request.description,
                portfolio_type=request.portfolio_type.value,
                cash_balance=request.initial_cash,
                benchmark_symbol=request.benchmark_symbol,
                target_allocation=request.target_allocation or 
                    self.default_allocations.get(request.portfolio_type, {}),
                created_at=datetime.utcnow()
            )
            
            db.add(portfolio)
            db.commit()
            db.refresh(portfolio)
            
            # Create initial cash transaction
            initial_transaction = Transaction(
                portfolio_id=portfolio.id,
                symbol="CASH",
                transaction_type=TransactionType.TRANSFER_IN.value,
                quantity=1,
                price=request.initial_cash,
                total_amount=request.initial_cash,
                timestamp=datetime.utcnow()
            )
            
            db.add(initial_transaction)
            db.commit()
            
            logger.info(f"Created portfolio {portfolio.id} for user {request.user_id}")
            
            return await self.get_portfolio(portfolio.id, db)
            
        except Exception as e:
            logger.error(f"Error creating portfolio: {str(e)}")
            db.rollback()
            raise
    
    async def get_portfolio(
        self,
        portfolio_id: str,
        db: Session
    ) -> PortfolioResponse:
        """Get portfolio with current market data"""
        try:
            # Get portfolio from database
            portfolio = db.query(Portfolio).filter(
                Portfolio.id == portfolio_id
            ).first()
            
            if not portfolio:
                raise ValueError(f"Portfolio {portfolio_id} not found")
            
            # Get holdings
            holdings = db.query(Holding).filter(
                Holding.portfolio_id == portfolio_id
            ).all()
            
            # Update holdings with current market data
            portfolio_holdings = []
            total_value = portfolio.cash_balance
            total_cost = 0
            
            for holding in holdings:
                if holding.quantity > 0:
                    # Get current price
                    current_price = await self.market_service.get_current_price(
                        holding.symbol
                    )
                    
                    market_value = holding.quantity * current_price
                    cost_basis = holding.quantity * holding.average_cost
                    unrealized_pnl = market_value - cost_basis
                    unrealized_pnl_percent = (
                        unrealized_pnl / cost_basis * 100 if cost_basis > 0 else 0
                    )
                    
                    portfolio_holding = PortfolioHolding(
                        symbol=holding.symbol,
                        quantity=holding.quantity,
                        average_cost=holding.average_cost,
                        current_price=current_price,
                        market_value=market_value,
                        unrealized_pnl=unrealized_pnl,
                        unrealized_pnl_percent=unrealized_pnl_percent,
                        weight=0,  # Will be calculated after total value
                        last_updated=datetime.utcnow()
                    )
                    
                    portfolio_holdings.append(portfolio_holding)
                    total_value += market_value
                    total_cost += cost_basis
            
            # Calculate weights
            for holding in portfolio_holdings:
                holding.weight = holding.market_value / total_value * 100
            
            # Calculate portfolio summary
            total_pnl = total_value - total_cost - portfolio.cash_balance
            total_pnl_percent = (
                total_pnl / (total_cost + portfolio.cash_balance) * 100 
                if (total_cost + portfolio.cash_balance) > 0 else 0
            )
            
            # Get day change (simplified - would need historical data)
            day_change = 0  # TODO: Implement day change calculation
            day_change_percent = 0
            
            summary = PortfolioSummary(
                total_value=total_value,
                total_cost=total_cost + portfolio.cash_balance,
                total_pnl=total_pnl,
                total_pnl_percent=total_pnl_percent,
                day_change=day_change,
                day_change_percent=day_change_percent,
                cash_balance=portfolio.cash_balance,
                invested_amount=total_cost,
                number_of_holdings=len(portfolio_holdings),
                last_updated=datetime.utcnow()
            )
            
            # Calculate performance metrics
            performance_metrics = await self.analytics.calculate_performance_metrics(
                portfolio_id, db
            )
            
            # Calculate risk metrics
            risk_metrics = await self.risk_engine.calculate_portfolio_risk(
                portfolio_holdings
            )
            
            # Calculate allocation
            allocation = self._calculate_allocation(portfolio_holdings, total_value)
            
            return PortfolioResponse(
                portfolio_id=portfolio_id,
                name=portfolio.name,
                summary=summary,
                holdings=portfolio_holdings,
                performance_metrics=performance_metrics,
                risk_metrics=risk_metrics,
                allocation=allocation
            )
            
        except Exception as e:
            logger.error(f"Error getting portfolio {portfolio_id}: {str(e)}")
            raise
    
    async def add_transaction(
        self,
        request: TransactionRequest,
        db: Session
    ) -> Dict[str, Any]:
        """Add a transaction to portfolio"""
        try:
            portfolio = db.query(Portfolio).filter(
                Portfolio.id == request.portfolio_id
            ).first()
            
            if not portfolio:
                raise ValueError(f"Portfolio {request.portfolio_id} not found")
            
            # Calculate transaction details
            total_amount = request.quantity * request.price + request.fee
            timestamp = request.timestamp or datetime.utcnow()
            
            # Create transaction record
            transaction = Transaction(
                portfolio_id=request.portfolio_id,
                symbol=request.symbol,
                transaction_type=request.transaction_type.value,
                quantity=request.quantity,
                price=request.price,
                fee=request.fee,
                total_amount=total_amount,
                timestamp=timestamp,
                notes=request.notes
            )
            
            db.add(transaction)
            
            # Update holdings
            await self._update_holdings(
                request.portfolio_id,
                request.symbol,
                request.transaction_type,
                request.quantity,
                request.price,
                request.fee,
                db
            )
            
            db.commit()
            
            logger.info(
                f"Added {request.transaction_type.value} transaction for "
                f"{request.symbol} in portfolio {request.portfolio_id}"
            )
            
            return {
                "transaction_id": transaction.id,
                "status": "success",
                "message": "Transaction added successfully"
            }
            
        except Exception as e:
            logger.error(f"Error adding transaction: {str(e)}")
            db.rollback()
            raise
    
    async def rebalance_portfolio(
        self,
        request: RebalanceRequest,
        db: Session
    ) -> Dict[str, Any]:
        """Rebalance portfolio according to strategy"""
        try:
            # Get current portfolio
            portfolio_response = await self.get_portfolio(request.portfolio_id, db)
            
            # Calculate target allocation based on strategy
            target_allocation = await self._calculate_target_allocation(
                request.strategy,
                portfolio_response,
                request.target_allocation
            )
            
            # Calculate required trades
            trades = await self._calculate_rebalance_trades(
                portfolio_response,
                target_allocation,
                request.min_trade_amount,
                request.max_deviation
            )
            
            if request.dry_run:
                return {
                    "status": "dry_run",
                    "target_allocation": target_allocation,
                    "required_trades": trades,
                    "estimated_cost": sum(trade["estimated_cost"] for trade in trades)
                }
            
            # Execute trades
            executed_trades = []
            for trade in trades:
                transaction_request = TransactionRequest(
                    portfolio_id=request.portfolio_id,
                    symbol=trade["symbol"],
                    transaction_type=TransactionType.BUY if trade["quantity"] > 0 else TransactionType.SELL,
                    quantity=abs(trade["quantity"]),
                    price=trade["price"],
                    notes=f"Rebalancing - {request.strategy.value}"
                )
                
                result = await self.add_transaction(transaction_request, db)
                executed_trades.append(result)
            
            return {
                "status": "completed",
                "target_allocation": target_allocation,
                "executed_trades": executed_trades,
                "total_trades": len(executed_trades)
            }
            
        except Exception as e:
            logger.error(f"Error rebalancing portfolio: {str(e)}")
            raise
    
    async def get_portfolio_history(
        self,
        portfolio_id: str,
        start_date: datetime,
        end_date: datetime,
        db: Session
    ) -> Dict[str, Any]:
        """Get portfolio performance history"""
        try:
            # Get transactions in date range
            transactions = db.query(Transaction).filter(
                Transaction.portfolio_id == portfolio_id,
                Transaction.timestamp >= start_date,
                Transaction.timestamp <= end_date
            ).order_by(Transaction.timestamp).all()
            
            # Calculate daily portfolio values
            daily_values = await self.analytics.calculate_daily_portfolio_values(
                portfolio_id, start_date, end_date, db
            )
            
            # Calculate performance metrics
            performance_metrics = await self.analytics.calculate_period_performance(
                daily_values, start_date, end_date
            )
            
            return {
                "portfolio_id": portfolio_id,
                "start_date": start_date,
                "end_date": end_date,
                "daily_values": daily_values,
                "transactions": [{
                    "timestamp": t.timestamp,
                    "symbol": t.symbol,
                    "type": t.transaction_type,
                    "quantity": t.quantity,
                    "price": t.price,
                    "total_amount": t.total_amount
                } for t in transactions],
                "performance_metrics": performance_metrics
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio history: {str(e)}")
            raise
    
    def _calculate_allocation(
        self,
        holdings: List[PortfolioHolding],
        total_value: float
    ) -> Dict[str, float]:
        """Calculate portfolio allocation by asset class"""
        allocation = {}
        
        for holding in holdings:
            # Simplified asset class mapping
            asset_class = self._get_asset_class(holding.symbol)
            
            if asset_class not in allocation:
                allocation[asset_class] = 0
            
            allocation[asset_class] += holding.weight
        
        return allocation
    
    def _get_asset_class(self, symbol: str) -> str:
        """Get asset class for symbol (simplified)"""
        # This would be more sophisticated in production
        if symbol.endswith("-USD") or symbol in ["BTC", "ETH", "ADA"]:
            return "crypto"
        elif symbol in ["TLT", "IEF", "SHY", "AGG"]:
            return "bonds"
        elif symbol == "CASH":
            return "cash"
        else:
            return "stocks"
    
    async def _update_holdings(
        self,
        portfolio_id: str,
        symbol: str,
        transaction_type: TransactionType,
        quantity: float,
        price: float,
        fee: float,
        db: Session
    ):
        """Update portfolio holdings based on transaction"""
        holding = db.query(PortfolioHolding).filter(
            PortfolioHolding.portfolio_id == portfolio_id,
            PortfolioHolding.symbol == symbol
        ).first()
        
        if transaction_type == TransactionType.BUY:
            if holding:
                # Update existing holding
                total_cost = holding.quantity * holding.average_cost + quantity * price + fee
                total_quantity = holding.quantity + quantity
                holding.average_cost = total_cost / total_quantity if total_quantity > 0 else 0
                holding.quantity = total_quantity
            else:
                # Create new holding
                holding = PortfolioHolding(
                    portfolio_id=portfolio_id,
                    symbol=symbol,
                    quantity=quantity,
                    average_cost=(quantity * price + fee) / quantity if quantity > 0 else 0
                )
                db.add(holding)
            
            # Update cash balance
            portfolio = db.query(Portfolio).filter(
                Portfolio.id == portfolio_id
            ).first()
            portfolio.cash_balance -= (quantity * price + fee)
            
        elif transaction_type == TransactionType.SELL:
            if holding and holding.quantity >= quantity:
                holding.quantity -= quantity
                
                # Update cash balance
                portfolio = db.query(Portfolio).filter(
                    Portfolio.id == portfolio_id
                ).first()
                portfolio.cash_balance += (quantity * price - fee)
            else:
                raise ValueError("Insufficient holdings for sale")
    
    async def _calculate_target_allocation(
        self,
        strategy: RebalanceStrategy,
        portfolio: PortfolioResponse,
        custom_allocation: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Calculate target allocation based on strategy"""
        if strategy == RebalanceStrategy.TARGET_ALLOCATION and custom_allocation:
            return custom_allocation
        
        elif strategy == RebalanceStrategy.EQUAL_WEIGHT:
            num_holdings = len(portfolio.holdings)
            if num_holdings > 0:
                weight = 1.0 / num_holdings
                return {holding.symbol: weight for holding in portfolio.holdings}
        
        elif strategy == RebalanceStrategy.MARKET_CAP:
            # Would need market cap data
            return {holding.symbol: holding.weight / 100 for holding in portfolio.holdings}
        
        # Default to current allocation
        return {holding.symbol: holding.weight / 100 for holding in portfolio.holdings}
    
    async def _calculate_rebalance_trades(
        self,
        portfolio: PortfolioResponse,
        target_allocation: Dict[str, float],
        min_trade_amount: float,
        max_deviation: float
    ) -> List[Dict[str, Any]]:
        """Calculate required trades for rebalancing"""
        trades = []
        total_value = portfolio.summary.total_value
        
        for symbol, target_weight in target_allocation.items():
            # Find current holding
            current_holding = next(
                (h for h in portfolio.holdings if h.symbol == symbol),
                None
            )
            
            current_weight = current_holding.weight / 100 if current_holding else 0
            weight_diff = target_weight - current_weight
            
            if abs(weight_diff) > max_deviation:
                target_value = target_weight * total_value
                current_value = current_holding.market_value if current_holding else 0
                trade_value = target_value - current_value
                
                if abs(trade_value) >= min_trade_amount:
                    # Get current price
                    current_price = await self.market_service.get_current_price(symbol)
                    quantity = trade_value / current_price
                    
                    trades.append({
                        "symbol": symbol,
                        "quantity": quantity,
                        "price": current_price,
                        "estimated_cost": abs(trade_value),
                        "weight_diff": weight_diff
                    })
        
        return trades

# Global portfolio manager instance
portfolio_manager = PortfolioManager()

def get_portfolio_manager() -> PortfolioManager:
    """Get portfolio manager instance"""
    return portfolio_manager