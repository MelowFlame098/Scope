"""Trading Execution Engine for FinScope - Phase 7 Implementation

Provides comprehensive trading capabilities including order management,
execution, and trade monitoring for automated and manual trading.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from decimal import Decimal

import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database import get_db
from db_models import Portfolio, Order, Trade, Position
from market_data import MarketDataService
from portfolio_manager import PortfolioManager
from risk_engine import RiskEngine

logger = logging.getLogger(__name__)

class OrderType(str, Enum):
    """Types of trading orders"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    BRACKET = "bracket"
    OCO = "oco"  # One-Cancels-Other

class OrderSide(str, Enum):
    """Order side (buy/sell)"""
    BUY = "buy"
    SELL = "sell"
    SELL_SHORT = "sell_short"
    BUY_TO_COVER = "buy_to_cover"

class OrderStatus(str, Enum):
    """Order status"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class TimeInForce(str, Enum):
    """Time in force for orders"""
    DAY = "day"
    GTC = "gtc"  # Good Till Cancelled
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill
    GTD = "gtd"  # Good Till Date

class ExecutionStrategy(str, Enum):
    """Order execution strategies"""
    AGGRESSIVE = "aggressive"  # Market orders, immediate execution
    PASSIVE = "passive"       # Limit orders, better prices
    TWAP = "twap"            # Time-Weighted Average Price
    VWAP = "vwap"            # Volume-Weighted Average Price
    ICEBERG = "iceberg"      # Large orders split into smaller chunks
    SMART = "smart"          # Intelligent routing

@dataclass
class OrderRequest:
    """Trading order request"""
    portfolio_id: str
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    execution_strategy: ExecutionStrategy = ExecutionStrategy.SMART
    notes: Optional[str] = None
    
    # Advanced order parameters
    trailing_amount: Optional[float] = None
    trailing_percent: Optional[float] = None
    bracket_profit_target: Optional[float] = None
    bracket_stop_loss: Optional[float] = None
    
    # Risk management
    max_position_size: Optional[float] = None
    risk_check: bool = True
    force_execution: bool = False

@dataclass
class OrderResponse:
    """Trading order response"""
    order_id: str
    status: OrderStatus
    symbol: str
    side: OrderSide
    quantity: float
    filled_quantity: float
    remaining_quantity: float
    average_fill_price: Optional[float]
    total_commission: float
    timestamp: datetime
    estimated_cost: float
    risk_assessment: Dict[str, Any]

@dataclass
class TradeExecution:
    """Trade execution details"""
    trade_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float
    timestamp: datetime
    execution_venue: str
    liquidity_flag: str  # Added/Removed liquidity

@dataclass
class PositionInfo:
    """Position information"""
    symbol: str
    quantity: float
    average_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_percent: float
    day_pnl: float
    day_pnl_percent: float

class TradingRequest(BaseModel):
    """Request for trading operations"""
    portfolio_id: str
    orders: List[Dict[str, Any]]
    execution_strategy: ExecutionStrategy = ExecutionStrategy.SMART
    risk_check: bool = True
    dry_run: bool = False

class TradingResponse(BaseModel):
    """Response for trading operations"""
    portfolio_id: str
    orders_submitted: List[OrderResponse]
    orders_rejected: List[Dict[str, Any]]
    total_estimated_cost: float
    risk_warnings: List[str]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class TradingEngine:
    """Advanced trading execution engine"""
    
    def __init__(self):
        self.market_service = MarketDataService()
        self.portfolio_manager = PortfolioManager()
        self.risk_engine = RiskEngine()
        
        # Trading configuration
        self.commission_rate = 0.001  # 0.1% commission
        self.min_commission = 1.0     # $1 minimum commission
        self.max_order_size = 1000000  # $1M max order size
        self.max_position_concentration = 0.25  # 25% max position size
        
        # Execution venues (simulated)
        self.execution_venues = [
            "SMART", "NYSE", "NASDAQ", "ARCA", "BATS", "IEX"
        ]
        
        # Order tracking
        self.active_orders: Dict[str, OrderRequest] = {}
        self.order_history: List[OrderResponse] = []
        
    async def submit_order(
        self,
        order_request: OrderRequest,
        db: Session
    ) -> OrderResponse:
        """Submit a trading order"""
        try:
            # Validate order
            validation_result = await self._validate_order(order_request, db)
            if not validation_result["valid"]:
                raise ValueError(f"Order validation failed: {validation_result['reason']}")
            
            # Risk assessment
            risk_assessment = await self._assess_order_risk(
                order_request, db
            ) if order_request.risk_check else {}
            
            # Check risk limits
            if order_request.risk_check and not order_request.force_execution:
                risk_check = await self._check_risk_limits(
                    order_request, risk_assessment, db
                )
                if not risk_check["passed"]:
                    raise ValueError(f"Risk check failed: {risk_check['reason']}")
            
            # Generate order ID
            order_id = self._generate_order_id()
            
            # Calculate estimated cost
            estimated_cost = await self._calculate_estimated_cost(order_request)
            
            # Create order record
            order = Order(
                id=order_id,
                portfolio_id=order_request.portfolio_id,
                symbol=order_request.symbol,
                side=order_request.side.value,
                quantity=order_request.quantity,
                order_type=order_request.order_type.value,
                limit_price=order_request.limit_price,
                stop_price=order_request.stop_price,
                time_in_force=order_request.time_in_force.value,
                status=OrderStatus.PENDING.value,
                created_at=datetime.utcnow(),
                notes=order_request.notes
            )
            
            db.add(order)
            db.commit()
            
            # Submit for execution
            execution_result = await self._execute_order(order_request, order_id)
            
            # Update order status
            order.status = execution_result["status"]
            order.filled_quantity = execution_result.get("filled_quantity", 0)
            order.average_fill_price = execution_result.get("average_fill_price")
            order.commission = execution_result.get("commission", 0)
            
            db.commit()
            
            # Create response
            response = OrderResponse(
                order_id=order_id,
                status=OrderStatus(execution_result["status"]),
                symbol=order_request.symbol,
                side=order_request.side,
                quantity=order_request.quantity,
                filled_quantity=execution_result.get("filled_quantity", 0),
                remaining_quantity=order_request.quantity - execution_result.get("filled_quantity", 0),
                average_fill_price=execution_result.get("average_fill_price"),
                total_commission=execution_result.get("commission", 0),
                timestamp=datetime.utcnow(),
                estimated_cost=estimated_cost,
                risk_assessment=risk_assessment
            )
            
            # Track order
            self.active_orders[order_id] = order_request
            self.order_history.append(response)
            
            # Update portfolio if filled
            if response.status == OrderStatus.FILLED:
                await self._update_portfolio_from_fill(order_request, response, db)
            
            logger.info(
                f"Order {order_id} submitted: {order_request.side.value} "
                f"{order_request.quantity} {order_request.symbol} - Status: {response.status.value}"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error submitting order: {str(e)}")
            db.rollback()
            raise
    
    async def cancel_order(
        self,
        order_id: str,
        db: Session
    ) -> Dict[str, Any]:
        """Cancel an active order"""
        try:
            # Get order from database
            order = db.query(Order).filter(Order.id == order_id).first()
            
            if not order:
                raise ValueError(f"Order {order_id} not found")
            
            if order.status not in [OrderStatus.PENDING.value, OrderStatus.SUBMITTED.value]:
                raise ValueError(f"Cannot cancel order in status: {order.status}")
            
            # Cancel order (simulated)
            order.status = OrderStatus.CANCELLED.value
            order.cancelled_at = datetime.utcnow()
            
            db.commit()
            
            # Remove from active orders
            if order_id in self.active_orders:
                del self.active_orders[order_id]
            
            logger.info(f"Order {order_id} cancelled")
            
            return {
                "order_id": order_id,
                "status": "cancelled",
                "message": "Order cancelled successfully"
            }
            
        except Exception as e:
            logger.error(f"Error cancelling order: {str(e)}")
            db.rollback()
            raise
    
    async def get_order_status(
        self,
        order_id: str,
        db: Session
    ) -> Dict[str, Any]:
        """Get order status and details"""
        try:
            order = db.query(Order).filter(Order.id == order_id).first()
            
            if not order:
                raise ValueError(f"Order {order_id} not found")
            
            # Get associated trades
            trades = db.query(Trade).filter(Trade.order_id == order_id).all()
            
            return {
                "order_id": order.id,
                "symbol": order.symbol,
                "side": order.side,
                "quantity": order.quantity,
                "order_type": order.order_type,
                "status": order.status,
                "filled_quantity": order.filled_quantity or 0,
                "remaining_quantity": order.quantity - (order.filled_quantity or 0),
                "average_fill_price": order.average_fill_price,
                "commission": order.commission or 0,
                "created_at": order.created_at,
                "trades": [{
                    "trade_id": t.id,
                    "quantity": t.quantity,
                    "price": t.price,
                    "commission": t.commission,
                    "timestamp": t.timestamp
                } for t in trades]
            }
            
        except Exception as e:
            logger.error(f"Error getting order status: {str(e)}")
            raise
    
    async def get_portfolio_positions(
        self,
        portfolio_id: str,
        db: Session
    ) -> List[PositionInfo]:
        """Get current portfolio positions"""
        try:
            # Get portfolio holdings
            portfolio_response = await self.portfolio_manager.get_portfolio(
                portfolio_id, db
            )
            
            positions = []
            for holding in portfolio_response.holdings:
                # Get current price and day change
                current_price = holding.current_price
                
                # Calculate day P&L (simplified - would need previous day's price)
                day_pnl = 0  # TODO: Implement day P&L calculation
                day_pnl_percent = 0
                
                position = PositionInfo(
                    symbol=holding.symbol,
                    quantity=holding.quantity,
                    average_cost=holding.average_cost,
                    current_price=current_price,
                    market_value=holding.market_value,
                    unrealized_pnl=holding.unrealized_pnl,
                    unrealized_pnl_percent=holding.unrealized_pnl_percent,
                    day_pnl=day_pnl,
                    day_pnl_percent=day_pnl_percent
                )
                
                positions.append(position)
            
            return positions
            
        except Exception as e:
            logger.error(f"Error getting portfolio positions: {str(e)}")
            return []
    
    async def execute_trading_strategy(
        self,
        request: TradingRequest,
        db: Session
    ) -> TradingResponse:
        """Execute multiple orders as part of a trading strategy"""
        try:
            submitted_orders = []
            rejected_orders = []
            total_estimated_cost = 0
            risk_warnings = []
            
            for order_data in request.orders:
                try:
                    # Create order request
                    order_request = OrderRequest(
                        portfolio_id=request.portfolio_id,
                        symbol=order_data["symbol"],
                        side=OrderSide(order_data["side"]),
                        quantity=order_data["quantity"],
                        order_type=OrderType(order_data.get("order_type", "market")),
                        limit_price=order_data.get("limit_price"),
                        stop_price=order_data.get("stop_price"),
                        time_in_force=TimeInForce(order_data.get("time_in_force", "day")),
                        execution_strategy=request.execution_strategy,
                        risk_check=request.risk_check,
                        notes=order_data.get("notes")
                    )
                    
                    if request.dry_run:
                        # Simulate order without execution
                        estimated_cost = await self._calculate_estimated_cost(order_request)
                        risk_assessment = await self._assess_order_risk(order_request, db)
                        
                        simulated_response = OrderResponse(
                            order_id=f"DRY_RUN_{len(submitted_orders)}",
                            status=OrderStatus.PENDING,
                            symbol=order_request.symbol,
                            side=order_request.side,
                            quantity=order_request.quantity,
                            filled_quantity=0,
                            remaining_quantity=order_request.quantity,
                            average_fill_price=None,
                            total_commission=0,
                            timestamp=datetime.utcnow(),
                            estimated_cost=estimated_cost,
                            risk_assessment=risk_assessment
                        )
                        
                        submitted_orders.append(simulated_response)
                        total_estimated_cost += estimated_cost
                        
                        # Check for risk warnings
                        if risk_assessment.get("risk_score", 0) > 70:
                            risk_warnings.append(
                                f"High risk detected for {order_request.symbol} order"
                            )
                    else:
                        # Execute actual order
                        order_response = await self.submit_order(order_request, db)
                        submitted_orders.append(order_response)
                        total_estimated_cost += order_response.estimated_cost
                        
                        # Check for risk warnings
                        if order_response.risk_assessment.get("risk_score", 0) > 70:
                            risk_warnings.append(
                                f"High risk detected for {order_request.symbol} order"
                            )
                
                except Exception as e:
                    rejected_orders.append({
                        "symbol": order_data.get("symbol", "Unknown"),
                        "reason": str(e),
                        "order_data": order_data
                    })
            
            return TradingResponse(
                portfolio_id=request.portfolio_id,
                orders_submitted=submitted_orders,
                orders_rejected=rejected_orders,
                total_estimated_cost=total_estimated_cost,
                risk_warnings=risk_warnings
            )
            
        except Exception as e:
            logger.error(f"Error executing trading strategy: {str(e)}")
            raise
    
    async def _validate_order(self, order_request: OrderRequest, db: Session) -> Dict[str, Any]:
        """Validate order parameters"""
        try:
            # Check portfolio exists
            portfolio = db.query(Portfolio).filter(
                Portfolio.id == order_request.portfolio_id
            ).first()
            
            if not portfolio:
                return {"valid": False, "reason": "Portfolio not found"}
            
            # Check symbol validity
            try:
                current_price = await self.market_service.get_current_price(
                    order_request.symbol
                )
                if current_price <= 0:
                    return {"valid": False, "reason": "Invalid symbol or no price data"}
            except Exception:
                return {"valid": False, "reason": "Symbol validation failed"}
            
            # Check quantity
            if order_request.quantity <= 0:
                return {"valid": False, "reason": "Quantity must be positive"}
            
            # Check order size limits
            estimated_value = order_request.quantity * current_price
            if estimated_value > self.max_order_size:
                return {"valid": False, "reason": f"Order size exceeds limit: ${estimated_value:,.2f}"}
            
            # Check limit price for limit orders
            if order_request.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
                if not order_request.limit_price or order_request.limit_price <= 0:
                    return {"valid": False, "reason": "Limit price required for limit orders"}
            
            # Check stop price for stop orders
            if order_request.order_type in [OrderType.STOP, OrderType.STOP_LIMIT, OrderType.TRAILING_STOP]:
                if not order_request.stop_price or order_request.stop_price <= 0:
                    return {"valid": False, "reason": "Stop price required for stop orders"}
            
            # Check sufficient cash for buy orders
            if order_request.side == OrderSide.BUY:
                required_cash = estimated_value + (estimated_value * self.commission_rate)
                if portfolio.cash_balance < required_cash:
                    return {"valid": False, "reason": "Insufficient cash balance"}
            
            # Check sufficient shares for sell orders
            if order_request.side == OrderSide.SELL:
                # Get current position
                position = db.query(Position).filter(
                    Position.portfolio_id == order_request.portfolio_id,
                    Position.symbol == order_request.symbol
                ).first()
                
                if not position or position.quantity < order_request.quantity:
                    return {"valid": False, "reason": "Insufficient shares to sell"}
            
            return {"valid": True, "reason": "Order validation passed"}
            
        except Exception as e:
            logger.error(f"Error validating order: {str(e)}")
            return {"valid": False, "reason": f"Validation error: {str(e)}"}
    
    async def _assess_order_risk(self, order_request: OrderRequest, db: Session) -> Dict[str, Any]:
        """Assess risk for the order"""
        try:
            # Get current portfolio
            portfolio_response = await self.portfolio_manager.get_portfolio(
                order_request.portfolio_id, db
            )
            
            # Calculate order impact
            current_price = await self.market_service.get_current_price(
                order_request.symbol
            )
            order_value = order_request.quantity * current_price
            total_portfolio_value = portfolio_response.summary.total_value
            
            # Position concentration risk
            position_weight = (order_value / total_portfolio_value) * 100 if total_portfolio_value > 0 else 0
            
            # Volatility risk (simplified)
            volatility_risk = 50  # Would calculate from historical data
            
            # Liquidity risk (simplified)
            liquidity_risk = 20  # Would calculate from volume data
            
            # Overall risk score
            risk_score = min(
                (position_weight * 0.4) + (volatility_risk * 0.4) + (liquidity_risk * 0.2),
                100
            )
            
            return {
                "risk_score": risk_score,
                "position_weight": position_weight,
                "volatility_risk": volatility_risk,
                "liquidity_risk": liquidity_risk,
                "order_value": order_value,
                "portfolio_impact": (order_value / total_portfolio_value) * 100 if total_portfolio_value > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error assessing order risk: {str(e)}")
            return {"risk_score": 0}
    
    async def _check_risk_limits(self, order_request: OrderRequest, risk_assessment: Dict[str, Any], db: Session) -> Dict[str, Any]:
        """Check order against risk limits"""
        try:
            # Check position concentration
            if risk_assessment.get("position_weight", 0) > self.max_position_concentration * 100:
                return {
                    "passed": False,
                    "reason": f"Position concentration exceeds limit: {risk_assessment['position_weight']:.1f}%"
                }
            
            # Check overall risk score
            if risk_assessment.get("risk_score", 0) > 80:
                return {
                    "passed": False,
                    "reason": f"Risk score too high: {risk_assessment['risk_score']:.1f}"
                }
            
            return {"passed": True, "reason": "Risk checks passed"}
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {str(e)}")
            return {"passed": False, "reason": f"Risk check error: {str(e)}"}
    
    async def _calculate_estimated_cost(self, order_request: OrderRequest) -> float:
        """Calculate estimated cost for order"""
        try:
            # Get current price
            current_price = await self.market_service.get_current_price(
                order_request.symbol
            )
            
            # Use limit price if available, otherwise current price
            execution_price = order_request.limit_price or current_price
            
            # Calculate order value
            order_value = order_request.quantity * execution_price
            
            # Calculate commission
            commission = max(
                order_value * self.commission_rate,
                self.min_commission
            )
            
            # Total cost (for buy orders) or net proceeds (for sell orders)
            if order_request.side == OrderSide.BUY:
                return order_value + commission
            else:
                return order_value - commission
            
        except Exception as e:
            logger.error(f"Error calculating estimated cost: {str(e)}")
            return 0
    
    async def _execute_order(self, order_request: OrderRequest, order_id: str) -> Dict[str, Any]:
        """Execute order (simulated)"""
        try:
            # Simulate order execution based on order type
            current_price = await self.market_service.get_current_price(
                order_request.symbol
            )
            
            if order_request.order_type == OrderType.MARKET:
                # Market orders execute immediately at current price
                fill_price = current_price
                filled_quantity = order_request.quantity
                status = OrderStatus.FILLED.value
                
            elif order_request.order_type == OrderType.LIMIT:
                # Limit orders may or may not execute
                if order_request.side == OrderSide.BUY:
                    if current_price <= order_request.limit_price:
                        fill_price = order_request.limit_price
                        filled_quantity = order_request.quantity
                        status = OrderStatus.FILLED.value
                    else:
                        fill_price = None
                        filled_quantity = 0
                        status = OrderStatus.SUBMITTED.value
                else:  # SELL
                    if current_price >= order_request.limit_price:
                        fill_price = order_request.limit_price
                        filled_quantity = order_request.quantity
                        status = OrderStatus.FILLED.value
                    else:
                        fill_price = None
                        filled_quantity = 0
                        status = OrderStatus.SUBMITTED.value
            
            else:
                # Other order types (simplified)
                fill_price = current_price
                filled_quantity = order_request.quantity
                status = OrderStatus.FILLED.value
            
            # Calculate commission
            commission = 0
            if filled_quantity > 0:
                order_value = filled_quantity * fill_price
                commission = max(
                    order_value * self.commission_rate,
                    self.min_commission
                )
            
            return {
                "status": status,
                "filled_quantity": filled_quantity,
                "average_fill_price": fill_price,
                "commission": commission,
                "execution_venue": np.random.choice(self.execution_venues)
            }
            
        except Exception as e:
            logger.error(f"Error executing order: {str(e)}")
            return {
                "status": OrderStatus.REJECTED.value,
                "filled_quantity": 0,
                "average_fill_price": None,
                "commission": 0
            }
    
    async def _update_portfolio_from_fill(self, order_request: OrderRequest, order_response: OrderResponse, db: Session):
        """Update portfolio after order fill"""
        try:
            # Create transaction record for portfolio manager
            from .portfolio_manager import TransactionRequest, TransactionType
            
            transaction_type = TransactionType.BUY if order_request.side == OrderSide.BUY else TransactionType.SELL
            
            transaction_request = TransactionRequest(
                portfolio_id=order_request.portfolio_id,
                symbol=order_request.symbol,
                transaction_type=transaction_type,
                quantity=order_response.filled_quantity,
                price=order_response.average_fill_price,
                fee=order_response.total_commission,
                notes=f"Order {order_response.order_id} execution"
            )
            
            await self.portfolio_manager.add_transaction(transaction_request, db)
            
        except Exception as e:
            logger.error(f"Error updating portfolio from fill: {str(e)}")
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID"""
        import uuid
        return f"ORD_{uuid.uuid4().hex[:8].upper()}"

# Global trading engine instance
trading_engine = TradingEngine()

def get_trading_engine() -> TradingEngine:
    """Get trading engine instance"""
    return trading_engine