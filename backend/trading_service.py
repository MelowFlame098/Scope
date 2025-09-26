import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
import json
import uuid
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc

from database import get_db, SessionLocal
from db_models import User, Portfolio, PortfolioHolding, Asset, TradingModel, ModelPrediction
from schemas import (
    PortfolioResponse, PortfolioHoldingResponse,
    AssetResponse, ModelPredictionResponse
)
from market_data import MarketDataService
from portfolio_service import PortfolioService
from notification_service import NotificationService, NotificationType, NotificationChannel

logger = logging.getLogger(__name__)

class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class TradingStrategy(str, Enum):
    BUY_AND_HOLD = "buy_and_hold"
    DOLLAR_COST_AVERAGING = "dollar_cost_averaging"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    RSI_OVERSOLD = "rsi_oversold"
    MOVING_AVERAGE_CROSSOVER = "ma_crossover"

class TradingService:
    def __init__(self):
        self.market_service = MarketDataService()
        self.portfolio_service = PortfolioService()
        self.notification_service = NotificationService()
        
        # Simulated order book for paper trading
        self.orders = {}
        self.trade_history = {}
        
        # Trading parameters
        self.max_position_size = 0.1  # 10% of portfolio
        self.max_daily_trades = 50
        self.min_trade_amount = 10.0
        
    async def create_order(
        self,
        user_id: str,
        portfolio_id: str,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "GTC",  # Good Till Cancelled
        db: Session = None
    ) -> Dict[str, Any]:
        """Create a trading order (paper trading simulation)."""
        try:
            if db is None:
                db = SessionLocal()
            
            # Validate portfolio ownership
            portfolio = await self.portfolio_service.get_portfolio(portfolio_id, user_id, db)
            if not portfolio:
                raise ValueError("Portfolio not found or access denied")
            
            # Get current market data
            asset_data = await self.market_service.get_asset_details(symbol)
            if not asset_data:
                raise ValueError(f"Asset {symbol} not found")
            
            current_price = asset_data.current_price
            
            # Validate order parameters
            if order_type == OrderType.MARKET:
                execution_price = current_price
            elif order_type == OrderType.LIMIT:
                if not price:
                    raise ValueError("Limit orders require a price")
                execution_price = price
            else:
                execution_price = price or current_price
            
            # Calculate order value
            order_value = quantity * execution_price
            
            # Validate minimum trade amount
            if order_value < self.min_trade_amount:
                raise ValueError(f"Order value must be at least ${self.min_trade_amount}")
            
            # Check portfolio constraints
            await self._validate_order_constraints(
                user_id, portfolio_id, symbol, side, order_value, db
            )
            
            # Create order
            order_id = str(uuid.uuid4())
            order = {
                "id": order_id,
                "user_id": user_id,
                "portfolio_id": portfolio_id,
                "symbol": symbol,
                "side": side,
                "order_type": order_type,
                "quantity": quantity,
                "price": price,
                "stop_price": stop_price,
                "execution_price": execution_price,
                "status": OrderStatus.PENDING,
                "filled_quantity": 0,
                "remaining_quantity": quantity,
                "time_in_force": time_in_force,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            self.orders[order_id] = order
            
            # Execute order if market order or conditions are met
            if order_type == OrderType.MARKET or self._should_execute_order(order, current_price):
                await self._execute_order(order_id, current_price, db)
            
            logger.info(f"Order created: {order_id} for user {user_id}")
            
            return order
            
        except Exception as e:
            logger.error(f"Error creating order: {e}")
            raise
        finally:
            if db:
                db.close()
    
    async def cancel_order(
        self,
        order_id: str,
        user_id: str
    ) -> bool:
        """Cancel a pending order."""
        try:
            order = self.orders.get(order_id)
            if not order:
                return False
            
            if order["user_id"] != user_id:
                raise ValueError("Access denied")
            
            if order["status"] not in [OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]:
                raise ValueError("Order cannot be cancelled")
            
            order["status"] = OrderStatus.CANCELLED
            order["updated_at"] = datetime.utcnow()
            
            logger.info(f"Order cancelled: {order_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            raise
    
    async def get_orders(
        self,
        user_id: str,
        portfolio_id: Optional[str] = None,
        status: Optional[OrderStatus] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get user's orders."""
        try:
            user_orders = [
                order for order in self.orders.values()
                if order["user_id"] == user_id
            ]
            
            # Apply filters
            if portfolio_id:
                user_orders = [
                    order for order in user_orders
                    if order["portfolio_id"] == portfolio_id
                ]
            
            if status:
                user_orders = [
                    order for order in user_orders
                    if order["status"] == status
                ]
            
            # Sort by creation time (newest first)
            user_orders.sort(key=lambda x: x["created_at"], reverse=True)
            
            return user_orders[:limit]
            
        except Exception as e:
            logger.error(f"Error getting orders for user {user_id}: {e}")
            return []
    
    async def get_trade_history(
        self,
        user_id: str,
        portfolio_id: Optional[str] = None,
        symbol: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get user's trade history."""
        try:
            user_trades = [
                trade for trade in self.trade_history.values()
                if trade["user_id"] == user_id
            ]
            
            # Apply filters
            if portfolio_id:
                user_trades = [
                    trade for trade in user_trades
                    if trade["portfolio_id"] == portfolio_id
                ]
            
            if symbol:
                user_trades = [
                    trade for trade in user_trades
                    if trade["symbol"] == symbol
                ]
            
            # Sort by execution time (newest first)
            user_trades.sort(key=lambda x: x["executed_at"], reverse=True)
            
            return user_trades[:limit]
            
        except Exception as e:
            logger.error(f"Error getting trade history for user {user_id}: {e}")
            return []
    
    async def execute_strategy(
        self,
        user_id: str,
        portfolio_id: str,
        strategy: TradingStrategy,
        parameters: Dict[str, Any],
        db: Session = None
    ) -> Dict[str, Any]:
        """Execute a trading strategy."""
        try:
            if db is None:
                db = SessionLocal()
            
            # Validate portfolio
            portfolio = await self.portfolio_service.get_portfolio(portfolio_id, user_id, db)
            if not portfolio:
                raise ValueError("Portfolio not found or access denied")
            
            # Execute strategy based on type
            if strategy == TradingStrategy.DOLLAR_COST_AVERAGING:
                return await self._execute_dca_strategy(user_id, portfolio_id, parameters, db)
            elif strategy == TradingStrategy.MOMENTUM:
                return await self._execute_momentum_strategy(user_id, portfolio_id, parameters, db)
            elif strategy == TradingStrategy.MEAN_REVERSION:
                return await self._execute_mean_reversion_strategy(user_id, portfolio_id, parameters, db)
            elif strategy == TradingStrategy.RSI_OVERSOLD:
                return await self._execute_rsi_strategy(user_id, portfolio_id, parameters, db)
            elif strategy == TradingStrategy.MOVING_AVERAGE_CROSSOVER:
                return await self._execute_ma_crossover_strategy(user_id, portfolio_id, parameters, db)
            else:
                raise ValueError(f"Strategy {strategy} not implemented")
                
        except Exception as e:
            logger.error(f"Error executing strategy {strategy}: {e}")
            raise
        finally:
            if db:
                db.close()
    
    async def get_strategy_recommendations(
        self,
        user_id: str,
        portfolio_id: str,
        db: Session = None
    ) -> List[Dict[str, Any]]:
        """Get trading strategy recommendations based on portfolio and market conditions."""
        try:
            if db is None:
                db = SessionLocal()
            
            # Get portfolio performance
            performance = await self.portfolio_service.calculate_portfolio_performance(
                portfolio_id, user_id, db
            )
            
            recommendations = []
            
            # Analyze portfolio and generate recommendations
            if performance["total_value"] > 0:
                # Check for rebalancing opportunities
                if len(performance["holdings_performance"]) > 1:
                    recommendations.append({
                        "strategy": TradingStrategy.MEAN_REVERSION,
                        "reason": "Portfolio shows diversification - mean reversion strategy could help rebalance",
                        "confidence": 0.7,
                        "parameters": {
                            "lookback_period": 20,
                            "threshold": 0.02
                        }
                    })
                
                # Check for momentum opportunities
                positive_performers = [
                    holding for holding in performance["holdings_performance"]
                    if holding["profit_loss_percentage"] > 5
                ]
                
                if positive_performers:
                    recommendations.append({
                        "strategy": TradingStrategy.MOMENTUM,
                        "reason": f"{len(positive_performers)} holdings showing strong positive momentum",
                        "confidence": 0.8,
                        "parameters": {
                            "momentum_period": 14,
                            "threshold": 0.05
                        }
                    })
                
                # Always recommend DCA for long-term growth
                recommendations.append({
                    "strategy": TradingStrategy.DOLLAR_COST_AVERAGING,
                    "reason": "Consistent investment strategy for long-term growth",
                    "confidence": 0.9,
                    "parameters": {
                        "amount": min(1000, performance["total_value"] * 0.05),
                        "frequency": "weekly",
                        "symbols": ["BTC", "ETH", "AAPL", "SPY"]
                    }
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting strategy recommendations: {e}")
            return []
        finally:
            if db:
                db.close()
    
    async def _validate_order_constraints(
        self,
        user_id: str,
        portfolio_id: str,
        symbol: str,
        side: OrderSide,
        order_value: float,
        db: Session
    ):
        """Validate order against portfolio constraints."""
        # Get portfolio performance
        performance = await self.portfolio_service.calculate_portfolio_performance(
            portfolio_id, user_id, db
        )
        
        total_value = performance["total_value"]
        
        if side == OrderSide.BUY:
            # Check if order exceeds maximum position size
            max_order_value = total_value * self.max_position_size
            if order_value > max_order_value:
                raise ValueError(f"Order value exceeds maximum position size ({self.max_position_size * 100}%)")
            
            # Check if user has sufficient buying power (simplified)
            # In real implementation, this would check actual cash balance
            available_cash = total_value * 0.1  # Assume 10% cash available
            if order_value > available_cash:
                raise ValueError("Insufficient buying power")
        
        elif side == OrderSide.SELL:
            # Check if user has sufficient holdings
            holding = next(
                (h for h in performance["holdings_performance"] if h["symbol"] == symbol),
                None
            )
            
            if not holding:
                raise ValueError(f"No holdings found for {symbol}")
            
            if order_value > holding["current_value"]:
                raise ValueError("Insufficient holdings to sell")
    
    def _should_execute_order(self, order: Dict[str, Any], current_price: float) -> bool:
        """Check if order should be executed based on current market price."""
        order_type = order["order_type"]
        side = order["side"]
        price = order["price"]
        stop_price = order["stop_price"]
        
        if order_type == OrderType.MARKET:
            return True
        elif order_type == OrderType.LIMIT:
            if side == OrderSide.BUY:
                return current_price <= price
            else:  # SELL
                return current_price >= price
        elif order_type == OrderType.STOP:
            if side == OrderSide.BUY:
                return current_price >= stop_price
            else:  # SELL
                return current_price <= stop_price
        elif order_type == OrderType.STOP_LIMIT:
            # First check stop condition
            stop_triggered = False
            if side == OrderSide.BUY:
                stop_triggered = current_price >= stop_price
            else:  # SELL
                stop_triggered = current_price <= stop_price
            
            # Then check limit condition if stop is triggered
            if stop_triggered:
                if side == OrderSide.BUY:
                    return current_price <= price
                else:  # SELL
                    return current_price >= price
        
        return False
    
    async def _execute_order(
        self,
        order_id: str,
        execution_price: float,
        db: Session
    ):
        """Execute an order and update portfolio."""
        try:
            order = self.orders[order_id]
            
            # Calculate execution details
            quantity = order["remaining_quantity"]
            total_value = quantity * execution_price
            
            # Update order status
            order["status"] = OrderStatus.FILLED
            order["filled_quantity"] = order["quantity"]
            order["remaining_quantity"] = 0
            order["execution_price"] = execution_price
            order["updated_at"] = datetime.utcnow()
            
            # Create trade record
            trade_id = str(uuid.uuid4())
            trade = {
                "id": trade_id,
                "order_id": order_id,
                "user_id": order["user_id"],
                "portfolio_id": order["portfolio_id"],
                "symbol": order["symbol"],
                "side": order["side"],
                "quantity": quantity,
                "price": execution_price,
                "total_value": total_value,
                "executed_at": datetime.utcnow()
            }
            
            self.trade_history[trade_id] = trade
            
            # Update portfolio holdings
            if order["side"] == OrderSide.BUY:
                # Add to portfolio
                from schemas import PortfolioHoldingCreate
                holding_data = PortfolioHoldingCreate(
                    symbol=order["symbol"],
                    quantity=quantity,
                    average_cost=execution_price
                )
                
                await self.portfolio_service.add_holding(
                    order["portfolio_id"],
                    order["user_id"],
                    holding_data,
                    db
                )
            else:  # SELL
                # Remove from portfolio (simplified - would need more complex logic for partial sales)
                holding_data = PortfolioHoldingCreate(
                    symbol=order["symbol"],
                    quantity=-quantity,  # Negative quantity to reduce holdings
                    average_cost=execution_price
                )
                
                await self.portfolio_service.add_holding(
                    order["portfolio_id"],
                    order["user_id"],
                    holding_data,
                    db
                )
            
            # Send notification
            await self._notify_order_execution(order, trade)
            
            logger.info(f"Order executed: {order_id} at price {execution_price}")
            
        except Exception as e:
            logger.error(f"Error executing order {order_id}: {e}")
            # Update order status to rejected
            order["status"] = OrderStatus.REJECTED
            order["updated_at"] = datetime.utcnow()
            raise
    
    async def _execute_dca_strategy(
        self,
        user_id: str,
        portfolio_id: str,
        parameters: Dict[str, Any],
        db: Session
    ) -> Dict[str, Any]:
        """Execute Dollar Cost Averaging strategy."""
        amount = parameters.get("amount", 100)
        symbols = parameters.get("symbols", ["BTC", "ETH"])
        
        results = []
        amount_per_symbol = amount / len(symbols)
        
        for symbol in symbols:
            try:
                # Get current price
                asset_data = await self.market_service.get_asset_details(symbol)
                if not asset_data:
                    continue
                
                current_price = asset_data.current_price
                quantity = amount_per_symbol / current_price
                
                # Create buy order
                order = await self.create_order(
                    user_id=user_id,
                    portfolio_id=portfolio_id,
                    symbol=symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=quantity,
                    db=db
                )
                
                results.append({
                    "symbol": symbol,
                    "order_id": order["id"],
                    "quantity": quantity,
                    "price": current_price,
                    "value": amount_per_symbol
                })
                
            except Exception as e:
                logger.error(f"Error executing DCA for {symbol}: {e}")
                results.append({
                    "symbol": symbol,
                    "error": str(e)
                })
        
        return {
            "strategy": TradingStrategy.DOLLAR_COST_AVERAGING,
            "total_amount": amount,
            "results": results
        }
    
    async def _execute_momentum_strategy(
        self,
        user_id: str,
        portfolio_id: str,
        parameters: Dict[str, Any],
        db: Session
    ) -> Dict[str, Any]:
        """Execute momentum trading strategy."""
        # Simplified momentum strategy implementation
        symbols = parameters.get("symbols", ["AAPL", "GOOGL", "MSFT"])
        momentum_period = parameters.get("momentum_period", 14)
        threshold = parameters.get("threshold", 0.05)
        
        results = []
        
        for symbol in symbols:
            try:
                # Get historical data for momentum calculation
                chart_data = await self.market_service.get_chart_data(
                    symbol, "1d", momentum_period + 5
                )
                
                if not chart_data or len(chart_data) < momentum_period:
                    continue
                
                # Calculate momentum (simplified)
                recent_prices = [point["close"] for point in chart_data[-momentum_period:]]
                momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
                
                if momentum > threshold:
                    # Strong positive momentum - buy signal
                    asset_data = await self.market_service.get_asset_details(symbol)
                    current_price = asset_data.current_price
                    
                    # Calculate position size (1% of portfolio)
                    performance = await self.portfolio_service.calculate_portfolio_performance(
                        portfolio_id, user_id, db
                    )
                    position_value = performance["total_value"] * 0.01
                    quantity = position_value / current_price
                    
                    order = await self.create_order(
                        user_id=user_id,
                        portfolio_id=portfolio_id,
                        symbol=symbol,
                        side=OrderSide.BUY,
                        order_type=OrderType.MARKET,
                        quantity=quantity,
                        db=db
                    )
                    
                    results.append({
                        "symbol": symbol,
                        "momentum": momentum,
                        "action": "buy",
                        "order_id": order["id"],
                        "quantity": quantity
                    })
                
            except Exception as e:
                logger.error(f"Error executing momentum strategy for {symbol}: {e}")
                results.append({
                    "symbol": symbol,
                    "error": str(e)
                })
        
        return {
            "strategy": TradingStrategy.MOMENTUM,
            "results": results
        }
    
    async def _execute_mean_reversion_strategy(
        self,
        user_id: str,
        portfolio_id: str,
        parameters: Dict[str, Any],
        db: Session
    ) -> Dict[str, Any]:
        """Execute mean reversion strategy."""
        # Placeholder for mean reversion strategy
        return {
            "strategy": TradingStrategy.MEAN_REVERSION,
            "message": "Mean reversion strategy not yet implemented",
            "results": []
        }
    
    async def _execute_rsi_strategy(
        self,
        user_id: str,
        portfolio_id: str,
        parameters: Dict[str, Any],
        db: Session
    ) -> Dict[str, Any]:
        """Execute RSI oversold strategy."""
        # Placeholder for RSI strategy
        return {
            "strategy": TradingStrategy.RSI_OVERSOLD,
            "message": "RSI strategy not yet implemented",
            "results": []
        }
    
    async def _execute_ma_crossover_strategy(
        self,
        user_id: str,
        portfolio_id: str,
        parameters: Dict[str, Any],
        db: Session
    ) -> Dict[str, Any]:
        """Execute moving average crossover strategy."""
        # Placeholder for MA crossover strategy
        return {
            "strategy": TradingStrategy.MOVING_AVERAGE_CROSSOVER,
            "message": "Moving average crossover strategy not yet implemented",
            "results": []
        }
    
    async def _notify_order_execution(self, order: Dict[str, Any], trade: Dict[str, Any]):
        """Notify user about order execution."""
        try:
            message = f"Order executed: {order['side'].upper()} {trade['quantity']} {order['symbol']} at ${trade['price']:.2f}"
            
            await self.notification_service.send_notification(
                user_id=order["user_id"],
                notification_type=NotificationType.TRADE_EXECUTION,
                message=message,
                channels=[NotificationChannel.WEBSOCKET, NotificationChannel.EMAIL],
                data={
                    "order_id": order["id"],
                    "trade_id": trade["id"],
                    "symbol": order["symbol"],
                    "side": order["side"],
                    "quantity": trade["quantity"],
                    "price": trade["price"],
                    "total_value": trade["total_value"]
                }
            )
            
        except Exception as e:
            logger.error(f"Error sending order execution notification: {e}")
    
    async def start_order_monitoring(self):
        """Start background task for monitoring and executing pending orders."""
        logger.info("Starting order monitoring service")
        
        while True:
            try:
                # Check pending orders every 10 seconds
                pending_orders = [
                    order for order in self.orders.values()
                    if order["status"] == OrderStatus.PENDING
                ]
                
                for order in pending_orders:
                    try:
                        # Get current market price
                        asset_data = await self.market_service.get_asset_details(order["symbol"])
                        if asset_data:
                            current_price = asset_data.current_price
                            
                            # Check if order should be executed
                            if self._should_execute_order(order, current_price):
                                db = SessionLocal()
                                try:
                                    await self._execute_order(order["id"], current_price, db)
                                finally:
                                    db.close()
                    
                    except Exception as e:
                        logger.error(f"Error processing order {order['id']}: {e}")
                        continue
                
                await asyncio.sleep(10)  # 10 seconds
                
            except Exception as e:
                logger.error(f"Error in order monitoring: {e}")
                await asyncio.sleep(30)  # 30 seconds on error

# Global trading service instance
trading_service = TradingService()