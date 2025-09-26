from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc
import asyncio
import json
import logging
from decimal import Decimal, ROUND_HALF_UP
import redis
import uuid

from database import get_db
from db_models import User, Portfolio, Order, Trade, Position, Asset
from market_data import MarketDataService
from websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)

class PaperTradingService:
    def __init__(self):
        self.market_data_service = MarketDataService()
        self.websocket_manager = WebSocketManager()
        self.active_orders: Dict[str, Order] = {}
        self.order_processors = {}
        
        # Redis connection for real-time order processing
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            self.redis_client.ping()
        except redis.ConnectionError:
            logger.warning("Redis not available, falling back to in-memory processing")
            self.redis_client = None
        
        # Start background task for processing orders
        asyncio.create_task(self._start_order_processor())
        
    async def create_paper_portfolio(
        self,
        db: Session,
        user_id: str,
        name: str,
        initial_balance: float = 100000.0,
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new paper trading portfolio"""
        try:
            # Check if user already has a paper portfolio with this name
            existing = db.query(Portfolio).filter(
                and_(
                    Portfolio.user_id == user_id,
                    Portfolio.name == name,
                    Portfolio.portfolio_type == 'paper'
                )
            ).first()
            
            if existing:
                raise ValueError("Portfolio with this name already exists")
            
            # Create new portfolio
            portfolio = Portfolio(
                user_id=user_id,
                name=name,
                description=description or f"Paper trading portfolio - {name}",
                portfolio_type='paper',
                initial_balance=initial_balance,
                current_balance=initial_balance,
                total_value=initial_balance,
                is_active=True
            )
            
            db.add(portfolio)
            db.commit()
            db.refresh(portfolio)
            
            logger.info(f"Paper portfolio created: {portfolio.id} for user {user_id}")
            
            return {
                'portfolio_id': portfolio.id,
                'name': portfolio.name,
                'initial_balance': portfolio.initial_balance,
                'current_balance': portfolio.current_balance,
                'created_at': portfolio.created_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating paper portfolio: {str(e)}")
            raise

    async def create_paper_order(
        self,
        db: Session,
        user_id: str,
        portfolio_id: str,
        symbol: str,
        side: str,  # 'buy' or 'sell'
        quantity: float,
        order_type: str = 'market',  # 'market', 'limit', 'stop', 'stop_limit'
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = 'day'  # 'day', 'gtc', 'ioc', 'fok'
    ) -> Dict[str, Any]:
        """Create a new paper trading order"""
        try:
            # Validate portfolio ownership and type
            portfolio = db.query(Portfolio).filter(
                and_(
                    Portfolio.id == portfolio_id, 
                    Portfolio.user_id == user_id,
                    Portfolio.portfolio_type == 'paper'
                )
            ).first()
            
            if not portfolio:
                raise ValueError("Paper trading portfolio not found or access denied")
            
            # Validate symbol format
            symbol = symbol.upper().strip()
            if not symbol or len(symbol) < 1:
                raise ValueError("Invalid symbol")
            
            # Validate quantity
            if quantity <= 0:
                raise ValueError("Quantity must be positive")
            
            # Get current market price
            current_price = await self.market_data_service.get_current_price(symbol)
            if not current_price:
                raise ValueError(f"Unable to get current price for {symbol}")
            
            # Validate order parameters
            if order_type == 'limit' and not limit_price:
                raise ValueError("Limit price required for limit orders")
            
            if order_type in ['stop', 'stop_limit'] and not stop_price:
                raise ValueError("Stop price required for stop orders")
            
            if limit_price and limit_price <= 0:
                raise ValueError("Limit price must be positive")
            
            if stop_price and stop_price <= 0:
                raise ValueError("Stop price must be positive")
            
            # Risk management checks
            await self._validate_risk_limits(db, portfolio, symbol, side, quantity, current_price)
            
            # Check buying power for buy orders
            if side == 'buy':
                required_capital = self._calculate_required_capital(
                    quantity, current_price, limit_price, order_type
                )
                available_cash = await self._get_available_cash(db, portfolio_id)
                
                if required_capital > available_cash:
                    raise ValueError(f"Insufficient buying power. Required: ${required_capital:,.2f}, Available: ${available_cash:,.2f}")
            
            # Check position for sell orders
            elif side == 'sell':
                available_quantity = await self._get_available_quantity(db, portfolio_id, symbol)
                if quantity > available_quantity:
                    raise ValueError(f"Insufficient position to sell. Requested: {quantity}, Available: {available_quantity}")
            
            # Create order with unique ID
            order_id = str(uuid.uuid4())
            order = Order(
                id=order_id,
                portfolio_id=portfolio_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                limit_price=limit_price,
                stop_price=stop_price,
                time_in_force=time_in_force,
                status='pending',
                created_at=datetime.utcnow()
            )
            
            db.add(order)
            
            # Add to active orders for processing
            self.active_orders[order_id] = {
                'order': order,
                'created_at': datetime.utcnow(),
                'last_check': datetime.utcnow()
            }
            
            # Add to Redis queue for processing
            if self.redis_client:
                order_data = {
                    'order_id': order_id,
                    'user_id': user_id,
                    'portfolio_id': portfolio_id,
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'order_type': order_type,
                    'limit_price': limit_price,
                    'stop_price': stop_price,
                    'current_price': current_price,
                    'created_at': datetime.utcnow().isoformat()
                }
                self.redis_client.lpush('paper_trading_orders', json.dumps(order_data))
                self.redis_client.publish('order_updates', json.dumps({
                    'type': 'order_created',
                    'user_id': user_id,
                    'order': order_data
                }))
            
            # Start order processing task if not already running
            if not hasattr(self, '_order_processor_task') or self._order_processor_task.done():
                self._order_processor_task = asyncio.create_task(self._process_orders())
            
            # Commit transaction
            db.commit()
            db.refresh(order)
            
            # Notify user about order creation
            await self._notify_order_update(user_id, order, 'created')
            
            logger.info(f"Paper order created: {order_id} for portfolio {portfolio_id}")
            
            return {
                'order_id': order_id,
                'status': 'pending',
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'order_type': order_type,
                'limit_price': limit_price,
                'stop_price': stop_price,
                'created_at': order.created_at.isoformat(),
                'estimated_value': quantity * current_price
            }
            
        except Exception as e:
            logger.error(f"Error creating paper order: {str(e)}")
            db.rollback()
            raise

    async def _validate_risk_limits(
        self,
        db: Session,
        portfolio: Portfolio,
        symbol: str,
        side: str,
        quantity: float,
        current_price: float
    ):
        """Validate risk management limits"""
        try:
            # Position size limit (max 10% of portfolio per position)
            position_value = quantity * current_price
            max_position_value = portfolio.total_value * 0.10
            
            if position_value > max_position_value:
                raise ValueError(f"Position size exceeds 10% limit. Max allowed: ${max_position_value:,.2f}")
            
            # Daily trading limit (max 25% of portfolio per day)
            today = datetime.utcnow().date()
            daily_volume = db.query(func.sum(Trade.quantity * Trade.price)).filter(
                and_(
                    Trade.portfolio_id == portfolio.id,
                    func.date(Trade.executed_at) == today
                )
            ).scalar() or 0
            
            max_daily_volume = portfolio.total_value * 0.25
            if daily_volume + position_value > max_daily_volume:
                raise ValueError(f"Daily trading limit exceeded. Limit: ${max_daily_volume:,.2f}")
            
            # Concentration limit (max 20% in single symbol)
            current_position = await self._get_position_value(db, portfolio.id, symbol)
            new_position_value = current_position + (position_value if side == 'buy' else -position_value)
            max_concentration = portfolio.total_value * 0.20
            
            if new_position_value > max_concentration:
                raise ValueError(f"Symbol concentration exceeds 20% limit. Max allowed: ${max_concentration:,.2f}")
                
        except Exception as e:
            logger.error(f"Risk validation error: {str(e)}")
            raise
    
    async def _process_order(self, db: Session, order: Order):
        """Process order based on market conditions"""
        try:
            while order.status == 'pending' and order.id in self.active_orders:
                current_price = await self.market_data_service.get_current_price(order.symbol)
                
                if not current_price:
                    await asyncio.sleep(1)
                    continue
                
                should_execute = False
                execution_price = current_price
                
                # Check execution conditions based on order type
                if order.order_type == 'market':
                    should_execute = True
                    execution_price = current_price
                
                elif order.order_type == 'limit':
                    if order.side == 'buy' and current_price <= order.limit_price:
                        should_execute = True
                        execution_price = min(order.limit_price, current_price)
                    elif order.side == 'sell' and current_price >= order.limit_price:
                        should_execute = True
                        execution_price = max(order.limit_price, current_price)
                
                elif order.order_type == 'stop':
                    if order.side == 'buy' and current_price >= order.stop_price:
                        should_execute = True
                        execution_price = current_price
                    elif order.side == 'sell' and current_price <= order.stop_price:
                        should_execute = True
                        execution_price = current_price
                
                elif order.order_type == 'stop_limit':
                    if order.side == 'buy' and current_price >= order.stop_price:
                        # Convert to limit order
                        order.order_type = 'limit'
                        db.commit()
                    elif order.side == 'sell' and current_price <= order.stop_price:
                        # Convert to limit order
                        order.order_type = 'limit'
                        db.commit()
                
                if should_execute:
                    await self._execute_order(db, order, execution_price)
                    break
                
                # Check if order should expire
                if self._should_expire_order(order):
                    await self._cancel_order(db, order, 'expired')
                    break
                
                await asyncio.sleep(0.1)  # Check every 100ms for real-time execution
                
        except Exception as e:
            logger.error(f"Error processing order {order.id}: {str(e)}")
            await self._cancel_order(db, order, 'error')
        finally:
            # Clean up order from active processing
            if order.id in self.active_orders:
                del self.active_orders[order.id]
            if order.id in self.order_processors:
                del self.order_processors[order.id]
    
    async def _execute_order(self, db: Session, order: Order, execution_price: float):
        """Execute the order at the given price"""
        try:
            # Calculate commission (simulate broker fees)
            commission = self._calculate_commission(order.quantity, execution_price)
            
            # Create trade record
            trade = Trade(
                order_id=order.id,
                portfolio_id=order.portfolio_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=execution_price,
                commission=commission
            )
            
            db.add(trade)
            
            # Update order status
            order.status = 'filled'
            order.filled_quantity = order.quantity
            order.average_fill_price = execution_price
            order.commission = commission
            order.updated_at = datetime.utcnow()
            
            # Update or create position
            await self._update_position(db, order, trade)
            
            db.commit()
            
            # Remove from active orders
            if order.id in self.active_orders:
                del self.active_orders[order.id]
            if order.id in self.order_processors:
                del self.order_processors[order.id]
            
            # Get user_id for notification
            portfolio = db.query(Portfolio).filter(Portfolio.id == order.portfolio_id).first()
            user_id = portfolio.user_id if portfolio else None
            
            # Publish execution to Redis
            if self.redis_client and user_id:
                execution_data = {
                    'type': 'order_executed',
                    'order_id': order.id,
                    'user_id': user_id,
                    'symbol': order.symbol,
                    'side': order.side,
                    'quantity': order.quantity,
                    'price': execution_price,
                    'commission': commission,
                    'executed_at': datetime.utcnow().isoformat()
                }
                self.redis_client.publish('order_updates', json.dumps(execution_data))
            
            # Notify user
            if user_id:
                await self._notify_order_update(user_id, order, 'filled')
                await self._notify_trade_execution(user_id, trade)
            
            logger.info(f"Order {order.id} executed at {execution_price} for {order.quantity} shares")
            
        except Exception as e:
            logger.error(f"Error executing order {order.id}: {str(e)}")
            raise
    
    async def _update_position(self, db: Session, order: Order, trade: Trade):
        """Update portfolio position after trade execution"""
        try:
            position = db.query(Position).filter(
                and_(
                    Position.portfolio_id == order.portfolio_id,
                    Position.symbol == order.symbol
                )
            ).first()
            
            if not position:
                # Create new position
                if order.side == 'buy':
                    position = Position(
                        portfolio_id=order.portfolio_id,
                        symbol=order.symbol,
                        quantity=trade.quantity,
                        average_cost=trade.price
                    )
                    db.add(position)
            else:
                # Update existing position
                if order.side == 'buy':
                    # Calculate new average cost
                    total_cost = (position.quantity * position.average_cost) + (trade.quantity * trade.price)
                    total_quantity = position.quantity + trade.quantity
                    position.average_cost = total_cost / total_quantity
                    position.quantity = total_quantity
                    
                elif order.side == 'sell':
                    # Calculate realized P&L
                    realized_pnl = (trade.price - position.average_cost) * trade.quantity
                    position.realized_pnl += realized_pnl
                    position.quantity -= trade.quantity
                    
                    # Remove position if quantity is zero
                    if position.quantity <= 0:
                        db.delete(position)
                        return
            
            # Update market value and unrealized P&L
            current_price = await self.market_data_service.get_current_price(order.symbol)
            if current_price and position.quantity > 0:
                position.market_value = position.quantity * current_price
                position.unrealized_pnl = (current_price - position.average_cost) * position.quantity
            
            position.updated_at = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error updating position: {str(e)}")
            raise
    
    async def cancel_order(self, db: Session, user_id: str, order_id: str) -> Dict[str, Any]:
        """Cancel a pending order"""
        try:
            order = db.query(Order).join(Portfolio).filter(
                and_(
                    Order.id == order_id,
                    Portfolio.user_id == user_id,
                    Order.status == 'pending'
                )
            ).first()
            
            if not order:
                raise ValueError("Order not found or cannot be cancelled")
            
            await self._cancel_order(db, order, 'cancelled')
            
            return {
                'order_id': order_id,
                'status': 'cancelled',
                'cancelled_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {str(e)}")
            raise
    
    async def _cancel_order(self, db: Session, order: Order, reason: str):
        """Internal method to cancel an order"""
        try:
            order.status = 'cancelled'
            order.updated_at = datetime.utcnow()
            order.notes = f"Cancelled: {reason}"
            
            db.commit()
            
            # Remove from active orders
            if order.id in self.active_orders:
                del self.active_orders[order.id]
            
            # Get user_id for notification
            portfolio = db.query(Portfolio).filter(Portfolio.id == order.portfolio_id).first()
            user_id = portfolio.user_id if portfolio else None
            
            if user_id:
                await self._notify_order_update(user_id, order, 'cancelled')
            
        except Exception as e:
            logger.error(f"Error in _cancel_order: {str(e)}")
            raise
    
    async def get_portfolio_summary(self, db: Session, user_id: str, portfolio_id: str) -> Dict[str, Any]:
        """Get real-time portfolio summary"""
        try:
            portfolio = db.query(Portfolio).filter(
                and_(Portfolio.id == portfolio_id, Portfolio.user_id == user_id)
            ).first()
            
            if not portfolio:
                raise ValueError("Portfolio not found")
            
            # Get all positions
            positions = db.query(Position).filter(Position.portfolio_id == portfolio_id).all()
            
            total_value = 0
            total_cost = 0
            total_unrealized_pnl = 0
            total_realized_pnl = 0
            
            position_details = []
            
            for position in positions:
                current_price = await self.market_data_service.get_current_price(position.symbol)
                if current_price:
                    market_value = position.quantity * current_price
                    unrealized_pnl = (current_price - position.average_cost) * position.quantity
                    
                    # Update position in database
                    position.market_value = market_value
                    position.unrealized_pnl = unrealized_pnl
                    
                    total_value += market_value
                    total_cost += position.quantity * position.average_cost
                    total_unrealized_pnl += unrealized_pnl
                    total_realized_pnl += position.realized_pnl
                    
                    position_details.append({
                        'symbol': position.symbol,
                        'quantity': position.quantity,
                        'average_cost': position.average_cost,
                        'current_price': current_price,
                        'market_value': market_value,
                        'unrealized_pnl': unrealized_pnl,
                        'realized_pnl': position.realized_pnl,
                        'pnl_percentage': (unrealized_pnl / (position.quantity * position.average_cost)) * 100 if position.quantity > 0 else 0
                    })
            
            db.commit()
            
            # Get available cash (simplified - assume starting with $100,000)
            available_cash = await self._get_available_cash(db, portfolio_id)
            total_portfolio_value = total_value + available_cash
            
            return {
                'portfolio_id': portfolio_id,
                'total_value': total_portfolio_value,
                'total_invested': total_cost,
                'available_cash': available_cash,
                'total_unrealized_pnl': total_unrealized_pnl,
                'total_realized_pnl': total_realized_pnl,
                'total_pnl': total_unrealized_pnl + total_realized_pnl,
                'pnl_percentage': ((total_unrealized_pnl + total_realized_pnl) / total_cost) * 100 if total_cost > 0 else 0,
                'positions': position_details,
                'updated_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {str(e)}")
            raise
    
    def _calculate_required_capital(self, quantity: float, current_price: float, limit_price: Optional[float], order_type: str) -> float:
        """Calculate required capital for buy order"""
        if order_type == 'market':
            price = current_price
        elif order_type == 'limit' and limit_price:
            price = limit_price
        else:
            price = current_price
        
        return quantity * price * 1.01  # Add 1% buffer for price movements
    
    async def _get_available_cash(self, db: Session, portfolio_id: str) -> float:
        """Get available cash in portfolio (simplified implementation)"""
        # In a real implementation, this would track cash balance
        # For now, assume starting balance minus invested amount
        starting_balance = 100000.0  # $100,000 starting balance
        
        # Calculate total invested
        positions = db.query(Position).filter(Position.portfolio_id == portfolio_id).all()
        total_invested = sum(pos.quantity * pos.average_cost for pos in positions)
        
        return max(0, starting_balance - total_invested)
    
    async def _get_available_quantity(self, db: Session, portfolio_id: str, symbol: str) -> float:
        """Get available quantity for selling"""
        position = db.query(Position).filter(
            and_(Position.portfolio_id == portfolio_id, Position.symbol == symbol)
        ).first()
        
        return position.quantity if position else 0
    
    def _calculate_commission(self, quantity: float, price: float) -> float:
        """Calculate trading commission (simplified)"""
        # Simulate commission: $0.005 per share, minimum $1
        commission = max(1.0, quantity * 0.005)
        return round(commission, 2)
    
    def _should_expire_order(self, order: Order) -> bool:
        """Check if order should expire based on time_in_force"""
        if order.time_in_force == 'day':
            # Expire at end of trading day (simplified)
            return datetime.utcnow() - order.created_at > timedelta(hours=8)
        return False
    
    async def _notify_order_update(self, user_id: str, order: Order, event_type: str):
        """Send order update via WebSocket"""
        try:
            message = {
                'type': 'order_update',
                'event': event_type,
                'data': {
                    'order_id': order.id,
                    'symbol': order.symbol,
                    'side': order.side,
                    'quantity': order.quantity,
                    'status': order.status,
                    'filled_quantity': order.filled_quantity,
                    'average_fill_price': order.average_fill_price,
                    'timestamp': datetime.utcnow().isoformat()
                }
            }
            
            await self.websocket_manager.send_to_user(user_id, json.dumps(message))
            
        except Exception as e:
            logger.error(f"Error sending order update notification: {str(e)}")
    
    async def _notify_trade_execution(self, user_id: str, trade: Trade):
        """Send trade execution notification via WebSocket"""
        try:
            message = {
                'type': 'trade_execution',
                'data': {
                    'trade_id': trade.id,
                    'symbol': trade.symbol,
                    'side': trade.side,
                    'quantity': trade.quantity,
                    'price': trade.price,
                    'commission': trade.commission,
                    'timestamp': trade.timestamp.isoformat()
                }
            }
            
            await self.websocket_manager.send_to_user(user_id, json.dumps(message))
            
        except Exception as e:
            logger.error(f"Error sending trade execution notification: {str(e)}")
    
    async def _start_order_processor(self):
        """Background task to process orders from Redis queue"""
        if not self.redis_client:
            return
            
        logger.info("Starting Redis-based order processor")
        
        while True:
            try:
                # Process orders from Redis queue
                order_data = self.redis_client.brpop('paper_trading_orders', timeout=1)
                
                if order_data:
                    _, order_json = order_data
                    order_info = json.loads(order_json)
                    
                    # Get database session
                    db = next(get_db())
                    
                    try:
                        # Find the order in database
                        order = db.query(Order).filter(Order.id == order_info['order_id']).first()
                        
                        if order and order.status == 'pending':
                            # Process the order if not already being processed
                            if order.id not in self.order_processors:
                                task = asyncio.create_task(self._process_order(db, order))
                                self.order_processors[order.id] = task
                                
                    except Exception as e:
                        logger.error(f"Error processing order from Redis queue: {str(e)}")
                    finally:
                        db.close()
                        
            except Exception as e:
                logger.error(f"Error in order processor: {str(e)}")
                await asyncio.sleep(1)

# Global instance
paper_trading_service = PaperTradingService()