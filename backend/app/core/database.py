"""Database configuration and management for FinScope application.

This module provides:
- Database connection management
- Session handling
- Connection pooling
- Migration utilities
- Database health checks
- Transaction management
"""

from typing import Generator, Optional, Dict, Any, List
from sqlalchemy import (
    create_engine, MetaData, inspect, text, event,
    Engine, Connection, pool
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, scoped_session
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from sqlalchemy.pool import QueuePool, NullPool
from contextlib import contextmanager
import logging
from datetime import datetime, timedelta
import time
import threading
from dataclasses import dataclass

from app.config.settings import get_settings, DatabaseSettings
from app.core.exceptions import DatabaseException
from app.core.logging_config import get_logger

logger = get_logger("database")


@dataclass
class DatabaseStats:
    """Database connection statistics."""
    total_connections: int
    active_connections: int
    idle_connections: int
    pool_size: int
    max_overflow: int
    checked_out: int
    checked_in: int
    invalidated: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "total_connections": self.total_connections,
            "active_connections": self.active_connections,
            "idle_connections": self.idle_connections,
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "checked_out": self.checked_out,
            "checked_in": self.checked_in,
            "invalidated": self.invalidated
        }


class DatabaseManager:
    """Database connection and session manager."""
    
    def __init__(self, settings: Optional[DatabaseSettings] = None):
        """Initialize database manager.
        
        Args:
            settings: Database settings
        """
        self.settings = settings or get_settings().database
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None
        self._scoped_session: Optional[scoped_session] = None
        self._metadata: Optional[MetaData] = None
        self._base: Optional[declarative_base] = None
        self._lock = threading.Lock()
        self._connection_count = 0
        self._last_health_check = datetime.utcnow()
        self._health_check_interval = timedelta(minutes=5)
        
        logger.info("Database manager initialized")
    
    @property
    def engine(self) -> Engine:
        """Get database engine.
        
        Returns:
            SQLAlchemy engine
        """
        if self._engine is None:
            self._create_engine()
        return self._engine
    
    @property
    def session_factory(self) -> sessionmaker:
        """Get session factory.
        
        Returns:
            SQLAlchemy session factory
        """
        if self._session_factory is None:
            self._create_session_factory()
        return self._session_factory
    
    @property
    def scoped_session(self) -> scoped_session:
        """Get scoped session.
        
        Returns:
            SQLAlchemy scoped session
        """
        if self._scoped_session is None:
            self._create_scoped_session()
        return self._scoped_session
    
    @property
    def metadata(self) -> MetaData:
        """Get metadata instance.
        
        Returns:
            SQLAlchemy metadata
        """
        if self._metadata is None:
            self._metadata = MetaData()
        return self._metadata
    
    @property
    def base(self) -> declarative_base:
        """Get declarative base.
        
        Returns:
            SQLAlchemy declarative base
        """
        if self._base is None:
            self._base = declarative_base(metadata=self.metadata)
        return self._base
    
    def _create_engine(self) -> None:
        """Create database engine with connection pooling."""
        with self._lock:
            if self._engine is not None:
                return
            
            # Engine configuration
            engine_kwargs = {
                "echo": self.settings.echo_sql,
                "echo_pool": self.settings.echo_sql,
                "future": True,
                "connect_args": self._get_connect_args()
            }
            
            # Connection pooling configuration
            if self.settings.pool_size > 0:
                engine_kwargs.update({
                    "poolclass": QueuePool,
                    "pool_size": self.settings.pool_size,
                    "max_overflow": self.settings.max_overflow,
                    "pool_timeout": self.settings.pool_timeout,
                    "pool_recycle": self.settings.pool_recycle,
                    "pool_pre_ping": True
                })
            else:
                # Disable pooling for testing or special cases
                engine_kwargs["poolclass"] = NullPool
            
            try:
                self._engine = create_engine(self.settings.url, **engine_kwargs)
                
                # Set up event listeners
                self._setup_event_listeners()
                
                logger.info(f"Database engine created: {self.settings.url}")
                
            except Exception as e:
                logger.error(f"Failed to create database engine: {e}")
                raise DatabaseException(f"Database engine creation failed: {e}")
    
    def _get_connect_args(self) -> Dict[str, Any]:
        """Get database-specific connection arguments.
        
        Returns:
            Connection arguments
        """
        connect_args = {}
        
        # PostgreSQL specific
        if "postgresql" in self.settings.url:
            connect_args.update({
                "connect_timeout": 10,
                "application_name": "FinScope"
            })
        
        # MySQL specific
        elif "mysql" in self.settings.url:
            connect_args.update({
                "connect_timeout": 10,
                "charset": "utf8mb4"
            })
        
        # SQLite specific
        elif "sqlite" in self.settings.url:
            connect_args.update({
                "check_same_thread": False,
                "timeout": 10
            })
        
        return connect_args
    
    def _setup_event_listeners(self) -> None:
        """Set up SQLAlchemy event listeners."""
        @event.listens_for(self._engine, "connect")
        def on_connect(dbapi_connection, connection_record):
            """Handle new database connections."""
            self._connection_count += 1
            logger.debug(f"New database connection established (total: {self._connection_count})")
        
        @event.listens_for(self._engine, "checkout")
        def on_checkout(dbapi_connection, connection_record, connection_proxy):
            """Handle connection checkout from pool."""
            logger.debug("Database connection checked out from pool")
        
        @event.listens_for(self._engine, "checkin")
        def on_checkin(dbapi_connection, connection_record):
            """Handle connection checkin to pool."""
            logger.debug("Database connection checked in to pool")
        
        @event.listens_for(self._engine, "invalidate")
        def on_invalidate(dbapi_connection, connection_record, exception):
            """Handle connection invalidation."""
            logger.warning(f"Database connection invalidated: {exception}")
    
    def _create_session_factory(self) -> None:
        """Create session factory."""
        with self._lock:
            if self._session_factory is not None:
                return
            
            self._session_factory = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False
            )
            
            logger.info("Session factory created")
    
    def _create_scoped_session(self) -> None:
        """Create scoped session."""
        with self._lock:
            if self._scoped_session is not None:
                return
            
            self._scoped_session = scoped_session(self.session_factory)
            
            logger.info("Scoped session created")
    
    def get_session(self) -> Session:
        """Get a new database session.
        
        Returns:
            Database session
        """
        return self.session_factory()
    
    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """Provide a transactional scope around a series of operations.
        
        Yields:
            Database session
        """
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    @contextmanager
    def connection_scope(self) -> Generator[Connection, None, None]:
        """Provide a connection scope.
        
        Yields:
            Database connection
        """
        connection = self.engine.connect()
        try:
            yield connection
        finally:
            connection.close()
    
    def execute_raw_sql(self, sql: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute raw SQL query.
        
        Args:
            sql: SQL query
            params: Query parameters
            
        Returns:
            Query result
            
        Raises:
            DatabaseException: If query execution fails
        """
        try:
            with self.connection_scope() as connection:
                result = connection.execute(text(sql), params or {})
                return result
        except SQLAlchemyError as e:
            logger.error(f"Raw SQL execution failed: {e}")
            raise DatabaseException(f"SQL execution failed: {e}")
    
    def check_health(self) -> Dict[str, Any]:
        """Check database health.
        
        Returns:
            Health status information
        """
        try:
            start_time = time.time()
            
            with self.connection_scope() as connection:
                # Simple query to test connectivity
                result = connection.execute(text("SELECT 1"))
                result.fetchone()
            
            response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Get pool statistics
            stats = self.get_pool_stats()
            
            self._last_health_check = datetime.utcnow()
            
            return {
                "status": "healthy",
                "response_time_ms": round(response_time, 2),
                "last_check": self._last_health_check.isoformat(),
                "pool_stats": stats.to_dict() if stats else None
            }
        
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_check": datetime.utcnow().isoformat()
            }
    
    def get_pool_stats(self) -> Optional[DatabaseStats]:
        """Get connection pool statistics.
        
        Returns:
            Pool statistics or None if not available
        """
        try:
            if hasattr(self.engine.pool, 'size'):
                pool = self.engine.pool
                return DatabaseStats(
                    total_connections=pool.size() + pool.overflow(),
                    active_connections=pool.checkedout(),
                    idle_connections=pool.checkedin(),
                    pool_size=pool.size(),
                    max_overflow=pool._max_overflow,
                    checked_out=pool.checkedout(),
                    checked_in=pool.checkedin(),
                    invalidated=pool.invalidated()
                )
        except Exception as e:
            logger.warning(f"Failed to get pool stats: {e}")
        
        return None
    
    def get_table_info(self) -> List[Dict[str, Any]]:
        """Get information about database tables.
        
        Returns:
            List of table information
        """
        try:
            inspector = inspect(self.engine)
            tables = []
            
            for table_name in inspector.get_table_names():
                columns = inspector.get_columns(table_name)
                indexes = inspector.get_indexes(table_name)
                foreign_keys = inspector.get_foreign_keys(table_name)
                
                tables.append({
                    "name": table_name,
                    "columns": len(columns),
                    "indexes": len(indexes),
                    "foreign_keys": len(foreign_keys),
                    "column_details": [
                        {
                            "name": col["name"],
                            "type": str(col["type"]),
                            "nullable": col["nullable"],
                            "primary_key": col.get("primary_key", False)
                        }
                        for col in columns
                    ]
                })
            
            return tables
        
        except Exception as e:
            logger.error(f"Failed to get table info: {e}")
            return []
    
    def create_all_tables(self) -> None:
        """Create all tables defined in metadata.
        
        Raises:
            DatabaseException: If table creation fails
        """
        try:
            self.metadata.create_all(bind=self.engine)
            logger.info("All tables created successfully")
        except SQLAlchemyError as e:
            logger.error(f"Table creation failed: {e}")
            raise DatabaseException(f"Table creation failed: {e}")
    
    def drop_all_tables(self) -> None:
        """Drop all tables defined in metadata.
        
        Raises:
            DatabaseException: If table dropping fails
        """
        try:
            self.metadata.drop_all(bind=self.engine)
            logger.info("All tables dropped successfully")
        except SQLAlchemyError as e:
            logger.error(f"Table dropping failed: {e}")
            raise DatabaseException(f"Table dropping failed: {e}")
    
    def close_all_sessions(self) -> None:
        """Close all active sessions."""
        if self._scoped_session:
            self._scoped_session.remove()
            logger.info("All scoped sessions closed")
    
    def dispose_engine(self) -> None:
        """Dispose of the database engine."""
        if self._engine:
            self._engine.dispose()
            self._engine = None
            logger.info("Database engine disposed")
    
    def shutdown(self) -> None:
        """Shutdown database manager."""
        logger.info("Shutting down database manager")
        self.close_all_sessions()
        self.dispose_engine()
        logger.info("Database manager shutdown complete")


class DatabaseMigration:
    """Database migration utilities."""
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize migration manager.
        
        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager
        self.logger = get_logger("migration")
    
    def create_migration_table(self) -> None:
        """Create migration tracking table."""
        sql = """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version VARCHAR(255) PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        try:
            self.db_manager.execute_raw_sql(sql)
            self.logger.info("Migration table created")
        except Exception as e:
            self.logger.error(f"Failed to create migration table: {e}")
            raise
    
    def get_applied_migrations(self) -> List[str]:
        """Get list of applied migrations.
        
        Returns:
            List of applied migration versions
        """
        try:
            result = self.db_manager.execute_raw_sql(
                "SELECT version FROM schema_migrations ORDER BY version"
            )
            return [row[0] for row in result.fetchall()]
        except Exception:
            # Migration table doesn't exist yet
            return []
    
    def record_migration(self, version: str) -> None:
        """Record a migration as applied.
        
        Args:
            version: Migration version
        """
        try:
            self.db_manager.execute_raw_sql(
                "INSERT INTO schema_migrations (version) VALUES (:version)",
                {"version": version}
            )
            self.logger.info(f"Migration {version} recorded")
        except Exception as e:
            self.logger.error(f"Failed to record migration {version}: {e}")
            raise


# Global database manager instance
db_manager = DatabaseManager()


# Dependency function for FastAPI
def get_db() -> Generator[Session, None, None]:
    """Get database session for dependency injection.
    
    Yields:
        Database session
    """
    with db_manager.session_scope() as session:
        yield session


# Utility functions
def init_database() -> None:
    """Initialize database with tables."""
    logger.info("Initializing database")
    
    try:
        # Create all tables
        db_manager.create_all_tables()
        
        # Create migration table
        migration_manager = DatabaseMigration(db_manager)
        migration_manager.create_migration_table()
        
        logger.info("Database initialization complete")
    
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


def check_database_health() -> Dict[str, Any]:
    """Check database health status.
    
    Returns:
        Health status information
    """
    return db_manager.check_health()


def get_database_stats() -> Dict[str, Any]:
    """Get database statistics.
    
    Returns:
        Database statistics
    """
    stats = db_manager.get_pool_stats()
    table_info = db_manager.get_table_info()
    
    return {
        "pool_stats": stats.to_dict() if stats else None,
        "table_count": len(table_info),
        "tables": table_info
    }


def shutdown_database() -> None:
    """Shutdown database connections."""
    logger.info("Shutting down database")
    db_manager.shutdown()
    logger.info("Database shutdown complete")