"""Base model classes for FinScope application.

This module provides:
- Base model with common functionality
- Timestamp mixins for created/updated tracking
- Soft delete functionality
- UUID primary keys
- Common model utilities
"""

from typing import Any, Dict, Optional, Type, TypeVar, List
from sqlalchemy import (
    Column, String, DateTime, Boolean, Text, Integer,
    event, inspect, MetaData
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy.orm import Session, Query
from sqlalchemy.sql import func
from datetime import datetime
import uuid
import json
from pydantic import BaseModel as PydanticBaseModel

from app.core.database import db_manager

# Create declarative base
Base = declarative_base(metadata=db_manager.metadata)

# Type variable for model classes
ModelType = TypeVar("ModelType", bound="BaseModel")


class TimestampMixin:
    """Mixin for automatic timestamp tracking."""
    
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        doc="Record creation timestamp"
    )
    
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        doc="Record last update timestamp"
    )


class SoftDeleteMixin:
    """Mixin for soft delete functionality."""
    
    is_deleted = Column(
        Boolean,
        default=False,
        nullable=False,
        doc="Soft delete flag"
    )
    
    deleted_at = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Soft delete timestamp"
    )
    
    def soft_delete(self) -> None:
        """Mark record as deleted."""
        self.is_deleted = True
        self.deleted_at = datetime.utcnow()
    
    def restore(self) -> None:
        """Restore soft deleted record."""
        self.is_deleted = False
        self.deleted_at = None


class BaseModel(Base, TimestampMixin):
    """Base model class with common functionality."""
    
    __abstract__ = True
    
    # Primary key
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        doc="Primary key UUID"
    )
    
    # Metadata for storing additional information
    metadata_json = Column(
        Text,
        nullable=True,
        doc="JSON metadata storage"
    )
    
    @declared_attr
    def __tablename__(cls) -> str:
        """Generate table name from class name.
        
        Returns:
            Table name in snake_case
        """
        # Convert CamelCase to snake_case
        import re
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', cls.__name__)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
    
    @property
    def metadata_dict(self) -> Dict[str, Any]:
        """Get metadata as dictionary.
        
        Returns:
            Metadata dictionary
        """
        if self.metadata_json:
            try:
                return json.loads(self.metadata_json)
            except (json.JSONDecodeError, TypeError):
                return {}
        return {}
    
    @metadata_dict.setter
    def metadata_dict(self, value: Dict[str, Any]) -> None:
        """Set metadata from dictionary.
        
        Args:
            value: Metadata dictionary
        """
        if value:
            self.metadata_json = json.dumps(value, default=str)
        else:
            self.metadata_json = None
    
    def set_metadata(self, key: str, value: Any) -> None:
        """Set a metadata value.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        metadata = self.metadata_dict
        metadata[key] = value
        self.metadata_dict = metadata
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get a metadata value.
        
        Args:
            key: Metadata key
            default: Default value if key not found
            
        Returns:
            Metadata value
        """
        return self.metadata_dict.get(key, default)
    
    def to_dict(
        self,
        include_relationships: bool = False,
        exclude_fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Convert model to dictionary.
        
        Args:
            include_relationships: Include relationship data
            exclude_fields: Fields to exclude
            
        Returns:
            Dictionary representation
        """
        exclude_fields = exclude_fields or []
        result = {}
        
        # Get column attributes
        mapper = inspect(self.__class__)
        for column in mapper.columns:
            if column.name not in exclude_fields:
                value = getattr(self, column.name)
                
                # Handle special types
                if isinstance(value, datetime):
                    result[column.name] = value.isoformat()
                elif isinstance(value, uuid.UUID):
                    result[column.name] = str(value)
                else:
                    result[column.name] = value
        
        # Include relationships if requested
        if include_relationships:
            for relationship in mapper.relationships:
                if relationship.key not in exclude_fields:
                    related_obj = getattr(self, relationship.key)
                    if related_obj is not None:
                        if hasattr(related_obj, '__iter__') and not isinstance(related_obj, str):
                            # Collection relationship
                            result[relationship.key] = [
                                item.to_dict() if hasattr(item, 'to_dict') else str(item)
                                for item in related_obj
                            ]
                        else:
                            # Single relationship
                            result[relationship.key] = (
                                related_obj.to_dict() if hasattr(related_obj, 'to_dict')
                                else str(related_obj)
                            )
        
        return result
    
    def update_from_dict(
        self,
        data: Dict[str, Any],
        exclude_fields: Optional[List[str]] = None
    ) -> None:
        """Update model from dictionary.
        
        Args:
            data: Data dictionary
            exclude_fields: Fields to exclude from update
        """
        exclude_fields = exclude_fields or ['id', 'created_at']
        
        mapper = inspect(self.__class__)
        for key, value in data.items():
            if key not in exclude_fields and hasattr(self, key):
                # Check if it's a column attribute
                if key in [col.name for col in mapper.columns]:
                    setattr(self, key, value)
    
    @classmethod
    def create(
        cls: Type[ModelType],
        db: Session,
        **kwargs
    ) -> ModelType:
        """Create new instance.
        
        Args:
            db: Database session
            **kwargs: Model attributes
            
        Returns:
            Created instance
        """
        instance = cls(**kwargs)
        db.add(instance)
        db.flush()  # Get the ID without committing
        return instance
    
    @classmethod
    def get_by_id(
        cls: Type[ModelType],
        db: Session,
        id: uuid.UUID
    ) -> Optional[ModelType]:
        """Get instance by ID.
        
        Args:
            db: Database session
            id: Instance ID
            
        Returns:
            Instance or None
        """
        return db.query(cls).filter(cls.id == id).first()
    
    @classmethod
    def get_all(
        cls: Type[ModelType],
        db: Session,
        skip: int = 0,
        limit: int = 100
    ) -> List[ModelType]:
        """Get all instances with pagination.
        
        Args:
            db: Database session
            skip: Number of records to skip
            limit: Maximum number of records
            
        Returns:
            List of instances
        """
        return db.query(cls).offset(skip).limit(limit).all()
    
    @classmethod
    def count(
        cls: Type[ModelType],
        db: Session
    ) -> int:
        """Count total instances.
        
        Args:
            db: Database session
            
        Returns:
            Total count
        """
        return db.query(cls).count()
    
    def save(self, db: Session) -> None:
        """Save instance to database.
        
        Args:
            db: Database session
        """
        db.add(self)
        db.flush()
    
    def delete(self, db: Session) -> None:
        """Delete instance from database.
        
        Args:
            db: Database session
        """
        db.delete(self)
        db.flush()
    
    def refresh(self, db: Session) -> None:
        """Refresh instance from database.
        
        Args:
            db: Database session
        """
        db.refresh(self)
    
    def __repr__(self) -> str:
        """String representation of the model.
        
        Returns:
            String representation
        """
        return f"<{self.__class__.__name__}(id={self.id})>"


class SoftDeleteModel(BaseModel, SoftDeleteMixin):
    """Base model with soft delete functionality."""
    
    __abstract__ = True
    
    @classmethod
    def get_active(
        cls: Type[ModelType],
        db: Session,
        skip: int = 0,
        limit: int = 100
    ) -> List[ModelType]:
        """Get all active (non-deleted) instances.
        
        Args:
            db: Database session
            skip: Number of records to skip
            limit: Maximum number of records
            
        Returns:
            List of active instances
        """
        return (
            db.query(cls)
            .filter(cls.is_deleted == False)
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    @classmethod
    def get_deleted(
        cls: Type[ModelType],
        db: Session,
        skip: int = 0,
        limit: int = 100
    ) -> List[ModelType]:
        """Get all deleted instances.
        
        Args:
            db: Database session
            skip: Number of records to skip
            limit: Maximum number of records
            
        Returns:
            List of deleted instances
        """
        return (
            db.query(cls)
            .filter(cls.is_deleted == True)
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    @classmethod
    def count_active(
        cls: Type[ModelType],
        db: Session
    ) -> int:
        """Count active instances.
        
        Args:
            db: Database session
            
        Returns:
            Active count
        """
        return db.query(cls).filter(cls.is_deleted == False).count()


class AuditMixin:
    """Mixin for audit trail functionality."""
    
    created_by = Column(
        UUID(as_uuid=True),
        nullable=True,
        doc="User who created the record"
    )
    
    updated_by = Column(
        UUID(as_uuid=True),
        nullable=True,
        doc="User who last updated the record"
    )
    
    version = Column(
        Integer,
        default=1,
        nullable=False,
        doc="Record version for optimistic locking"
    )


class AuditModel(BaseModel, AuditMixin):
    """Base model with audit trail functionality."""
    
    __abstract__ = True
    
    def increment_version(self) -> None:
        """Increment version for optimistic locking."""
        self.version += 1


# Event listeners for automatic timestamp updates
@event.listens_for(BaseModel, 'before_update', propagate=True)
def receive_before_update(mapper, connection, target):
    """Update timestamp before update."""
    if hasattr(target, 'updated_at'):
        target.updated_at = datetime.utcnow()


@event.listens_for(SoftDeleteMixin, 'before_update', propagate=True)
def receive_before_soft_delete(mapper, connection, target):
    """Set deleted_at timestamp when soft deleting."""
    if hasattr(target, 'is_deleted') and target.is_deleted and not target.deleted_at:
        target.deleted_at = datetime.utcnow()


# Pydantic models for API serialization
class BaseSchema(PydanticBaseModel):
    """Base Pydantic schema for API serialization."""
    
    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            uuid.UUID: lambda v: str(v)
        }


class TimestampSchema(BaseSchema):
    """Schema with timestamp fields."""
    created_at: datetime
    updated_at: datetime


class SoftDeleteSchema(TimestampSchema):
    """Schema with soft delete fields."""
    is_deleted: bool
    deleted_at: Optional[datetime] = None


class BaseResponseSchema(BaseSchema):
    """Base response schema with ID and timestamps."""
    id: uuid.UUID
    created_at: datetime
    updated_at: datetime
    metadata_dict: Optional[Dict[str, Any]] = None


# Utility functions
def get_model_by_tablename(tablename: str) -> Optional[Type[BaseModel]]:
    """Get model class by table name.
    
    Args:
        tablename: Table name
        
    Returns:
        Model class or None
    """
    for cls in Base.registry._class_registry.values():
        if hasattr(cls, '__tablename__') and cls.__tablename__ == tablename:
            return cls
    return None


def get_all_models() -> List[Type[BaseModel]]:
    """Get all model classes.
    
    Returns:
        List of model classes
    """
    models = []
    for cls in Base.registry._class_registry.values():
        if (hasattr(cls, '__tablename__') and 
            issubclass(cls, BaseModel) and 
            cls is not BaseModel):
            models.append(cls)
    return models


def create_all_tables() -> None:
    """Create all database tables."""
    Base.metadata.create_all(bind=db_manager.engine)


def drop_all_tables() -> None:
    """Drop all database tables."""
    Base.metadata.drop_all(bind=db_manager.engine)