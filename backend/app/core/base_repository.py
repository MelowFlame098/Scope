"""Base repository pattern for FinScope application.

This module provides base repository classes for consistent database operations:
- CRUD operations
- Query building
- Transaction management
- Error handling
- Pagination
- Filtering
"""

from abc import ABC, abstractmethod
from typing import (
    Any, Dict, List, Optional, Type, TypeVar, Generic, Union,
    Sequence, Tuple
)
from sqlalchemy.orm import Session, Query
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from sqlalchemy import and_, or_, desc, asc, func, text
from sqlalchemy.sql import Select
import logging
from datetime import datetime

from app.core.exceptions import (
    DatabaseException,
    ValidationException,
    NotFoundException
)

# Type variables
ModelType = TypeVar('ModelType')
CreateSchemaType = TypeVar('CreateSchemaType')
UpdateSchemaType = TypeVar('UpdateSchemaType')


class FilterCriteria:
    """Filter criteria for database queries."""
    
    def __init__(self):
        """Initialize filter criteria."""
        self.filters: List[Any] = []
        self.or_filters: List[Any] = []
        self.order_by: List[Any] = []
        self.limit: Optional[int] = None
        self.offset: Optional[int] = None
    
    def add_filter(self, condition: Any) -> 'FilterCriteria':
        """Add AND filter condition.
        
        Args:
            condition: SQLAlchemy filter condition
            
        Returns:
            Self for method chaining
        """
        self.filters.append(condition)
        return self
    
    def add_or_filter(self, condition: Any) -> 'FilterCriteria':
        """Add OR filter condition.
        
        Args:
            condition: SQLAlchemy filter condition
            
        Returns:
            Self for method chaining
        """
        self.or_filters.append(condition)
        return self
    
    def add_order(self, column: Any, ascending: bool = True) -> 'FilterCriteria':
        """Add order by clause.
        
        Args:
            column: Column to order by
            ascending: Whether to order ascending
            
        Returns:
            Self for method chaining
        """
        if ascending:
            self.order_by.append(asc(column))
        else:
            self.order_by.append(desc(column))
        return self
    
    def set_pagination(self, limit: int, offset: int = 0) -> 'FilterCriteria':
        """Set pagination parameters.
        
        Args:
            limit: Maximum number of results
            offset: Number of results to skip
            
        Returns:
            Self for method chaining
        """
        self.limit = limit
        self.offset = offset
        return self


class PaginationResult(Generic[ModelType]):
    """Pagination result container."""
    
    def __init__(
        self,
        items: List[ModelType],
        total: int,
        page: int,
        per_page: int,
        pages: int
    ):
        """Initialize pagination result.
        
        Args:
            items: List of items for current page
            total: Total number of items
            page: Current page number (1-based)
            per_page: Items per page
            pages: Total number of pages
        """
        self.items = items
        self.total = total
        self.page = page
        self.per_page = per_page
        self.pages = pages
        self.has_prev = page > 1
        self.has_next = page < pages
        self.prev_num = page - 1 if self.has_prev else None
        self.next_num = page + 1 if self.has_next else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "items": self.items,
            "total": self.total,
            "page": self.page,
            "per_page": self.per_page,
            "pages": self.pages,
            "has_prev": self.has_prev,
            "has_next": self.has_next,
            "prev_num": self.prev_num,
            "next_num": self.next_num
        }


class BaseRepository(Generic[ModelType], ABC):
    """Base repository class for database operations."""
    
    def __init__(self, model: Type[ModelType]):
        """Initialize repository.
        
        Args:
            model: SQLAlchemy model class
        """
        self.model = model
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _handle_db_error(self, error: Exception, operation: str) -> None:
        """Handle database errors consistently.
        
        Args:
            error: Database exception
            operation: Operation that failed
            
        Raises:
            DatabaseException: Wrapped database exception
        """
        error_msg = f"Database {operation} failed: {str(error)}"
        self.logger.error(error_msg, exc_info=True)
        
        if isinstance(error, IntegrityError):
            raise DatabaseException(
                f"Data integrity violation during {operation}",
                original_error=error
            )
        else:
            raise DatabaseException(error_msg, original_error=error)
    
    def _apply_filters(self, query: Query, criteria: FilterCriteria) -> Query:
        """Apply filter criteria to query.
        
        Args:
            query: SQLAlchemy query
            criteria: Filter criteria
            
        Returns:
            Filtered query
        """
        # Apply AND filters
        if criteria.filters:
            query = query.filter(and_(*criteria.filters))
        
        # Apply OR filters
        if criteria.or_filters:
            query = query.filter(or_(*criteria.or_filters))
        
        # Apply ordering
        if criteria.order_by:
            query = query.order_by(*criteria.order_by)
        
        # Apply pagination
        if criteria.offset is not None:
            query = query.offset(criteria.offset)
        
        if criteria.limit is not None:
            query = query.limit(criteria.limit)
        
        return query
    
    # CRUD Operations
    def create(self, db: Session, obj_in: Union[CreateSchemaType, Dict[str, Any]]) -> ModelType:
        """Create a new record.
        
        Args:
            db: Database session
            obj_in: Data for creating the record
            
        Returns:
            Created record
            
        Raises:
            DatabaseException: If creation fails
        """
        try:
            if isinstance(obj_in, dict):
                db_obj = self.model(**obj_in)
            else:
                obj_data = obj_in.dict() if hasattr(obj_in, 'dict') else obj_in
                db_obj = self.model(**obj_data)
            
            db.add(db_obj)
            db.flush()  # Get ID without committing
            db.refresh(db_obj)
            return db_obj
        except SQLAlchemyError as e:
            self._handle_db_error(e, "create")
    
    def get(self, db: Session, id: Any) -> Optional[ModelType]:
        """Get record by ID.
        
        Args:
            db: Database session
            id: Record ID
            
        Returns:
            Record or None if not found
        """
        try:
            return db.query(self.model).filter(self.model.id == id).first()
        except SQLAlchemyError as e:
            self._handle_db_error(e, "get")
    
    def get_or_404(self, db: Session, id: Any) -> ModelType:
        """Get record by ID or raise 404.
        
        Args:
            db: Database session
            id: Record ID
            
        Returns:
            Record
            
        Raises:
            NotFoundException: If record not found
        """
        obj = self.get(db, id)
        if obj is None:
            raise NotFoundException(f"{self.model.__name__} with id {id} not found")
        return obj
    
    def get_multi(
        self,
        db: Session,
        skip: int = 0,
        limit: int = 100,
        criteria: Optional[FilterCriteria] = None
    ) -> List[ModelType]:
        """Get multiple records.
        
        Args:
            db: Database session
            skip: Number of records to skip
            limit: Maximum number of records
            criteria: Filter criteria
            
        Returns:
            List of records
        """
        try:
            query = db.query(self.model)
            
            if criteria:
                query = self._apply_filters(query, criteria)
            else:
                query = query.offset(skip).limit(limit)
            
            return query.all()
        except SQLAlchemyError as e:
            self._handle_db_error(e, "get_multi")
    
    def get_paginated(
        self,
        db: Session,
        page: int = 1,
        per_page: int = 20,
        criteria: Optional[FilterCriteria] = None
    ) -> PaginationResult[ModelType]:
        """Get paginated records.
        
        Args:
            db: Database session
            page: Page number (1-based)
            per_page: Items per page
            criteria: Filter criteria
            
        Returns:
            Pagination result
        """
        try:
            query = db.query(self.model)
            
            if criteria:
                # Apply filters but not pagination
                temp_criteria = FilterCriteria()
                temp_criteria.filters = criteria.filters
                temp_criteria.or_filters = criteria.or_filters
                temp_criteria.order_by = criteria.order_by
                query = self._apply_filters(query, temp_criteria)
            
            # Get total count
            total = query.count()
            
            # Calculate pagination
            pages = (total + per_page - 1) // per_page
            offset = (page - 1) * per_page
            
            # Get items for current page
            items = query.offset(offset).limit(per_page).all()
            
            return PaginationResult(
                items=items,
                total=total,
                page=page,
                per_page=per_page,
                pages=pages
            )
        except SQLAlchemyError as e:
            self._handle_db_error(e, "get_paginated")
    
    def update(
        self,
        db: Session,
        db_obj: ModelType,
        obj_in: Union[UpdateSchemaType, Dict[str, Any]]
    ) -> ModelType:
        """Update a record.
        
        Args:
            db: Database session
            db_obj: Existing record
            obj_in: Update data
            
        Returns:
            Updated record
            
        Raises:
            DatabaseException: If update fails
        """
        try:
            if isinstance(obj_in, dict):
                update_data = obj_in
            else:
                update_data = obj_in.dict(exclude_unset=True) if hasattr(obj_in, 'dict') else obj_in
            
            for field, value in update_data.items():
                if hasattr(db_obj, field):
                    setattr(db_obj, field, value)
            
            # Update timestamp if available
            if hasattr(db_obj, 'updated_at'):
                db_obj.updated_at = datetime.utcnow()
            
            db.flush()
            db.refresh(db_obj)
            return db_obj
        except SQLAlchemyError as e:
            self._handle_db_error(e, "update")
    
    def delete(self, db: Session, id: Any) -> bool:
        """Delete a record by ID.
        
        Args:
            db: Database session
            id: Record ID
            
        Returns:
            True if deleted, False if not found
            
        Raises:
            DatabaseException: If deletion fails
        """
        try:
            obj = self.get(db, id)
            if obj:
                db.delete(obj)
                db.flush()
                return True
            return False
        except SQLAlchemyError as e:
            self._handle_db_error(e, "delete")
    
    def delete_obj(self, db: Session, db_obj: ModelType) -> None:
        """Delete a record object.
        
        Args:
            db: Database session
            db_obj: Record to delete
            
        Raises:
            DatabaseException: If deletion fails
        """
        try:
            db.delete(db_obj)
            db.flush()
        except SQLAlchemyError as e:
            self._handle_db_error(e, "delete_obj")
    
    # Query helpers
    def exists(self, db: Session, id: Any) -> bool:
        """Check if record exists.
        
        Args:
            db: Database session
            id: Record ID
            
        Returns:
            True if record exists
        """
        try:
            return db.query(
                db.query(self.model).filter(self.model.id == id).exists()
            ).scalar()
        except SQLAlchemyError as e:
            self._handle_db_error(e, "exists")
    
    def count(self, db: Session, criteria: Optional[FilterCriteria] = None) -> int:
        """Count records.
        
        Args:
            db: Database session
            criteria: Filter criteria
            
        Returns:
            Number of records
        """
        try:
            query = db.query(func.count(self.model.id))
            
            if criteria:
                # Apply only filters, not pagination or ordering
                if criteria.filters:
                    query = query.filter(and_(*criteria.filters))
                if criteria.or_filters:
                    query = query.filter(or_(*criteria.or_filters))
            
            return query.scalar()
        except SQLAlchemyError as e:
            self._handle_db_error(e, "count")
    
    def find_by(self, db: Session, **kwargs) -> Optional[ModelType]:
        """Find record by field values.
        
        Args:
            db: Database session
            **kwargs: Field values to search by
            
        Returns:
            First matching record or None
        """
        try:
            query = db.query(self.model)
            for key, value in kwargs.items():
                if hasattr(self.model, key):
                    query = query.filter(getattr(self.model, key) == value)
            return query.first()
        except SQLAlchemyError as e:
            self._handle_db_error(e, "find_by")
    
    def find_all_by(self, db: Session, **kwargs) -> List[ModelType]:
        """Find all records by field values.
        
        Args:
            db: Database session
            **kwargs: Field values to search by
            
        Returns:
            List of matching records
        """
        try:
            query = db.query(self.model)
            for key, value in kwargs.items():
                if hasattr(self.model, key):
                    query = query.filter(getattr(self.model, key) == value)
            return query.all()
        except SQLAlchemyError as e:
            self._handle_db_error(e, "find_all_by")
    
    # Bulk operations
    def bulk_create(self, db: Session, objs_in: List[Union[CreateSchemaType, Dict[str, Any]]]) -> List[ModelType]:
        """Create multiple records.
        
        Args:
            db: Database session
            objs_in: List of data for creating records
            
        Returns:
            List of created records
            
        Raises:
            DatabaseException: If creation fails
        """
        try:
            db_objs = []
            for obj_in in objs_in:
                if isinstance(obj_in, dict):
                    db_obj = self.model(**obj_in)
                else:
                    obj_data = obj_in.dict() if hasattr(obj_in, 'dict') else obj_in
                    db_obj = self.model(**obj_data)
                db_objs.append(db_obj)
            
            db.add_all(db_objs)
            db.flush()
            
            for db_obj in db_objs:
                db.refresh(db_obj)
            
            return db_objs
        except SQLAlchemyError as e:
            self._handle_db_error(e, "bulk_create")
    
    def bulk_update(self, db: Session, updates: List[Dict[str, Any]]) -> int:
        """Bulk update records.
        
        Args:
            db: Database session
            updates: List of update dictionaries with 'id' and update fields
            
        Returns:
            Number of updated records
            
        Raises:
            DatabaseException: If update fails
        """
        try:
            updated_count = 0
            for update_data in updates:
                if 'id' not in update_data:
                    continue
                
                record_id = update_data.pop('id')
                result = db.query(self.model).filter(
                    self.model.id == record_id
                ).update(update_data)
                updated_count += result
            
            db.flush()
            return updated_count
        except SQLAlchemyError as e:
            self._handle_db_error(e, "bulk_update")
    
    def bulk_delete(self, db: Session, ids: List[Any]) -> int:
        """Bulk delete records by IDs.
        
        Args:
            db: Database session
            ids: List of record IDs
            
        Returns:
            Number of deleted records
            
        Raises:
            DatabaseException: If deletion fails
        """
        try:
            result = db.query(self.model).filter(
                self.model.id.in_(ids)
            ).delete(synchronize_session=False)
            db.flush()
            return result
        except SQLAlchemyError as e:
            self._handle_db_error(e, "bulk_delete")


class ReadOnlyRepository(BaseRepository[ModelType]):
    """Read-only repository that only allows query operations."""
    
    def create(self, *args, **kwargs):
        """Disabled create operation."""
        raise NotImplementedError("Create operation not allowed in read-only repository")
    
    def update(self, *args, **kwargs):
        """Disabled update operation."""
        raise NotImplementedError("Update operation not allowed in read-only repository")
    
    def delete(self, *args, **kwargs):
        """Disabled delete operation."""
        raise NotImplementedError("Delete operation not allowed in read-only repository")
    
    def delete_obj(self, *args, **kwargs):
        """Disabled delete_obj operation."""
        raise NotImplementedError("Delete operation not allowed in read-only repository")
    
    def bulk_create(self, *args, **kwargs):
        """Disabled bulk_create operation."""
        raise NotImplementedError("Bulk create operation not allowed in read-only repository")
    
    def bulk_update(self, *args, **kwargs):
        """Disabled bulk_update operation."""
        raise NotImplementedError("Bulk update operation not allowed in read-only repository")
    
    def bulk_delete(self, *args, **kwargs):
        """Disabled bulk_delete operation."""
        raise NotImplementedError("Bulk delete operation not allowed in read-only repository")