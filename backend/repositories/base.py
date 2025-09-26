from typing import Generic, TypeVar, Type, Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import and_, or_, desc, asc
from abc import ABC, abstractmethod

ModelType = TypeVar("ModelType")
CreateSchemaType = TypeVar("CreateSchemaType")
UpdateSchemaType = TypeVar("UpdateSchemaType")

class BaseRepository(Generic[ModelType, CreateSchemaType, UpdateSchemaType], ABC):
    """
    Base repository class providing common CRUD operations
    """
    
    def __init__(self, model: Type[ModelType]):
        """
        Initialize repository with model class
        
        Args:
            model: SQLAlchemy model class
        """
        self.model = model
    
    def get(self, db: Session, id: Any) -> Optional[ModelType]:
        """
        Get a single record by ID
        
        Args:
            db: Database session
            id: Record ID
            
        Returns:
            Model instance or None
        """
        try:
            return db.query(self.model).filter(self.model.id == id).first()
        except SQLAlchemyError as e:
            db.rollback()
            raise e
    
    def get_multi(
        self, 
        db: Session, 
        *, 
        skip: int = 0, 
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        order_desc: bool = False
    ) -> List[ModelType]:
        """
        Get multiple records with pagination and filtering
        
        Args:
            db: Database session
            skip: Number of records to skip
            limit: Maximum number of records to return
            filters: Dictionary of field filters
            order_by: Field name to order by
            order_desc: Whether to order in descending order
            
        Returns:
            List of model instances
        """
        try:
            query = db.query(self.model)
            
            # Apply filters
            if filters:
                for field, value in filters.items():
                    if hasattr(self.model, field):
                        if isinstance(value, list):
                            query = query.filter(getattr(self.model, field).in_(value))
                        else:
                            query = query.filter(getattr(self.model, field) == value)
            
            # Apply ordering
            if order_by and hasattr(self.model, order_by):
                order_field = getattr(self.model, order_by)
                if order_desc:
                    query = query.order_by(desc(order_field))
                else:
                    query = query.order_by(asc(order_field))
            
            return query.offset(skip).limit(limit).all()
        except SQLAlchemyError as e:
            db.rollback()
            raise e
    
    def create(self, db: Session, *, obj_in: CreateSchemaType) -> ModelType:
        """
        Create a new record
        
        Args:
            db: Database session
            obj_in: Create schema with data
            
        Returns:
            Created model instance
        """
        try:
            obj_in_data = obj_in.dict() if hasattr(obj_in, 'dict') else obj_in
            db_obj = self.model(**obj_in_data)
            db.add(db_obj)
            db.commit()
            db.refresh(db_obj)
            return db_obj
        except SQLAlchemyError as e:
            db.rollback()
            raise e
    
    def update(
        self, 
        db: Session, 
        *, 
        db_obj: ModelType, 
        obj_in: UpdateSchemaType
    ) -> ModelType:
        """
        Update an existing record
        
        Args:
            db: Database session
            db_obj: Existing model instance
            obj_in: Update schema with new data
            
        Returns:
            Updated model instance
        """
        try:
            obj_data = obj_in.dict(exclude_unset=True) if hasattr(obj_in, 'dict') else obj_in
            
            for field, value in obj_data.items():
                if hasattr(db_obj, field):
                    setattr(db_obj, field, value)
            
            db.add(db_obj)
            db.commit()
            db.refresh(db_obj)
            return db_obj
        except SQLAlchemyError as e:
            db.rollback()
            raise e
    
    def delete(self, db: Session, *, id: Any) -> ModelType:
        """
        Delete a record by ID
        
        Args:
            db: Database session
            id: Record ID
            
        Returns:
            Deleted model instance
        """
        try:
            obj = db.query(self.model).get(id)
            if obj:
                db.delete(obj)
                db.commit()
            return obj
        except SQLAlchemyError as e:
            db.rollback()
            raise e
    
    def count(self, db: Session, *, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Count records with optional filtering
        
        Args:
            db: Database session
            filters: Dictionary of field filters
            
        Returns:
            Number of matching records
        """
        try:
            query = db.query(self.model)
            
            if filters:
                for field, value in filters.items():
                    if hasattr(self.model, field):
                        if isinstance(value, list):
                            query = query.filter(getattr(self.model, field).in_(value))
                        else:
                            query = query.filter(getattr(self.model, field) == value)
            
            return query.count()
        except SQLAlchemyError as e:
            db.rollback()
            raise e
    
    def exists(self, db: Session, *, id: Any) -> bool:
        """
        Check if a record exists by ID
        
        Args:
            db: Database session
            id: Record ID
            
        Returns:
            True if record exists, False otherwise
        """
        try:
            return db.query(self.model).filter(self.model.id == id).first() is not None
        except SQLAlchemyError as e:
            db.rollback()
            raise e