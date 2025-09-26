# Preference Manager
# Phase 9: AI-First Platform Implementation

import logging
import asyncio
import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Set, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import sqlite3
import aiosqlite
from pathlib import Path

logger = logging.getLogger(__name__)

class PreferenceCategory(Enum):
    INTERFACE = "interface"                # UI/UX preferences
    TRADING = "trading"                    # Trading-related preferences
    ANALYTICS = "analytics"                # Analytics and reporting preferences
    NOTIFICATIONS = "notifications"        # Notification preferences
    SECURITY = "security"                  # Security and privacy preferences
    ACCESSIBILITY = "accessibility"        # Accessibility preferences
    PERFORMANCE = "performance"            # Performance optimization preferences
    PERSONALIZATION = "personalization"    # AI personalization preferences

class PreferenceType(Enum):
    BOOLEAN = "boolean"        # True/False values
    INTEGER = "integer"        # Numeric integer values
    FLOAT = "float"            # Numeric float values
    STRING = "string"          # Text values
    LIST = "list"              # List of values
    DICT = "dict"              # Dictionary/object values
    ENUM = "enum"              # Enumerated values
    JSON = "json"              # Complex JSON objects

class PreferenceScope(Enum):
    USER = "user"              # User-specific preferences
    ORGANIZATION = "organization"  # Organization-wide preferences
    GLOBAL = "global"          # System-wide defaults
    SESSION = "session"        # Session-specific preferences
    DEVICE = "device"          # Device-specific preferences

class PreferencePriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class SyncStatus(Enum):
    SYNCED = "synced"          # Preference is synchronized
    PENDING = "pending"        # Waiting to be synchronized
    CONFLICT = "conflict"      # Synchronization conflict
    ERROR = "error"            # Synchronization error
    LOCAL_ONLY = "local_only"  # Local-only preference

@dataclass
class PreferenceDefinition:
    key: str
    category: PreferenceCategory
    preference_type: PreferenceType
    default_value: Any
    description: str
    scope: PreferenceScope = PreferenceScope.USER
    priority: PreferencePriority = PreferencePriority.MEDIUM
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    allowed_values: Optional[List[Any]] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    requires_restart: bool = False
    sensitive: bool = False  # Contains sensitive information
    user_configurable: bool = True
    admin_only: bool = False
    deprecated: bool = False
    migration_path: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class PreferenceValue:
    key: str
    value: Any
    scope: PreferenceScope
    user_id: Optional[str] = None
    organization_id: Optional[str] = None
    device_id: Optional[str] = None
    session_id: Optional[str] = None
    priority: PreferencePriority = PreferencePriority.MEDIUM
    sync_status: SyncStatus = SyncStatus.SYNCED
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: int = 1
    checksum: Optional[str] = None

@dataclass
class PreferenceChange:
    change_id: str
    key: str
    old_value: Any
    new_value: Any
    scope: PreferenceScope
    user_id: Optional[str] = None
    change_reason: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PreferenceConflict:
    conflict_id: str
    key: str
    local_value: Any
    remote_value: Any
    scope: PreferenceScope
    user_id: Optional[str] = None
    conflict_type: str = "value_mismatch"
    detected_at: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolution_strategy: Optional[str] = None
    resolved_at: Optional[datetime] = None

@dataclass
class PreferenceBackup:
    backup_id: str
    user_id: str
    preferences: Dict[str, Any]
    backup_type: str = "manual"  # manual, automatic, migration
    created_at: datetime = field(default_factory=datetime.now)
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

class PreferenceManager:
    """Advanced preference management system with synchronization and conflict resolution"""
    
    def __init__(self, db_path: str = "preferences.db"):
        self.db_path = db_path
        
        # Preference definitions registry
        self.preference_definitions = {}  # key -> PreferenceDefinition
        
        # In-memory cache for fast access
        self.preference_cache = {}  # (scope, user_id, key) -> PreferenceValue
        self.cache_timestamps = {}  # (scope, user_id, key) -> timestamp
        
        # Change tracking
        self.change_history = deque(maxlen=10000)  # Recent changes
        self.change_listeners = defaultdict(list)  # key -> [callback]
        
        # Synchronization
        self.sync_queue = deque()  # Pending sync operations
        self.conflict_queue = deque()  # Unresolved conflicts
        self.sync_callbacks = []  # Sync event callbacks
        
        # Validation
        self.validators = {}  # key -> validation_function
        self.transformers = {}  # key -> transformation_function
        
        # Performance tracking
        self.access_stats = defaultdict(int)  # key -> access_count
        self.performance_metrics = defaultdict(list)  # operation -> [duration]
        
        # Backup management
        self.backup_schedule = {}  # user_id -> next_backup_time
        self.backup_retention = timedelta(days=30)
        
        logger.info("Preference manager initialized")
    
    async def initialize(self):
        """Initialize the preference manager and database"""
        try:
            await self._create_database_schema()
            await self._load_preference_definitions()
            await self._initialize_default_preferences()
            
            logger.info("Preference manager initialization completed")
            
        except Exception as e:
            logger.error(f"Error initializing preference manager: {e}")
            raise
    
    async def register_preference(self, definition: PreferenceDefinition) -> bool:
        """Register a new preference definition"""
        try:
            # Validate definition
            if not await self._validate_preference_definition(definition):
                return False
            
            # Store definition
            self.preference_definitions[definition.key] = definition
            
            # Save to database
            await self._save_preference_definition(definition)
            
            # Register validator if provided
            if definition.validation_rules:
                self.validators[definition.key] = self._create_validator(definition)
            
            logger.info(f"Registered preference: {definition.key}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering preference {definition.key}: {e}")
            return False
    
    async def set_preference(self, key: str, value: Any, scope: PreferenceScope = PreferenceScope.USER,
                           user_id: Optional[str] = None, organization_id: Optional[str] = None,
                           device_id: Optional[str] = None, session_id: Optional[str] = None,
                           priority: PreferencePriority = PreferencePriority.MEDIUM) -> bool:
        """Set a preference value"""
        try:
            start_time = datetime.now()
            
            # Validate preference exists
            if key not in self.preference_definitions:
                logger.warning(f"Preference {key} not registered")
                return False
            
            definition = self.preference_definitions[key]
            
            # Check permissions
            if not await self._check_preference_permissions(key, user_id, 'write'):
                logger.warning(f"User {user_id} does not have permission to set {key}")
                return False
            
            # Validate value
            if not await self._validate_preference_value(key, value):
                logger.warning(f"Invalid value for preference {key}: {value}")
                return False
            
            # Transform value if needed
            transformed_value = await self._transform_preference_value(key, value)
            
            # Get current value for change tracking
            current_value = await self.get_preference(key, scope, user_id, organization_id, device_id, session_id)
            
            # Create preference value
            pref_value = PreferenceValue(
                key=key,
                value=transformed_value,
                scope=scope,
                user_id=user_id,
                organization_id=organization_id,
                device_id=device_id,
                session_id=session_id,
                priority=priority,
                sync_status=SyncStatus.PENDING if scope != PreferenceScope.SESSION else SyncStatus.LOCAL_ONLY,
                checksum=self._calculate_checksum(transformed_value)
            )
            
            # Store in cache
            cache_key = self._get_cache_key(scope, user_id, key, organization_id, device_id, session_id)
            self.preference_cache[cache_key] = pref_value
            self.cache_timestamps[cache_key] = datetime.now()
            
            # Save to database
            await self._save_preference_value(pref_value)
            
            # Track change
            change = PreferenceChange(
                change_id=f"change_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                key=key,
                old_value=current_value,
                new_value=transformed_value,
                scope=scope,
                user_id=user_id,
                change_reason="user_update"
            )
            self.change_history.append(change)
            
            # Notify listeners
            await self._notify_preference_change(change)
            
            # Add to sync queue if needed
            if pref_value.sync_status == SyncStatus.PENDING:
                self.sync_queue.append(pref_value)
            
            # Update performance metrics
            duration = (datetime.now() - start_time).total_seconds()
            self.performance_metrics['set_preference'].append(duration)
            
            logger.info(f"Set preference {key} = {transformed_value} for scope {scope.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting preference {key}: {e}")
            return False
    
    async def get_preference(self, key: str, scope: PreferenceScope = PreferenceScope.USER,
                           user_id: Optional[str] = None, organization_id: Optional[str] = None,
                           device_id: Optional[str] = None, session_id: Optional[str] = None,
                           default: Any = None) -> Any:
        """Get a preference value with scope hierarchy"""
        try:
            start_time = datetime.now()
            
            # Check permissions
            if not await self._check_preference_permissions(key, user_id, 'read'):
                logger.warning(f"User {user_id} does not have permission to read {key}")
                return default
            
            # Try cache first
            cache_key = self._get_cache_key(scope, user_id, key, organization_id, device_id, session_id)
            if cache_key in self.preference_cache:
                cached_value = self.preference_cache[cache_key]
                
                # Check if cache is still valid
                if await self._is_cache_valid(cache_key):
                    self.access_stats[key] += 1
                    return cached_value.value
            
            # Search with scope hierarchy
            value = await self._get_preference_with_hierarchy(key, scope, user_id, organization_id, device_id, session_id)
            
            # Update performance metrics
            duration = (datetime.now() - start_time).total_seconds()
            self.performance_metrics['get_preference'].append(duration)
            self.access_stats[key] += 1
            
            return value if value is not None else default
            
        except Exception as e:
            logger.error(f"Error getting preference {key}: {e}")
            return default
    
    async def get_all_preferences(self, scope: PreferenceScope = PreferenceScope.USER,
                                user_id: Optional[str] = None, organization_id: Optional[str] = None,
                                device_id: Optional[str] = None, session_id: Optional[str] = None,
                                category: Optional[PreferenceCategory] = None) -> Dict[str, Any]:
        """Get all preferences for a scope, optionally filtered by category"""
        try:
            preferences = {}
            
            # Get all registered preference keys
            keys_to_fetch = list(self.preference_definitions.keys())
            
            # Filter by category if specified
            if category:
                keys_to_fetch = [
                    key for key in keys_to_fetch 
                    if self.preference_definitions[key].category == category
                ]
            
            # Fetch each preference
            for key in keys_to_fetch:
                value = await self.get_preference(key, scope, user_id, organization_id, device_id, session_id)
                if value is not None:
                    preferences[key] = value
            
            return preferences
            
        except Exception as e:
            logger.error(f"Error getting all preferences: {e}")
            return {}
    
    async def delete_preference(self, key: str, scope: PreferenceScope = PreferenceScope.USER,
                              user_id: Optional[str] = None, organization_id: Optional[str] = None,
                              device_id: Optional[str] = None, session_id: Optional[str] = None) -> bool:
        """Delete a preference value"""
        try:
            # Check permissions
            if not await self._check_preference_permissions(key, user_id, 'delete'):
                logger.warning(f"User {user_id} does not have permission to delete {key}")
                return False
            
            # Get current value for change tracking
            current_value = await self.get_preference(key, scope, user_id, organization_id, device_id, session_id)
            
            # Remove from cache
            cache_key = self._get_cache_key(scope, user_id, key, organization_id, device_id, session_id)
            if cache_key in self.preference_cache:
                del self.preference_cache[cache_key]
                del self.cache_timestamps[cache_key]
            
            # Delete from database
            await self._delete_preference_value(key, scope, user_id, organization_id, device_id, session_id)
            
            # Track change
            change = PreferenceChange(
                change_id=f"delete_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                key=key,
                old_value=current_value,
                new_value=None,
                scope=scope,
                user_id=user_id,
                change_reason="user_delete"
            )
            self.change_history.append(change)
            
            # Notify listeners
            await self._notify_preference_change(change)
            
            logger.info(f"Deleted preference {key} for scope {scope.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting preference {key}: {e}")
            return False
    
    async def add_change_listener(self, key: str, callback: Callable[[PreferenceChange], None]):
        """Add a listener for preference changes"""
        try:
            self.change_listeners[key].append(callback)
            logger.info(f"Added change listener for preference {key}")
            
        except Exception as e:
            logger.error(f"Error adding change listener for {key}: {e}")
    
    async def remove_change_listener(self, key: str, callback: Callable[[PreferenceChange], None]):
        """Remove a preference change listener"""
        try:
            if key in self.change_listeners and callback in self.change_listeners[key]:
                self.change_listeners[key].remove(callback)
                logger.info(f"Removed change listener for preference {key}")
            
        except Exception as e:
            logger.error(f"Error removing change listener for {key}: {e}")
    
    async def create_backup(self, user_id: str, backup_type: str = "manual", 
                          description: str = "") -> str:
        """Create a backup of user preferences"""
        try:
            backup_id = f"backup_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Get all user preferences
            preferences = await self.get_all_preferences(PreferenceScope.USER, user_id)
            
            # Create backup
            backup = PreferenceBackup(
                backup_id=backup_id,
                user_id=user_id,
                preferences=preferences,
                backup_type=backup_type,
                description=description
            )
            
            # Save backup
            await self._save_preference_backup(backup)
            
            logger.info(f"Created preference backup {backup_id} for user {user_id}")
            return backup_id
            
        except Exception as e:
            logger.error(f"Error creating backup for user {user_id}: {e}")
            return ""
    
    async def restore_backup(self, backup_id: str, user_id: str) -> bool:
        """Restore preferences from backup"""
        try:
            # Load backup
            backup = await self._load_preference_backup(backup_id)
            if not backup or backup.user_id != user_id:
                logger.warning(f"Backup {backup_id} not found or access denied")
                return False
            
            # Create current backup before restore
            await self.create_backup(user_id, "pre_restore", f"Backup before restoring {backup_id}")
            
            # Restore preferences
            for key, value in backup.preferences.items():
                await self.set_preference(key, value, PreferenceScope.USER, user_id)
            
            logger.info(f"Restored preferences from backup {backup_id} for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring backup {backup_id}: {e}")
            return False
    
    async def sync_preferences(self, user_id: str) -> Dict[str, Any]:
        """Synchronize preferences with remote storage"""
        try:
            sync_result = {
                'synced_count': 0,
                'conflict_count': 0,
                'error_count': 0,
                'conflicts': []
            }
            
            # Process sync queue
            pending_syncs = [item for item in self.sync_queue if item.user_id == user_id]
            
            for pref_value in pending_syncs:
                try:
                    # Simulate remote sync (in real implementation, this would call external API)
                    success = await self._sync_preference_to_remote(pref_value)
                    
                    if success:
                        pref_value.sync_status = SyncStatus.SYNCED
                        sync_result['synced_count'] += 1
                    else:
                        pref_value.sync_status = SyncStatus.ERROR
                        sync_result['error_count'] += 1
                    
                    # Update in database
                    await self._update_preference_sync_status(pref_value)
                    
                except Exception as e:
                    logger.error(f"Error syncing preference {pref_value.key}: {e}")
                    sync_result['error_count'] += 1
            
            # Check for conflicts
            conflicts = await self._detect_sync_conflicts(user_id)
            sync_result['conflict_count'] = len(conflicts)
            sync_result['conflicts'] = [conflict.conflict_id for conflict in conflicts]
            
            # Notify sync completion
            for callback in self.sync_callbacks:
                try:
                    await callback(user_id, sync_result)
                except Exception as e:
                    logger.error(f"Error in sync callback: {e}")
            
            return sync_result
            
        except Exception as e:
            logger.error(f"Error syncing preferences for user {user_id}: {e}")
            return {'error': 'Sync failed'}
    
    async def resolve_conflict(self, conflict_id: str, resolution_strategy: str) -> bool:
        """Resolve a preference synchronization conflict"""
        try:
            # Find conflict
            conflict = None
            for c in self.conflict_queue:
                if c.conflict_id == conflict_id:
                    conflict = c
                    break
            
            if not conflict:
                logger.warning(f"Conflict {conflict_id} not found")
                return False
            
            # Apply resolution strategy
            resolved_value = None
            
            if resolution_strategy == "use_local":
                resolved_value = conflict.local_value
            elif resolution_strategy == "use_remote":
                resolved_value = conflict.remote_value
            elif resolution_strategy == "merge":
                resolved_value = await self._merge_preference_values(conflict.local_value, conflict.remote_value)
            elif resolution_strategy == "use_latest":
                # Use the value with the latest timestamp (simplified)
                resolved_value = conflict.remote_value  # Assume remote is latest
            
            if resolved_value is not None:
                # Set resolved value
                await self.set_preference(
                    conflict.key, resolved_value, conflict.scope, conflict.user_id
                )
                
                # Mark conflict as resolved
                conflict.resolved = True
                conflict.resolution_strategy = resolution_strategy
                conflict.resolved_at = datetime.now()
                
                # Remove from conflict queue
                self.conflict_queue.remove(conflict)
                
                logger.info(f"Resolved conflict {conflict_id} using strategy {resolution_strategy}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error resolving conflict {conflict_id}: {e}")
            return False
    
    async def get_preference_statistics(self) -> Dict[str, Any]:
        """Get comprehensive preference usage statistics"""
        try:
            stats = {
                'total_preferences': len(self.preference_definitions),
                'cache_size': len(self.preference_cache),
                'pending_syncs': len(self.sync_queue),
                'unresolved_conflicts': len(self.conflict_queue),
                'recent_changes': len(self.change_history),
                'access_statistics': dict(self.access_stats),
                'performance_metrics': {},
                'category_breakdown': defaultdict(int),
                'scope_breakdown': defaultdict(int)
            }
            
            # Calculate performance metrics
            for operation, durations in self.performance_metrics.items():
                if durations:
                    stats['performance_metrics'][operation] = {
                        'avg_duration': sum(durations) / len(durations),
                        'min_duration': min(durations),
                        'max_duration': max(durations),
                        'total_calls': len(durations)
                    }
            
            # Category breakdown
            for definition in self.preference_definitions.values():
                stats['category_breakdown'][definition.category.value] += 1
            
            # Scope breakdown from cache
            for cache_key in self.preference_cache:
                scope = self.preference_cache[cache_key].scope
                stats['scope_breakdown'][scope.value] += 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting preference statistics: {e}")
            return {'error': 'Failed to generate statistics'}
    
    async def cleanup_expired_preferences(self):
        """Clean up expired preferences and old backups"""
        try:
            current_time = datetime.now()
            cleaned_count = 0
            
            # Clean expired preferences from cache
            expired_keys = []
            for cache_key, pref_value in self.preference_cache.items():
                if pref_value.expires_at and pref_value.expires_at <= current_time:
                    expired_keys.append(cache_key)
            
            for key in expired_keys:
                del self.preference_cache[key]
                if key in self.cache_timestamps:
                    del self.cache_timestamps[key]
                cleaned_count += 1
            
            # Clean old backups
            cutoff_date = current_time - self.backup_retention
            await self._cleanup_old_backups(cutoff_date)
            
            # Clean old change history
            if len(self.change_history) > 5000:
                # Keep only the most recent 5000 changes
                while len(self.change_history) > 5000:
                    self.change_history.popleft()
            
            logger.info(f"Cleaned up {cleaned_count} expired preferences")
            
        except Exception as e:
            logger.error(f"Error cleaning up expired preferences: {e}")
    
    # Helper methods for database operations
    async def _create_database_schema(self):
        """Create database schema for preferences"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Preference definitions table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS preference_definitions (
                        key TEXT PRIMARY KEY,
                        category TEXT NOT NULL,
                        preference_type TEXT NOT NULL,
                        default_value TEXT,
                        description TEXT,
                        scope TEXT,
                        priority INTEGER,
                        validation_rules TEXT,
                        allowed_values TEXT,
                        min_value REAL,
                        max_value REAL,
                        requires_restart BOOLEAN,
                        sensitive BOOLEAN,
                        user_configurable BOOLEAN,
                        admin_only BOOLEAN,
                        deprecated BOOLEAN,
                        migration_path TEXT,
                        created_at TEXT,
                        updated_at TEXT
                    )
                """)
                
                # Preference values table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS preference_values (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        key TEXT NOT NULL,
                        value TEXT,
                        scope TEXT NOT NULL,
                        user_id TEXT,
                        organization_id TEXT,
                        device_id TEXT,
                        session_id TEXT,
                        priority INTEGER,
                        sync_status TEXT,
                        created_at TEXT,
                        updated_at TEXT,
                        expires_at TEXT,
                        metadata TEXT,
                        version INTEGER,
                        checksum TEXT,
                        UNIQUE(key, scope, user_id, organization_id, device_id, session_id)
                    )
                """)
                
                # Preference backups table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS preference_backups (
                        backup_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        preferences TEXT NOT NULL,
                        backup_type TEXT,
                        created_at TEXT,
                        description TEXT,
                        metadata TEXT
                    )
                """)
                
                # Create indexes
                await db.execute("CREATE INDEX IF NOT EXISTS idx_pref_values_user ON preference_values(user_id)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_pref_values_key ON preference_values(key)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_pref_backups_user ON preference_backups(user_id)")
                
                await db.commit()
            
        except Exception as e:
            logger.error(f"Error creating database schema: {e}")
            raise
    
    async def _load_preference_definitions(self):
        """Load preference definitions from database"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute("SELECT * FROM preference_definitions") as cursor:
                    async for row in cursor:
                        definition = PreferenceDefinition(
                            key=row[0],
                            category=PreferenceCategory(row[1]),
                            preference_type=PreferenceType(row[2]),
                            default_value=json.loads(row[3]) if row[3] else None,
                            description=row[4],
                            scope=PreferenceScope(row[5]) if row[5] else PreferenceScope.USER,
                            priority=PreferencePriority(row[6]) if row[6] else PreferencePriority.MEDIUM,
                            validation_rules=json.loads(row[7]) if row[7] else {},
                            allowed_values=json.loads(row[8]) if row[8] else None,
                            min_value=row[9],
                            max_value=row[10],
                            requires_restart=bool(row[11]),
                            sensitive=bool(row[12]),
                            user_configurable=bool(row[13]),
                            admin_only=bool(row[14]),
                            deprecated=bool(row[15]),
                            migration_path=row[16],
                            created_at=datetime.fromisoformat(row[17]) if row[17] else datetime.now(),
                            updated_at=datetime.fromisoformat(row[18]) if row[18] else datetime.now()
                        )
                        self.preference_definitions[definition.key] = definition
            
        except Exception as e:
            logger.error(f"Error loading preference definitions: {e}")
    
    async def _initialize_default_preferences(self):
        """Initialize default preference definitions"""
        try:
            default_preferences = [
                PreferenceDefinition(
                    key="ui.theme",
                    category=PreferenceCategory.INTERFACE,
                    preference_type=PreferenceType.ENUM,
                    default_value="light",
                    description="User interface theme",
                    allowed_values=["light", "dark", "auto"]
                ),
                PreferenceDefinition(
                    key="ui.language",
                    category=PreferenceCategory.INTERFACE,
                    preference_type=PreferenceType.STRING,
                    default_value="en",
                    description="User interface language"
                ),
                PreferenceDefinition(
                    key="notifications.email_enabled",
                    category=PreferenceCategory.NOTIFICATIONS,
                    preference_type=PreferenceType.BOOLEAN,
                    default_value=True,
                    description="Enable email notifications"
                ),
                PreferenceDefinition(
                    key="trading.default_order_size",
                    category=PreferenceCategory.TRADING,
                    preference_type=PreferenceType.FLOAT,
                    default_value=1000.0,
                    description="Default order size for trading",
                    min_value=1.0,
                    max_value=1000000.0
                ),
                PreferenceDefinition(
                    key="analytics.chart_refresh_interval",
                    category=PreferenceCategory.ANALYTICS,
                    preference_type=PreferenceType.INTEGER,
                    default_value=30,
                    description="Chart refresh interval in seconds",
                    min_value=5,
                    max_value=300
                )
            ]
            
            for definition in default_preferences:
                if definition.key not in self.preference_definitions:
                    await self.register_preference(definition)
            
        except Exception as e:
            logger.error(f"Error initializing default preferences: {e}")
    
    def _get_cache_key(self, scope: PreferenceScope, user_id: Optional[str], key: str,
                      organization_id: Optional[str] = None, device_id: Optional[str] = None,
                      session_id: Optional[str] = None) -> str:
        """Generate cache key for preference"""
        return f"{scope.value}:{user_id or 'none'}:{organization_id or 'none'}:{device_id or 'none'}:{session_id or 'none'}:{key}"
    
    def _calculate_checksum(self, value: Any) -> str:
        """Calculate checksum for preference value"""
        try:
            value_str = json.dumps(value, sort_keys=True)
            return hashlib.md5(value_str.encode()).hexdigest()
        except:
            return ""
    
    async def _validate_preference_definition(self, definition: PreferenceDefinition) -> bool:
        """Validate preference definition"""
        try:
            # Check required fields
            if not definition.key or not definition.description:
                return False
            
            # Validate key format
            if not definition.key.replace('.', '').replace('_', '').isalnum():
                return False
            
            # Validate default value type
            if definition.default_value is not None:
                if not await self._validate_value_type(definition.default_value, definition.preference_type):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating preference definition: {e}")
            return False
    
    async def _validate_value_type(self, value: Any, expected_type: PreferenceType) -> bool:
        """Validate value matches expected type"""
        try:
            if expected_type == PreferenceType.BOOLEAN:
                return isinstance(value, bool)
            elif expected_type == PreferenceType.INTEGER:
                return isinstance(value, int)
            elif expected_type == PreferenceType.FLOAT:
                return isinstance(value, (int, float))
            elif expected_type == PreferenceType.STRING:
                return isinstance(value, str)
            elif expected_type == PreferenceType.LIST:
                return isinstance(value, list)
            elif expected_type == PreferenceType.DICT:
                return isinstance(value, dict)
            elif expected_type == PreferenceType.JSON:
                # Any JSON-serializable value
                try:
                    json.dumps(value)
                    return True
                except:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating value type: {e}")
            return False
    
    async def _validate_preference_value(self, key: str, value: Any) -> bool:
        """Validate preference value against definition"""
        try:
            if key not in self.preference_definitions:
                return False
            
            definition = self.preference_definitions[key]
            
            # Type validation
            if not await self._validate_value_type(value, definition.preference_type):
                return False
            
            # Range validation
            if isinstance(value, (int, float)):
                if definition.min_value is not None and value < definition.min_value:
                    return False
                if definition.max_value is not None and value > definition.max_value:
                    return False
            
            # Allowed values validation
            if definition.allowed_values and value not in definition.allowed_values:
                return False
            
            # Custom validator
            if key in self.validators:
                return await self.validators[key](value)
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating preference value: {e}")
            return False
    
    async def _transform_preference_value(self, key: str, value: Any) -> Any:
        """Transform preference value if transformer is registered"""
        try:
            if key in self.transformers:
                return await self.transformers[key](value)
            return value
            
        except Exception as e:
            logger.error(f"Error transforming preference value: {e}")
            return value
    
    async def _check_preference_permissions(self, key: str, user_id: Optional[str], operation: str) -> bool:
        """Check if user has permission for preference operation"""
        try:
            if key not in self.preference_definitions:
                return False
            
            definition = self.preference_definitions[key]
            
            # Admin-only preferences
            if definition.admin_only:
                # In real implementation, check if user is admin
                return True  # Simplified for now
            
            # User-configurable check
            if operation in ['write', 'delete'] and not definition.user_configurable:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking preference permissions: {e}")
            return False
    
    async def _get_preference_with_hierarchy(self, key: str, scope: PreferenceScope,
                                           user_id: Optional[str], organization_id: Optional[str],
                                           device_id: Optional[str], session_id: Optional[str]) -> Any:
        """Get preference value with scope hierarchy fallback"""
        try:
            # Define scope hierarchy (most specific to least specific)
            scope_hierarchy = []
            
            if scope == PreferenceScope.SESSION and session_id:
                scope_hierarchy.append((PreferenceScope.SESSION, user_id, organization_id, device_id, session_id))
            
            if scope in [PreferenceScope.DEVICE, PreferenceScope.SESSION] and device_id:
                scope_hierarchy.append((PreferenceScope.DEVICE, user_id, organization_id, device_id, None))
            
            if scope in [PreferenceScope.USER, PreferenceScope.DEVICE, PreferenceScope.SESSION] and user_id:
                scope_hierarchy.append((PreferenceScope.USER, user_id, None, None, None))
            
            if organization_id:
                scope_hierarchy.append((PreferenceScope.ORGANIZATION, None, organization_id, None, None))
            
            scope_hierarchy.append((PreferenceScope.GLOBAL, None, None, None, None))
            
            # Try each scope in hierarchy
            for scope_level, uid, oid, did, sid in scope_hierarchy:
                value = await self._get_preference_from_storage(key, scope_level, uid, oid, did, sid)
                if value is not None:
                    return value
            
            # Fall back to default value
            if key in self.preference_definitions:
                return self.preference_definitions[key].default_value
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting preference with hierarchy: {e}")
            return None
    
    async def _get_preference_from_storage(self, key: str, scope: PreferenceScope,
                                         user_id: Optional[str], organization_id: Optional[str],
                                         device_id: Optional[str], session_id: Optional[str]) -> Any:
        """Get preference value from storage (cache or database)"""
        try:
            # Check cache first
            cache_key = self._get_cache_key(scope, user_id, key, organization_id, device_id, session_id)
            if cache_key in self.preference_cache:
                return self.preference_cache[cache_key].value
            
            # Load from database
            async with aiosqlite.connect(self.db_path) as db:
                query = """
                    SELECT value FROM preference_values 
                    WHERE key = ? AND scope = ? AND 
                          (user_id = ? OR user_id IS NULL) AND
                          (organization_id = ? OR organization_id IS NULL) AND
                          (device_id = ? OR device_id IS NULL) AND
                          (session_id = ? OR session_id IS NULL)
                    ORDER BY priority DESC, updated_at DESC
                    LIMIT 1
                """
                
                async with db.execute(query, (key, scope.value, user_id, organization_id, device_id, session_id)) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        value = json.loads(row[0]) if row[0] else None
                        
                        # Cache the value
                        pref_value = PreferenceValue(
                            key=key,
                            value=value,
                            scope=scope,
                            user_id=user_id,
                            organization_id=organization_id,
                            device_id=device_id,
                            session_id=session_id
                        )
                        self.preference_cache[cache_key] = pref_value
                        self.cache_timestamps[cache_key] = datetime.now()
                        
                        return value
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting preference from storage: {e}")
            return None
    
    async def _save_preference_definition(self, definition: PreferenceDefinition):
        """Save preference definition to database"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO preference_definitions (
                        key, category, preference_type, default_value, description,
                        scope, priority, validation_rules, allowed_values,
                        min_value, max_value, requires_restart, sensitive,
                        user_configurable, admin_only, deprecated, migration_path,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    definition.key,
                    definition.category.value,
                    definition.preference_type.value,
                    json.dumps(definition.default_value),
                    definition.description,
                    definition.scope.value,
                    definition.priority.value,
                    json.dumps(definition.validation_rules),
                    json.dumps(definition.allowed_values),
                    definition.min_value,
                    definition.max_value,
                    definition.requires_restart,
                    definition.sensitive,
                    definition.user_configurable,
                    definition.admin_only,
                    definition.deprecated,
                    definition.migration_path,
                    definition.created_at.isoformat(),
                    definition.updated_at.isoformat()
                ))
                await db.commit()
            
        except Exception as e:
            logger.error(f"Error saving preference definition: {e}")
    
    async def _save_preference_value(self, pref_value: PreferenceValue):
        """Save preference value to database"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO preference_values (
                        key, value, scope, user_id, organization_id, device_id, session_id,
                        priority, sync_status, created_at, updated_at, expires_at,
                        metadata, version, checksum
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pref_value.key,
                    json.dumps(pref_value.value),
                    pref_value.scope.value,
                    pref_value.user_id,
                    pref_value.organization_id,
                    pref_value.device_id,
                    pref_value.session_id,
                    pref_value.priority.value,
                    pref_value.sync_status.value,
                    pref_value.created_at.isoformat(),
                    pref_value.updated_at.isoformat(),
                    pref_value.expires_at.isoformat() if pref_value.expires_at else None,
                    json.dumps(pref_value.metadata),
                    pref_value.version,
                    pref_value.checksum
                ))
                await db.commit()
            
        except Exception as e:
            logger.error(f"Error saving preference value: {e}")
    
    async def _delete_preference_value(self, key: str, scope: PreferenceScope,
                                     user_id: Optional[str], organization_id: Optional[str],
                                     device_id: Optional[str], session_id: Optional[str]):
        """Delete preference value from database"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    DELETE FROM preference_values 
                    WHERE key = ? AND scope = ? AND 
                          user_id = ? AND organization_id = ? AND 
                          device_id = ? AND session_id = ?
                """, (key, scope.value, user_id, organization_id, device_id, session_id))
                await db.commit()
            
        except Exception as e:
            logger.error(f"Error deleting preference value: {e}")
    
    async def _save_preference_backup(self, backup: PreferenceBackup):
        """Save preference backup to database"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO preference_backups (
                        backup_id, user_id, preferences, backup_type,
                        created_at, description, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    backup.backup_id,
                    backup.user_id,
                    json.dumps(backup.preferences),
                    backup.backup_type,
                    backup.created_at.isoformat(),
                    backup.description,
                    json.dumps(backup.metadata)
                ))
                await db.commit()
            
        except Exception as e:
            logger.error(f"Error saving preference backup: {e}")
    
    async def _load_preference_backup(self, backup_id: str) -> Optional[PreferenceBackup]:
        """Load preference backup from database"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(
                    "SELECT * FROM preference_backups WHERE backup_id = ?", 
                    (backup_id,)
                ) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        return PreferenceBackup(
                            backup_id=row[0],
                            user_id=row[1],
                            preferences=json.loads(row[2]),
                            backup_type=row[3],
                            created_at=datetime.fromisoformat(row[4]),
                            description=row[5],
                            metadata=json.loads(row[6]) if row[6] else {}
                        )
            
            return None
            
        except Exception as e:
            logger.error(f"Error loading preference backup: {e}")
            return None
    
    async def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached preference is still valid"""
        try:
            if cache_key not in self.cache_timestamps:
                return False
            
            # Cache is valid for 5 minutes
            cache_age = datetime.now() - self.cache_timestamps[cache_key]
            return cache_age < timedelta(minutes=5)
            
        except Exception as e:
            logger.error(f"Error checking cache validity: {e}")
            return False
    
    async def _notify_preference_change(self, change: PreferenceChange):
        """Notify listeners of preference change"""
        try:
            # Notify specific key listeners
            for callback in self.change_listeners.get(change.key, []):
                try:
                    await callback(change)
                except Exception as e:
                    logger.error(f"Error in change listener callback: {e}")
            
            # Notify wildcard listeners
            for callback in self.change_listeners.get('*', []):
                try:
                    await callback(change)
                except Exception as e:
                    logger.error(f"Error in wildcard change listener callback: {e}")
            
        except Exception as e:
            logger.error(f"Error notifying preference change: {e}")
    
    def _create_validator(self, definition: PreferenceDefinition) -> Callable:
        """Create validator function from definition rules"""
        async def validator(value: Any) -> bool:
            try:
                rules = definition.validation_rules
                
                # Custom validation rules can be implemented here
                if 'regex' in rules and isinstance(value, str):
                    import re
                    pattern = rules['regex']
                    return bool(re.match(pattern, value))
                
                if 'length' in rules and isinstance(value, (str, list)):
                    min_len = rules['length'].get('min', 0)
                    max_len = rules['length'].get('max', float('inf'))
                    return min_len <= len(value) <= max_len
                
                return True
                
            except Exception as e:
                logger.error(f"Error in custom validator: {e}")
                return False
        
        return validator
    
    async def _sync_preference_to_remote(self, pref_value: PreferenceValue) -> bool:
        """Sync preference to remote storage (placeholder)"""
        try:
            # In real implementation, this would sync to external service
            # For now, just simulate success
            await asyncio.sleep(0.1)  # Simulate network delay
            return True
            
        except Exception as e:
            logger.error(f"Error syncing preference to remote: {e}")
            return False
    
    async def _update_preference_sync_status(self, pref_value: PreferenceValue):
        """Update preference sync status in database"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    UPDATE preference_values 
                    SET sync_status = ?, updated_at = ?
                    WHERE key = ? AND scope = ? AND user_id = ?
                """, (
                    pref_value.sync_status.value,
                    datetime.now().isoformat(),
                    pref_value.key,
                    pref_value.scope.value,
                    pref_value.user_id
                ))
                await db.commit()
            
        except Exception as e:
            logger.error(f"Error updating preference sync status: {e}")
    
    async def _detect_sync_conflicts(self, user_id: str) -> List[PreferenceConflict]:
        """Detect synchronization conflicts (placeholder)"""
        try:
            # In real implementation, this would compare local and remote values
            # For now, return empty list
            return []
            
        except Exception as e:
            logger.error(f"Error detecting sync conflicts: {e}")
            return []
    
    async def _merge_preference_values(self, local_value: Any, remote_value: Any) -> Any:
        """Merge conflicting preference values"""
        try:
            # Simple merge strategy - prefer remote for now
            # In real implementation, this would be more sophisticated
            return remote_value
            
        except Exception as e:
            logger.error(f"Error merging preference values: {e}")
            return local_value
    
    async def _cleanup_old_backups(self, cutoff_date: datetime):
        """Clean up old preference backups"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    "DELETE FROM preference_backups WHERE created_at < ?",
                    (cutoff_date.isoformat(),)
                )
                await db.commit()
            
        except Exception as e:
            logger.error(f"Error cleaning up old backups: {e}")

# Export classes and functions
__all__ = [
    'PreferenceCategory',
    'PreferenceType',
    'PreferenceScope',
    'PreferencePriority',
    'SyncStatus',
    'PreferenceDefinition',
    'PreferenceValue',
    'PreferenceChange',
    'PreferenceConflict',
    'PreferenceBackup',
    'PreferenceManager'
]