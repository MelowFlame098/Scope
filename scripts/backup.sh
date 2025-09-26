#!/bin/bash

# FinScope Database Backup Script
# This script creates automated backups of the PostgreSQL database

set -e

# Configuration
BACKUP_DIR="/backups"
DB_HOST="postgres"
DB_PORT="5432"
DB_NAME="${POSTGRES_DB:-finscope}"
DB_USER="${POSTGRES_USER:-finscope_user}"
RETENTION_DAYS=30
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="${BACKUP_DIR}/finscope_backup_${TIMESTAMP}.sql"
COMPRESSED_FILE="${BACKUP_FILE}.gz"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] SUCCESS:${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# Function to check if database is accessible
check_database() {
    log "Checking database connectivity..."
    if pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" > /dev/null 2>&1; then
        log_success "Database is accessible"
        return 0
    else
        log_error "Database is not accessible"
        return 1
    fi
}

# Function to create database backup
create_backup() {
    log "Starting database backup..."
    log "Database: $DB_NAME"
    log "Host: $DB_HOST:$DB_PORT"
    log "User: $DB_USER"
    log "Backup file: $BACKUP_FILE"
    
    # Create the backup
    if pg_dump -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
        --verbose \
        --no-password \
        --format=custom \
        --compress=9 \
        --file="$BACKUP_FILE" 2>/dev/null; then
        
        log_success "Database backup created successfully"
        
        # Get backup file size
        BACKUP_SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
        log "Backup size: $BACKUP_SIZE"
        
        return 0
    else
        log_error "Failed to create database backup"
        return 1
    fi
}

# Function to compress backup
compress_backup() {
    if [ -f "$BACKUP_FILE" ]; then
        log "Compressing backup file..."
        
        if gzip "$BACKUP_FILE"; then
            log_success "Backup compressed successfully"
            COMPRESSED_SIZE=$(du -h "$COMPRESSED_FILE" | cut -f1)
            log "Compressed size: $COMPRESSED_SIZE"
            return 0
        else
            log_error "Failed to compress backup"
            return 1
        fi
    else
        log_error "Backup file not found for compression"
        return 1
    fi
}

# Function to clean old backups
cleanup_old_backups() {
    log "Cleaning up old backups (older than $RETENTION_DAYS days)..."
    
    # Count files before cleanup
    OLD_COUNT=$(find "$BACKUP_DIR" -name "finscope_backup_*.sql.gz" -mtime +$RETENTION_DAYS | wc -l)
    
    if [ "$OLD_COUNT" -gt 0 ]; then
        log "Found $OLD_COUNT old backup(s) to remove"
        
        # Remove old backups
        find "$BACKUP_DIR" -name "finscope_backup_*.sql.gz" -mtime +$RETENTION_DAYS -delete
        
        log_success "Removed $OLD_COUNT old backup(s)"
    else
        log "No old backups to remove"
    fi
    
    # Show current backup count
    CURRENT_COUNT=$(find "$BACKUP_DIR" -name "finscope_backup_*.sql.gz" | wc -l)
    log "Current backup count: $CURRENT_COUNT"
}

# Function to verify backup integrity
verify_backup() {
    if [ -f "$COMPRESSED_FILE" ]; then
        log "Verifying backup integrity..."
        
        # Test gzip integrity
        if gzip -t "$COMPRESSED_FILE" 2>/dev/null; then
            log_success "Backup file integrity verified"
            return 0
        else
            log_error "Backup file is corrupted"
            return 1
        fi
    else
        log_error "Compressed backup file not found"
        return 1
    fi
}

# Function to send notification (placeholder)
send_notification() {
    local status=$1
    local message=$2
    
    # This is a placeholder for notification logic
    # You can integrate with services like Slack, Discord, email, etc.
    log "Notification: $status - $message"
    
    # Example: Send to webhook (uncomment and configure as needed)
    # if [ -n "$WEBHOOK_URL" ]; then
    #     curl -X POST "$WEBHOOK_URL" \
    #         -H "Content-Type: application/json" \
    #         -d "{\"text\": \"FinScope Backup $status: $message\"}"
    # fi
}

# Main backup process
main() {
    log "=== FinScope Database Backup Started ==="
    
    local success=true
    
    # Check database connectivity
    if ! check_database; then
        send_notification "FAILED" "Database connectivity check failed"
        exit 1
    fi
    
    # Create backup
    if ! create_backup; then
        send_notification "FAILED" "Database backup creation failed"
        exit 1
    fi
    
    # Compress backup
    if ! compress_backup; then
        send_notification "FAILED" "Backup compression failed"
        success=false
    fi
    
    # Verify backup
    if ! verify_backup; then
        send_notification "FAILED" "Backup verification failed"
        success=false
    fi
    
    # Cleanup old backups
    cleanup_old_backups
    
    if [ "$success" = true ]; then
        log_success "=== FinScope Database Backup Completed Successfully ==="
        send_notification "SUCCESS" "Database backup completed successfully"
    else
        log_error "=== FinScope Database Backup Completed with Errors ==="
        send_notification "WARNING" "Database backup completed with some errors"
    fi
}

# Handle script interruption
trap 'log_error "Backup process interrupted"; exit 1' INT TERM

# Run main function
main "$@"