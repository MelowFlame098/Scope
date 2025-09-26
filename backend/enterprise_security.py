from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from database import get_db
from auth import get_current_user
import hashlib
import secrets
import json

router = APIRouter(prefix="/security", tags=["enterprise-security"])

# Security models
class SecurityEvent(BaseModel):
    id: str
    event_type: str  # login, logout, failed_login, data_access, api_call
    user_id: str
    ip_address: str
    user_agent: str
    timestamp: datetime
    severity: str  # low, medium, high, critical
    details: Dict[str, Any]
    resolved: bool = False

class AccessPolicy(BaseModel):
    id: str
    name: str
    description: str
    rules: List[Dict[str, Any]]
    enabled: bool = True
    created_at: datetime
    updated_at: datetime

class SecurityAudit(BaseModel):
    id: str
    audit_type: str  # access, data, system, compliance
    user_id: Optional[str] = None
    resource: str
    action: str
    timestamp: datetime
    ip_address: str
    success: bool
    details: Dict[str, Any]

class ThreatDetection(BaseModel):
    id: str
    threat_type: str  # brute_force, suspicious_activity, data_breach
    severity: str
    description: str
    affected_users: List[str]
    detected_at: datetime
    status: str  # active, investigating, resolved
    mitigation_steps: List[str]

class ComplianceReport(BaseModel):
    id: str
    report_type: str  # sox, gdpr, pci_dss, iso27001
    period_start: datetime
    period_end: datetime
    compliance_score: float
    violations: List[Dict[str, Any]]
    recommendations: List[str]
    generated_at: datetime

class SecurityMetrics(BaseModel):
    total_events: int
    critical_events: int
    failed_logins: int
    successful_logins: int
    data_access_events: int
    threat_detections: int
    compliance_score: float
    last_updated: datetime

class EnterpriseSecurityService:
    def __init__(self):
        self.security_events = []
        self.access_policies = []
        self.audit_logs = []
        self.threat_detections = []
        self.compliance_reports = []
        self._initialize_default_policies()
        
    def _initialize_default_policies(self):
        """Initialize default security policies"""
        default_policies = [
            {
                "id": "policy_001",
                "name": "Multi-Factor Authentication",
                "description": "Require MFA for all users",
                "rules": [
                    {"type": "mfa_required", "value": True},
                    {"type": "mfa_methods", "value": ["totp", "sms", "email"]}
                ],
                "enabled": True,
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            },
            {
                "id": "policy_002",
                "name": "Password Policy",
                "description": "Strong password requirements",
                "rules": [
                    {"type": "min_length", "value": 12},
                    {"type": "require_uppercase", "value": True},
                    {"type": "require_lowercase", "value": True},
                    {"type": "require_numbers", "value": True},
                    {"type": "require_symbols", "value": True},
                    {"type": "password_expiry_days", "value": 90}
                ],
                "enabled": True,
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            },
            {
                "id": "policy_003",
                "name": "Session Management",
                "description": "Secure session handling",
                "rules": [
                    {"type": "session_timeout_minutes", "value": 30},
                    {"type": "concurrent_sessions", "value": 3},
                    {"type": "secure_cookies", "value": True}
                ],
                "enabled": True,
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            }
        ]
        
        for policy_data in default_policies:
            policy = AccessPolicy(**policy_data)
            self.access_policies.append(policy)
    
    def log_security_event(self, event_type: str, user_id: str, ip_address: str, 
                          user_agent: str, details: Dict[str, Any], 
                          severity: str = "medium") -> SecurityEvent:
        """Log a security event"""
        event = SecurityEvent(
            id=f"event_{len(self.security_events) + 1}",
            event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            timestamp=datetime.now(),
            severity=severity,
            details=details
        )
        
        self.security_events.append(event)
        
        # Check for threats
        self._analyze_for_threats(event)
        
        return event
    
    def _analyze_for_threats(self, event: SecurityEvent):
        """Analyze security event for potential threats"""
        # Check for brute force attacks
        if event.event_type == "failed_login":
            recent_failures = [
                e for e in self.security_events 
                if e.event_type == "failed_login" 
                and e.user_id == event.user_id 
                and e.timestamp > datetime.now() - timedelta(minutes=15)
            ]
            
            if len(recent_failures) >= 5:
                threat = ThreatDetection(
                    id=f"threat_{len(self.threat_detections) + 1}",
                    threat_type="brute_force",
                    severity="high",
                    description=f"Multiple failed login attempts for user {event.user_id}",
                    affected_users=[event.user_id],
                    detected_at=datetime.now(),
                    status="active",
                    mitigation_steps=[
                        "Lock user account",
                        "Notify security team",
                        "Review access logs"
                    ]
                )
                self.threat_detections.append(threat)
    
    def get_security_events(self, limit: int = 100, severity: Optional[str] = None) -> List[SecurityEvent]:
        """Get security events with optional filtering"""
        events = self.security_events
        
        if severity:
            events = [e for e in events if e.severity == severity]
        
        return sorted(events, key=lambda x: x.timestamp, reverse=True)[:limit]
    
    def get_access_policies(self) -> List[AccessPolicy]:
        """Get all access policies"""
        return self.access_policies
    
    def create_access_policy(self, policy_data: Dict[str, Any]) -> AccessPolicy:
        """Create a new access policy"""
        policy = AccessPolicy(
            id=f"policy_{len(self.access_policies) + 1}",
            name=policy_data["name"],
            description=policy_data["description"],
            rules=policy_data["rules"],
            enabled=policy_data.get("enabled", True),
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self.access_policies.append(policy)
        return policy
    
    def audit_log(self, audit_type: str, user_id: Optional[str], resource: str, 
                  action: str, ip_address: str, success: bool, 
                  details: Dict[str, Any]) -> SecurityAudit:
        """Create an audit log entry"""
        audit = SecurityAudit(
            id=f"audit_{len(self.audit_logs) + 1}",
            audit_type=audit_type,
            user_id=user_id,
            resource=resource,
            action=action,
            timestamp=datetime.now(),
            ip_address=ip_address,
            success=success,
            details=details
        )
        
        self.audit_logs.append(audit)
        return audit
    
    def get_audit_logs(self, limit: int = 100, audit_type: Optional[str] = None) -> List[SecurityAudit]:
        """Get audit logs with optional filtering"""
        logs = self.audit_logs
        
        if audit_type:
            logs = [l for l in logs if l.audit_type == audit_type]
        
        return sorted(logs, key=lambda x: x.timestamp, reverse=True)[:limit]
    
    def get_threat_detections(self, status: Optional[str] = None) -> List[ThreatDetection]:
        """Get threat detections with optional status filtering"""
        threats = self.threat_detections
        
        if status:
            threats = [t for t in threats if t.status == status]
        
        return sorted(threats, key=lambda x: x.detected_at, reverse=True)
    
    def generate_compliance_report(self, report_type: str, 
                                 period_start: datetime, 
                                 period_end: datetime) -> ComplianceReport:
        """Generate a compliance report"""
        # Mock compliance analysis
        violations = []
        compliance_score = 85.5
        
        # Check for common violations
        if report_type == "gdpr":
            violations = [
                {"type": "data_retention", "description": "Some user data retained beyond policy", "severity": "medium"},
                {"type": "consent_tracking", "description": "Missing consent records for 2 users", "severity": "low"}
            ]
        elif report_type == "sox":
            violations = [
                {"type": "access_control", "description": "Privileged access not properly documented", "severity": "high"}
            ]
        
        report = ComplianceReport(
            id=f"report_{len(self.compliance_reports) + 1}",
            report_type=report_type,
            period_start=period_start,
            period_end=period_end,
            compliance_score=compliance_score,
            violations=violations,
            recommendations=[
                "Implement automated data retention policies",
                "Enhance access control documentation",
                "Regular compliance training for staff"
            ],
            generated_at=datetime.now()
        )
        
        self.compliance_reports.append(report)
        return report
    
    def get_security_metrics(self) -> SecurityMetrics:
        """Get security metrics dashboard"""
        total_events = len(self.security_events)
        critical_events = len([e for e in self.security_events if e.severity == "critical"])
        failed_logins = len([e for e in self.security_events if e.event_type == "failed_login"])
        successful_logins = len([e for e in self.security_events if e.event_type == "login"])
        data_access_events = len([e for e in self.security_events if e.event_type == "data_access"])
        threat_detections = len(self.threat_detections)
        
        return SecurityMetrics(
            total_events=total_events,
            critical_events=critical_events,
            failed_logins=failed_logins,
            successful_logins=successful_logins,
            data_access_events=data_access_events,
            threat_detections=threat_detections,
            compliance_score=87.5,
            last_updated=datetime.now()
        )

# Initialize service
enterprise_security_service = EnterpriseSecurityService()

# API Endpoints
@router.get("/events")
async def get_security_events(
    limit: int = 100,
    severity: Optional[str] = None,
    current_user = Depends(get_current_user)
):
    """Get security events"""
    events = enterprise_security_service.get_security_events(limit, severity)
    return events

@router.post("/events")
async def log_security_event(
    event_data: Dict[str, Any],
    request: Request,
    current_user = Depends(get_current_user)
):
    """Log a security event"""
    event = enterprise_security_service.log_security_event(
        event_type=event_data["event_type"],
        user_id=current_user.id,
        ip_address=request.client.host,
        user_agent=request.headers.get("user-agent", ""),
        details=event_data.get("details", {}),
        severity=event_data.get("severity", "medium")
    )
    return event

@router.get("/policies")
async def get_access_policies(current_user = Depends(get_current_user)):
    """Get access policies"""
    policies = enterprise_security_service.get_access_policies()
    return policies

@router.post("/policies")
async def create_access_policy(
    policy_data: Dict[str, Any],
    current_user = Depends(get_current_user)
):
    """Create a new access policy"""
    policy = enterprise_security_service.create_access_policy(policy_data)
    return policy

@router.get("/audit")
async def get_audit_logs(
    limit: int = 100,
    audit_type: Optional[str] = None,
    current_user = Depends(get_current_user)
):
    """Get audit logs"""
    logs = enterprise_security_service.get_audit_logs(limit, audit_type)
    return logs

@router.get("/threats")
async def get_threat_detections(
    status: Optional[str] = None,
    current_user = Depends(get_current_user)
):
    """Get threat detections"""
    threats = enterprise_security_service.get_threat_detections(status)
    return threats

@router.post("/compliance/report")
async def generate_compliance_report(
    report_data: Dict[str, Any],
    current_user = Depends(get_current_user)
):
    """Generate a compliance report"""
    report = enterprise_security_service.generate_compliance_report(
        report_type=report_data["report_type"],
        period_start=datetime.fromisoformat(report_data["period_start"]),
        period_end=datetime.fromisoformat(report_data["period_end"])
    )
    return report

@router.get("/metrics")
async def get_security_metrics(current_user = Depends(get_current_user)):
    """Get security metrics"""
    metrics = enterprise_security_service.get_security_metrics()
    return metrics

@router.get("/health")
async def security_health_check():
    """Security service health check"""
    return {
        "status": "healthy",
        "service": "enterprise_security",
        "timestamp": datetime.now(),
        "features": [
            "security_events",
            "access_policies",
            "audit_logging",
            "threat_detection",
            "compliance_reporting"
        ]
    }

# Export service and router
__all__ = ["EnterpriseSecurityService", "enterprise_security_service", "router"]