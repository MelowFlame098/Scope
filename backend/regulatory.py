from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import json

router = APIRouter(prefix="/api/regulatory", tags=["regulatory"])

class ComplianceStatus(str, Enum):
    COMPLIANT = "compliant"
    WARNING = "warning"
    VIOLATION = "violation"
    PENDING = "pending"
    RESOLVED = "resolved"

class RuleSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ReportType(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    AD_HOC = "ad_hoc"

class RegulatoryFramework(str, Enum):
    SEC = "sec"
    FINRA = "finra"
    CFTC = "cftc"
    MIFID_II = "mifid_ii"
    BASEL_III = "basel_iii"
    GDPR = "gdpr"
    SOX = "sox"
    DODD_FRANK = "dodd_frank"

class AuditStatus(str, Enum):
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

# Pydantic Models
class ComplianceRule(BaseModel):
    id: str
    name: str
    description: str
    framework: RegulatoryFramework
    category: str
    severity: RuleSeverity
    threshold: Optional[float] = None
    thresholdType: Optional[str] = None  # percentage, amount, ratio
    isActive: bool
    createdAt: datetime
    updatedAt: datetime
    lastChecked: Optional[datetime] = None
    checkFrequency: str  # daily, weekly, monthly
    automatedCheck: bool
    documentation: str
    penalties: List[str] = []
    exemptions: List[str] = []

class ComplianceViolation(BaseModel):
    id: str
    ruleId: str
    rule: ComplianceRule
    portfolioId: Optional[str] = None
    clientId: Optional[str] = None
    description: str
    severity: RuleSeverity
    status: ComplianceStatus
    detectedAt: datetime
    resolvedAt: Optional[datetime] = None
    currentValue: float
    thresholdValue: float
    violationAmount: Optional[float] = None
    impact: str
    actionRequired: str
    assignedTo: Optional[str] = None
    notes: str = ""
    attachments: List[str] = []
    remediation: Optional[str] = None

class RegulatoryReport(BaseModel):
    id: str
    name: str
    type: ReportType
    framework: RegulatoryFramework
    description: str
    generatedAt: datetime
    periodStart: datetime
    periodEnd: datetime
    status: str
    fileUrl: Optional[str] = None
    submittedAt: Optional[datetime] = None
    submittedBy: Optional[str] = None
    acknowledgmentId: Optional[str] = None
    dueDate: datetime
    isOverdue: bool
    summary: Dict[str, Any] = {}
    sections: List[Dict[str, Any]] = []

class AuditTrail(BaseModel):
    id: str
    timestamp: datetime
    userId: str
    userName: str
    action: str
    resource: str
    resourceId: str
    details: Dict[str, Any]
    ipAddress: str
    userAgent: str
    sessionId: str
    outcome: str  # success, failure, warning
    riskLevel: str

class ComplianceAudit(BaseModel):
    id: str
    name: str
    type: str  # internal, external, regulatory
    framework: RegulatoryFramework
    auditor: str
    auditorContact: str
    scheduledDate: datetime
    startDate: Optional[datetime] = None
    endDate: Optional[datetime] = None
    status: AuditStatus
    scope: List[str] = []
    findings: List[Dict[str, Any]] = []
    recommendations: List[str] = []
    actionItems: List[Dict[str, Any]] = []
    riskRating: str
    complianceScore: float
    reportUrl: Optional[str] = None
    followUpDate: Optional[datetime] = None

class RiskAssessment(BaseModel):
    id: str
    name: str
    type: str  # operational, market, credit, liquidity, compliance
    description: str
    assessmentDate: datetime
    assessor: str
    riskLevel: str  # low, medium, high, critical
    probability: float  # 0-1
    impact: float  # 0-1
    riskScore: float  # probability * impact
    mitigationStrategies: List[str] = []
    controls: List[Dict[str, Any]] = []
    residualRisk: float
    reviewDate: datetime
    status: str
    owner: str

class PolicyDocument(BaseModel):
    id: str
    title: str
    category: str
    framework: RegulatoryFramework
    version: str
    description: str
    content: str
    createdAt: datetime
    updatedAt: datetime
    effectiveDate: datetime
    expiryDate: Optional[datetime] = None
    approvedBy: str
    reviewDate: datetime
    isActive: bool
    tags: List[str] = []
    attachments: List[str] = []

class ComplianceMetrics(BaseModel):
    date: datetime
    totalRules: int
    activeRules: int
    complianceScore: float
    violationsCount: int
    criticalViolations: int
    highViolations: int
    mediumViolations: int
    lowViolations: int
    resolvedViolations: int
    avgResolutionTime: float  # hours
    overdueReports: int
    completedAudits: int
    riskScore: float
    trendsData: Dict[str, List[float]] = {}

class RuleRequest(BaseModel):
    name: str
    description: str
    framework: RegulatoryFramework
    category: str
    severity: RuleSeverity
    threshold: Optional[float] = None
    thresholdType: Optional[str] = None
    checkFrequency: str
    automatedCheck: bool = True
    documentation: str

class ViolationRequest(BaseModel):
    ruleId: str
    portfolioId: Optional[str] = None
    clientId: Optional[str] = None
    description: str
    currentValue: float
    impact: str
    actionRequired: str
    assignedTo: Optional[str] = None

# Service Class
class RegulatoryService:
    def __init__(self):
        self.rules = {}
        self.violations = {}
        self.reports = {}
        self.audit_trails = {}
        self.audits = {}
        self.risk_assessments = {}
        self.policies = {}
        self._initialize_mock_data()
    
    def _initialize_mock_data(self):
        """Initialize with mock data for demonstration"""
        # Mock compliance rules
        mock_rules = [
            {
                "id": "rule-1",
                "name": "Position Concentration Limit",
                "description": "No single position should exceed 5% of total portfolio value",
                "framework": RegulatoryFramework.SEC,
                "category": "Risk Management",
                "severity": RuleSeverity.HIGH,
                "threshold": 5.0,
                "thresholdType": "percentage",
                "isActive": True,
                "createdAt": datetime.now() - timedelta(days=365),
                "updatedAt": datetime.now() - timedelta(days=30),
                "lastChecked": datetime.now() - timedelta(hours=1),
                "checkFrequency": "daily",
                "automatedCheck": True,
                "documentation": "SEC Rule 3a-4 regarding investment company diversification",
                "penalties": ["Warning letter", "Fine up to $100,000", "License suspension"]
            },
            {
                "id": "rule-2",
                "name": "Liquidity Coverage Ratio",
                "description": "Maintain minimum 10% cash or cash equivalents",
                "framework": RegulatoryFramework.BASEL_III,
                "category": "Liquidity Risk",
                "severity": RuleSeverity.CRITICAL,
                "threshold": 10.0,
                "thresholdType": "percentage",
                "isActive": True,
                "createdAt": datetime.now() - timedelta(days=300),
                "updatedAt": datetime.now() - timedelta(days=15),
                "lastChecked": datetime.now() - timedelta(minutes=30),
                "checkFrequency": "daily",
                "automatedCheck": True,
                "documentation": "Basel III Liquidity Coverage Ratio requirements",
                "penalties": ["Regulatory action", "Capital requirements increase"]
            },
            {
                "id": "rule-3",
                "name": "Best Execution Reporting",
                "description": "Report execution quality metrics quarterly",
                "framework": RegulatoryFramework.MIFID_II,
                "category": "Trade Reporting",
                "severity": RuleSeverity.MEDIUM,
                "isActive": True,
                "createdAt": datetime.now() - timedelta(days=200),
                "updatedAt": datetime.now() - timedelta(days=10),
                "lastChecked": datetime.now() - timedelta(days=7),
                "checkFrequency": "quarterly",
                "automatedCheck": False,
                "documentation": "MiFID II RTS 27 best execution reporting",
                "penalties": ["Administrative fine", "Public censure"]
            }
        ]
        
        for rule_data in mock_rules:
            rule = ComplianceRule(**rule_data)
            self.rules[rule.id] = rule
        
        # Mock violations
        mock_violations = [
            {
                "id": "violation-1",
                "ruleId": "rule-1",
                "rule": self.rules["rule-1"],
                "portfolioId": "portfolio-1",
                "description": "AAPL position exceeds 5% concentration limit",
                "severity": RuleSeverity.HIGH,
                "status": ComplianceStatus.WARNING,
                "detectedAt": datetime.now() - timedelta(hours=2),
                "currentValue": 5.2,
                "thresholdValue": 5.0,
                "violationAmount": 3000000,
                "impact": "Increased concentration risk in technology sector",
                "actionRequired": "Reduce AAPL position to below 5% or obtain risk committee approval",
                "assignedTo": "Risk Manager",
                "notes": "Position increased due to recent price appreciation"
            },
            {
                "id": "violation-2",
                "ruleId": "rule-2",
                "rule": self.rules["rule-2"],
                "portfolioId": "portfolio-2",
                "description": "Cash position below minimum liquidity requirement",
                "severity": RuleSeverity.CRITICAL,
                "status": ComplianceStatus.VIOLATION,
                "detectedAt": datetime.now() - timedelta(hours=6),
                "currentValue": 8.5,
                "thresholdValue": 10.0,
                "violationAmount": 15000000,
                "impact": "Insufficient liquidity to meet redemption requests",
                "actionRequired": "Increase cash position immediately or liquidate positions",
                "assignedTo": "Portfolio Manager",
                "notes": "Large redemption request processed yesterday"
            }
        ]
        
        for violation_data in mock_violations:
            violation = ComplianceViolation(**violation_data)
            self.violations[violation.id] = violation
        
        # Mock reports
        mock_reports = [
            {
                "id": "report-1",
                "name": "Monthly Risk Report",
                "type": ReportType.MONTHLY,
                "framework": RegulatoryFramework.SEC,
                "description": "Monthly portfolio risk and compliance summary",
                "generatedAt": datetime.now() - timedelta(days=1),
                "periodStart": datetime.now() - timedelta(days=30),
                "periodEnd": datetime.now(),
                "status": "Generated",
                "dueDate": datetime.now() + timedelta(days=14),
                "isOverdue": False,
                "summary": {
                    "totalViolations": 2,
                    "criticalViolations": 1,
                    "complianceScore": 94.5,
                    "riskScore": 6.8
                }
            },
            {
                "id": "report-2",
                "name": "Quarterly Best Execution Report",
                "type": ReportType.QUARTERLY,
                "framework": RegulatoryFramework.MIFID_II,
                "description": "Quarterly execution quality analysis",
                "generatedAt": datetime.now() - timedelta(days=5),
                "periodStart": datetime.now() - timedelta(days=90),
                "periodEnd": datetime.now() - timedelta(days=1),
                "status": "Submitted",
                "submittedAt": datetime.now() - timedelta(days=3),
                "dueDate": datetime.now() - timedelta(days=1),
                "isOverdue": False,
                "summary": {
                    "totalTrades": 15420,
                    "avgExecutionQuality": 98.7,
                    "slippageBps": 2.3,
                    "fillRate": 99.1
                }
            }
        ]
        
        for report_data in mock_reports:
            report = RegulatoryReport(**report_data)
            self.reports[report.id] = report
        
        # Mock audit trails
        mock_audit_trails = [
            {
                "id": "audit-1",
                "timestamp": datetime.now() - timedelta(minutes=30),
                "userId": "user-123",
                "userName": "John Smith",
                "action": "CREATE_ORDER",
                "resource": "trading_order",
                "resourceId": "order-456",
                "details": {
                    "symbol": "AAPL",
                    "side": "buy",
                    "quantity": 1000,
                    "price": 175.50
                },
                "ipAddress": "192.168.1.100",
                "userAgent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "sessionId": "session-789",
                "outcome": "success",
                "riskLevel": "medium"
            }
        ]
        
        for audit_data in mock_audit_trails:
            audit = AuditTrail(**audit_data)
            self.audit_trails[audit.id] = audit
    
    async def get_compliance_overview(self) -> Dict[str, Any]:
        """Get compliance dashboard overview"""
        total_rules = len(self.rules)
        active_rules = len([r for r in self.rules.values() if r.isActive])
        total_violations = len(self.violations)
        critical_violations = len([v for v in self.violations.values() if v.severity == RuleSeverity.CRITICAL])
        pending_violations = len([v for v in self.violations.values() if v.status == ComplianceStatus.PENDING])
        overdue_reports = len([r for r in self.reports.values() if r.isOverdue])
        
        compliance_score = max(0, 100 - (total_violations * 5) - (critical_violations * 10))
        
        return {
            "complianceScore": compliance_score,
            "totalRules": total_rules,
            "activeRules": active_rules,
            "totalViolations": total_violations,
            "criticalViolations": critical_violations,
            "pendingViolations": pending_violations,
            "resolvedViolations": total_violations - pending_violations,
            "overdueReports": overdue_reports,
            "riskScore": 6.8,
            "trendsData": {
                "violations": [3, 2, 4, 1, 2, 3, 2],
                "compliance": [95, 94, 92, 96, 94, 93, 95]
            },
            "recentActivity": {
                "newViolations": 2,
                "resolvedViolations": 1,
                "generatedReports": 3,
                "completedAudits": 1
            }
        }
    
    async def get_compliance_rules(self, framework: Optional[RegulatoryFramework] = None, active_only: bool = True) -> List[ComplianceRule]:
        """Get compliance rules"""
        rules = list(self.rules.values())
        
        if framework:
            rules = [r for r in rules if r.framework == framework]
        
        if active_only:
            rules = [r for r in rules if r.isActive]
        
        rules.sort(key=lambda x: x.severity.value, reverse=True)
        return rules
    
    async def get_compliance_rule(self, rule_id: str) -> ComplianceRule:
        """Get specific compliance rule"""
        if rule_id not in self.rules:
            raise HTTPException(status_code=404, detail="Rule not found")
        return self.rules[rule_id]
    
    async def create_compliance_rule(self, rule_data: RuleRequest) -> ComplianceRule:
        """Create new compliance rule"""
        rule_id = f"rule-{int(datetime.now().timestamp())}"
        rule = ComplianceRule(
            id=rule_id,
            name=rule_data.name,
            description=rule_data.description,
            framework=rule_data.framework,
            category=rule_data.category,
            severity=rule_data.severity,
            threshold=rule_data.threshold,
            thresholdType=rule_data.thresholdType,
            isActive=True,
            createdAt=datetime.now(),
            updatedAt=datetime.now(),
            checkFrequency=rule_data.checkFrequency,
            automatedCheck=rule_data.automatedCheck,
            documentation=rule_data.documentation
        )
        
        self.rules[rule_id] = rule
        return rule
    
    async def get_violations(self, status: Optional[ComplianceStatus] = None, severity: Optional[RuleSeverity] = None) -> List[ComplianceViolation]:
        """Get compliance violations"""
        violations = list(self.violations.values())
        
        if status:
            violations = [v for v in violations if v.status == status]
        
        if severity:
            violations = [v for v in violations if v.severity == severity]
        
        violations.sort(key=lambda x: x.detectedAt, reverse=True)
        return violations
    
    async def get_violation(self, violation_id: str) -> ComplianceViolation:
        """Get specific violation"""
        if violation_id not in self.violations:
            raise HTTPException(status_code=404, detail="Violation not found")
        return self.violations[violation_id]
    
    async def create_violation(self, violation_data: ViolationRequest) -> ComplianceViolation:
        """Create new violation"""
        if violation_data.ruleId not in self.rules:
            raise HTTPException(status_code=404, detail="Rule not found")
        
        violation_id = f"violation-{int(datetime.now().timestamp())}"
        rule = self.rules[violation_data.ruleId]
        
        violation = ComplianceViolation(
            id=violation_id,
            ruleId=violation_data.ruleId,
            rule=rule,
            portfolioId=violation_data.portfolioId,
            clientId=violation_data.clientId,
            description=violation_data.description,
            severity=rule.severity,
            status=ComplianceStatus.PENDING,
            detectedAt=datetime.now(),
            currentValue=violation_data.currentValue,
            thresholdValue=rule.threshold or 0,
            impact=violation_data.impact,
            actionRequired=violation_data.actionRequired,
            assignedTo=violation_data.assignedTo
        )
        
        self.violations[violation_id] = violation
        return violation
    
    async def resolve_violation(self, violation_id: str, resolution_notes: str) -> ComplianceViolation:
        """Resolve a violation"""
        if violation_id not in self.violations:
            raise HTTPException(status_code=404, detail="Violation not found")
        
        violation = self.violations[violation_id]
        violation.status = ComplianceStatus.RESOLVED
        violation.resolvedAt = datetime.now()
        violation.notes = resolution_notes
        
        return violation
    
    async def get_regulatory_reports(self, framework: Optional[RegulatoryFramework] = None, report_type: Optional[ReportType] = None) -> List[RegulatoryReport]:
        """Get regulatory reports"""
        reports = list(self.reports.values())
        
        if framework:
            reports = [r for r in reports if r.framework == framework]
        
        if report_type:
            reports = [r for r in reports if r.type == report_type]
        
        reports.sort(key=lambda x: x.generatedAt, reverse=True)
        return reports
    
    async def get_audit_trail(self, limit: int = 100, user_id: Optional[str] = None, action: Optional[str] = None) -> List[AuditTrail]:
        """Get audit trail"""
        trails = list(self.audit_trails.values())
        
        if user_id:
            trails = [t for t in trails if t.userId == user_id]
        
        if action:
            trails = [t for t in trails if t.action == action]
        
        trails.sort(key=lambda x: x.timestamp, reverse=True)
        return trails[:limit]
    
    async def log_audit_event(self, user_id: str, action: str, resource: str, resource_id: str, details: Dict[str, Any]) -> AuditTrail:
        """Log audit event"""
        audit_id = f"audit-{int(datetime.now().timestamp())}"
        audit = AuditTrail(
            id=audit_id,
            timestamp=datetime.now(),
            userId=user_id,
            userName="User Name",  # In real app, get from user service
            action=action,
            resource=resource,
            resourceId=resource_id,
            details=details,
            ipAddress="192.168.1.100",  # In real app, get from request
            userAgent="User Agent",  # In real app, get from request
            sessionId="session-id",  # In real app, get from session
            outcome="success",
            riskLevel="medium"
        )
        
        self.audit_trails[audit_id] = audit
        return audit
    
    async def get_compliance_metrics(self, period: str = "30d") -> ComplianceMetrics:
        """Get compliance metrics"""
        total_rules = len(self.rules)
        active_rules = len([r for r in self.rules.values() if r.isActive])
        violations = list(self.violations.values())
        
        return ComplianceMetrics(
            date=datetime.now(),
            totalRules=total_rules,
            activeRules=active_rules,
            complianceScore=94.5,
            violationsCount=len(violations),
            criticalViolations=len([v for v in violations if v.severity == RuleSeverity.CRITICAL]),
            highViolations=len([v for v in violations if v.severity == RuleSeverity.HIGH]),
            mediumViolations=len([v for v in violations if v.severity == RuleSeverity.MEDIUM]),
            lowViolations=len([v for v in violations if v.severity == RuleSeverity.LOW]),
            resolvedViolations=len([v for v in violations if v.status == ComplianceStatus.RESOLVED]),
            avgResolutionTime=24.5,
            overdueReports=len([r for r in self.reports.values() if r.isOverdue]),
            completedAudits=5,
            riskScore=6.8,
            trendsData={
                "violations": [3, 2, 4, 1, 2, 3, 2],
                "compliance": [95, 94, 92, 96, 94, 93, 95],
                "risk": [6.5, 6.8, 7.2, 6.9, 6.7, 6.8, 6.8]
            }
        )
    
    async def run_compliance_check(self, rule_id: str, portfolio_id: Optional[str] = None) -> Dict[str, Any]:
        """Run compliance check for a specific rule"""
        if rule_id not in self.rules:
            raise HTTPException(status_code=404, detail="Rule not found")
        
        rule = self.rules[rule_id]
        
        # Mock compliance check logic
        check_result = {
            "ruleId": rule_id,
            "ruleName": rule.name,
            "portfolioId": portfolio_id,
            "checkTime": datetime.now(),
            "status": "compliant",
            "currentValue": 4.8,
            "threshold": rule.threshold,
            "passed": True,
            "details": "All positions within concentration limits"
        }
        
        # Update rule last checked time
        rule.lastChecked = datetime.now()
        
        return check_result

# Initialize service
regulatory_service = RegulatoryService()

# API Endpoints
@router.get("/overview")
async def get_compliance_overview():
    """Get compliance dashboard overview"""
    return await regulatory_service.get_compliance_overview()

@router.get("/rules", response_model=List[ComplianceRule])
async def get_compliance_rules(
    framework: Optional[RegulatoryFramework] = Query(None),
    active_only: bool = Query(True)
):
    """Get compliance rules"""
    return await regulatory_service.get_compliance_rules(framework, active_only)

@router.get("/rules/{rule_id}", response_model=ComplianceRule)
async def get_compliance_rule(rule_id: str):
    """Get specific compliance rule"""
    return await regulatory_service.get_compliance_rule(rule_id)

@router.post("/rules", response_model=ComplianceRule)
async def create_compliance_rule(rule_data: RuleRequest):
    """Create new compliance rule"""
    return await regulatory_service.create_compliance_rule(rule_data)

@router.get("/violations", response_model=List[ComplianceViolation])
async def get_violations(
    status: Optional[ComplianceStatus] = Query(None),
    severity: Optional[RuleSeverity] = Query(None)
):
    """Get compliance violations"""
    return await regulatory_service.get_violations(status, severity)

@router.get("/violations/{violation_id}", response_model=ComplianceViolation)
async def get_violation(violation_id: str):
    """Get specific violation"""
    return await regulatory_service.get_violation(violation_id)

@router.post("/violations", response_model=ComplianceViolation)
async def create_violation(violation_data: ViolationRequest):
    """Create new violation"""
    return await regulatory_service.create_violation(violation_data)

@router.put("/violations/{violation_id}/resolve")
async def resolve_violation(
    violation_id: str,
    resolution_notes: str = Query(..., description="Resolution notes")
):
    """Resolve a violation"""
    violation = await regulatory_service.resolve_violation(violation_id, resolution_notes)
    return {"message": "Violation resolved successfully", "violation": violation}

@router.get("/reports", response_model=List[RegulatoryReport])
async def get_regulatory_reports(
    framework: Optional[RegulatoryFramework] = Query(None),
    report_type: Optional[ReportType] = Query(None)
):
    """Get regulatory reports"""
    return await regulatory_service.get_regulatory_reports(framework, report_type)

@router.get("/audit-trail", response_model=List[AuditTrail])
async def get_audit_trail(
    limit: int = Query(100, ge=1, le=1000),
    user_id: Optional[str] = Query(None),
    action: Optional[str] = Query(None)
):
    """Get audit trail"""
    return await regulatory_service.get_audit_trail(limit, user_id, action)

@router.post("/audit-trail")
async def log_audit_event(
    user_id: str,
    action: str,
    resource: str,
    resource_id: str,
    details: Dict[str, Any]
):
    """Log audit event"""
    audit = await regulatory_service.log_audit_event(user_id, action, resource, resource_id, details)
    return {"message": "Audit event logged", "auditId": audit.id}

@router.get("/metrics", response_model=ComplianceMetrics)
async def get_compliance_metrics(
    period: str = Query("30d", regex="^(7d|30d|90d|1y)$")
):
    """Get compliance metrics"""
    return await regulatory_service.get_compliance_metrics(period)

@router.post("/check/{rule_id}")
async def run_compliance_check(
    rule_id: str,
    portfolio_id: Optional[str] = Query(None)
):
    """Run compliance check"""
    return await regulatory_service.run_compliance_check(rule_id, portfolio_id)

@router.get("/frameworks")
async def get_regulatory_frameworks():
    """Get available regulatory frameworks"""
    return {
        "frameworks": [
            {"id": framework.value, "name": framework.value.upper(), "description": f"{framework.value.upper()} regulatory framework"}
            for framework in RegulatoryFramework
        ]
    }

@router.get("/stats")
async def get_regulatory_stats():
    """Get regulatory compliance statistics"""
    return {
        "totalRules": len(regulatory_service.rules),
        "totalViolations": len(regulatory_service.violations),
        "totalReports": len(regulatory_service.reports),
        "complianceScore": 94.5,
        "riskScore": 6.8,
        "auditEvents": len(regulatory_service.audit_trails)
    }