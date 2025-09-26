from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from database import get_db
from auth import get_current_user
import json

router = APIRouter(prefix="/compliance", tags=["compliance"])

# Compliance models
class ComplianceCheckRequest(BaseModel):
    check_type: str  # trade, portfolio, risk, regulatory
    entity_id: str
    parameters: Dict[str, Any]

class PolicyRequest(BaseModel):
    policy_type: str
    name: str
    description: str
    rules: List[Dict[str, Any]]
    enforcement_level: str  # warning, block, audit

class ComplianceRule(BaseModel):
    id: str
    name: str
    description: str
    rule_type: str  # position_limit, concentration, sector_limit, risk_limit
    parameters: Dict[str, Any]
    enforcement_level: str
    enabled: bool
    created_at: datetime
    updated_at: datetime

class ComplianceViolation(BaseModel):
    id: str
    rule_id: str
    rule_name: str
    entity_id: str
    entity_type: str  # portfolio, trade, user
    violation_type: str
    severity: str  # low, medium, high, critical
    description: str
    detected_at: datetime
    status: str  # open, investigating, resolved, false_positive
    resolution_notes: Optional[str] = None
    resolved_at: Optional[datetime] = None

class ComplianceCheck(BaseModel):
    id: str
    check_type: str
    entity_id: str
    status: str  # passed, failed, warning
    violations: List[ComplianceViolation]
    checked_at: datetime
    details: Dict[str, Any]

class CompliancePolicy(BaseModel):
    id: str
    name: str
    description: str
    policy_type: str
    rules: List[ComplianceRule]
    enforcement_level: str
    enabled: bool
    created_at: datetime
    updated_at: datetime

class ComplianceReport(BaseModel):
    id: str
    report_type: str  # daily, weekly, monthly, quarterly
    period_start: datetime
    period_end: datetime
    total_checks: int
    passed_checks: int
    failed_checks: int
    violations: List[ComplianceViolation]
    compliance_score: float
    generated_at: datetime

class ComplianceMetrics(BaseModel):
    total_rules: int
    active_rules: int
    total_violations: int
    open_violations: int
    compliance_score: float
    last_check: datetime
    checks_today: int
    violations_today: int

class ComplianceEngine:
    def __init__(self):
        self.rules = []
        self.policies = []
        self.violations = []
        self.checks = []
        self.reports = []
        self._initialize_default_rules()
        
    def _initialize_default_rules(self):
        """Initialize default compliance rules"""
        default_rules = [
            {
                "id": "rule_001",
                "name": "Position Concentration Limit",
                "description": "No single position should exceed 10% of portfolio",
                "rule_type": "concentration",
                "parameters": {"max_percentage": 10.0, "scope": "portfolio"},
                "enforcement_level": "block",
                "enabled": True,
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            },
            {
                "id": "rule_002",
                "name": "Sector Concentration Limit",
                "description": "No sector should exceed 25% of portfolio",
                "rule_type": "sector_limit",
                "parameters": {"max_percentage": 25.0, "scope": "sector"},
                "enforcement_level": "warning",
                "enabled": True,
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            },
            {
                "id": "rule_003",
                "name": "Daily Trading Limit",
                "description": "Daily trading volume should not exceed $1M",
                "rule_type": "position_limit",
                "parameters": {"max_amount": 1000000.0, "period": "daily"},
                "enforcement_level": "block",
                "enabled": True,
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            },
            {
                "id": "rule_004",
                "name": "Risk Limit - VaR",
                "description": "Portfolio VaR should not exceed 2%",
                "rule_type": "risk_limit",
                "parameters": {"max_var": 2.0, "confidence_level": 95},
                "enforcement_level": "warning",
                "enabled": True,
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            }
        ]
        
        for rule_data in default_rules:
            rule = ComplianceRule(**rule_data)
            self.rules.append(rule)
    
    def create_rule(self, rule_data: Dict[str, Any]) -> ComplianceRule:
        """Create a new compliance rule"""
        rule = ComplianceRule(
            id=f"rule_{len(self.rules) + 1:03d}",
            name=rule_data["name"],
            description=rule_data["description"],
            rule_type=rule_data["rule_type"],
            parameters=rule_data["parameters"],
            enforcement_level=rule_data.get("enforcement_level", "warning"),
            enabled=rule_data.get("enabled", True),
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self.rules.append(rule)
        return rule
    
    def get_rules(self, rule_type: Optional[str] = None, enabled_only: bool = True) -> List[ComplianceRule]:
        """Get compliance rules with optional filtering"""
        rules = self.rules
        
        if enabled_only:
            rules = [r for r in rules if r.enabled]
        
        if rule_type:
            rules = [r for r in rules if r.rule_type == rule_type]
        
        return rules
    
    def run_compliance_check(self, check_type: str, entity_id: str, 
                           parameters: Dict[str, Any]) -> ComplianceCheck:
        """Run compliance check on an entity"""
        violations = []
        status = "passed"
        
        # Get relevant rules for this check type
        relevant_rules = [r for r in self.rules if r.enabled and 
                         (r.rule_type == check_type or check_type == "all")]
        
        # Mock compliance checking logic
        for rule in relevant_rules:
            violation = self._check_rule_violation(rule, entity_id, parameters)
            if violation:
                violations.append(violation)
                if rule.enforcement_level in ["block", "critical"]:
                    status = "failed"
                elif status == "passed":
                    status = "warning"
        
        check = ComplianceCheck(
            id=f"check_{len(self.checks) + 1}",
            check_type=check_type,
            entity_id=entity_id,
            status=status,
            violations=violations,
            checked_at=datetime.now(),
            details=parameters
        )
        
        self.checks.append(check)
        return check
    
    def _check_rule_violation(self, rule: ComplianceRule, entity_id: str, 
                            parameters: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Check if a specific rule is violated"""
        # Mock violation detection logic
        violation_detected = False
        violation_description = ""
        
        if rule.rule_type == "concentration":
            position_percentage = parameters.get("position_percentage", 0)
            max_percentage = rule.parameters.get("max_percentage", 10)
            if position_percentage > max_percentage:
                violation_detected = True
                violation_description = f"Position concentration {position_percentage}% exceeds limit {max_percentage}%"
        
        elif rule.rule_type == "sector_limit":
            sector_percentage = parameters.get("sector_percentage", 0)
            max_percentage = rule.parameters.get("max_percentage", 25)
            if sector_percentage > max_percentage:
                violation_detected = True
                violation_description = f"Sector concentration {sector_percentage}% exceeds limit {max_percentage}%"
        
        elif rule.rule_type == "position_limit":
            trade_amount = parameters.get("trade_amount", 0)
            max_amount = rule.parameters.get("max_amount", 1000000)
            if trade_amount > max_amount:
                violation_detected = True
                violation_description = f"Trade amount ${trade_amount:,.2f} exceeds daily limit ${max_amount:,.2f}"
        
        elif rule.rule_type == "risk_limit":
            var_value = parameters.get("var", 0)
            max_var = rule.parameters.get("max_var", 2.0)
            if var_value > max_var:
                violation_detected = True
                violation_description = f"Portfolio VaR {var_value}% exceeds limit {max_var}%"
        
        if violation_detected:
            severity = "high" if rule.enforcement_level == "block" else "medium"
            
            violation = ComplianceViolation(
                id=f"violation_{len(self.violations) + 1}",
                rule_id=rule.id,
                rule_name=rule.name,
                entity_id=entity_id,
                entity_type=parameters.get("entity_type", "unknown"),
                violation_type=rule.rule_type,
                severity=severity,
                description=violation_description,
                detected_at=datetime.now(),
                status="open"
            )
            
            self.violations.append(violation)
            return violation
        
        return None
    
    def get_violations(self, status: Optional[str] = None, 
                      severity: Optional[str] = None) -> List[ComplianceViolation]:
        """Get violations with optional filtering"""
        violations = self.violations
        
        if status:
            violations = [v for v in violations if v.status == status]
        
        if severity:
            violations = [v for v in violations if v.severity == severity]
        
        return sorted(violations, key=lambda x: x.detected_at, reverse=True)
    
    def resolve_violation(self, violation_id: str, resolution_notes: str) -> bool:
        """Resolve a compliance violation"""
        for violation in self.violations:
            if violation.id == violation_id:
                violation.status = "resolved"
                violation.resolution_notes = resolution_notes
                violation.resolved_at = datetime.now()
                return True
        return False
    
    def create_policy(self, policy_data: Dict[str, Any]) -> CompliancePolicy:
        """Create a compliance policy"""
        policy = CompliancePolicy(
            id=f"policy_{len(self.policies) + 1}",
            name=policy_data["name"],
            description=policy_data["description"],
            policy_type=policy_data["policy_type"],
            rules=[],  # Rules would be associated separately
            enforcement_level=policy_data.get("enforcement_level", "warning"),
            enabled=policy_data.get("enabled", True),
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self.policies.append(policy)
        return policy
    
    def generate_compliance_report(self, report_type: str, 
                                 period_start: datetime, 
                                 period_end: datetime) -> ComplianceReport:
        """Generate a compliance report"""
        # Filter checks and violations for the period
        period_checks = [
            c for c in self.checks 
            if period_start <= c.checked_at <= period_end
        ]
        
        period_violations = [
            v for v in self.violations 
            if period_start <= v.detected_at <= period_end
        ]
        
        total_checks = len(period_checks)
        passed_checks = len([c for c in period_checks if c.status == "passed"])
        failed_checks = total_checks - passed_checks
        
        compliance_score = (passed_checks / total_checks * 100) if total_checks > 0 else 100
        
        report = ComplianceReport(
            id=f"report_{len(self.reports) + 1}",
            report_type=report_type,
            period_start=period_start,
            period_end=period_end,
            total_checks=total_checks,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            violations=period_violations,
            compliance_score=compliance_score,
            generated_at=datetime.now()
        )
        
        self.reports.append(report)
        return report
    
    def get_compliance_metrics(self) -> ComplianceMetrics:
        """Get compliance metrics"""
        today = datetime.now().date()
        
        total_rules = len(self.rules)
        active_rules = len([r for r in self.rules if r.enabled])
        total_violations = len(self.violations)
        open_violations = len([v for v in self.violations if v.status == "open"])
        
        checks_today = len([
            c for c in self.checks 
            if c.checked_at.date() == today
        ])
        
        violations_today = len([
            v for v in self.violations 
            if v.detected_at.date() == today
        ])
        
        # Calculate compliance score
        recent_checks = [
            c for c in self.checks 
            if c.checked_at > datetime.now() - timedelta(days=30)
        ]
        
        if recent_checks:
            passed_recent = len([c for c in recent_checks if c.status == "passed"])
            compliance_score = (passed_recent / len(recent_checks)) * 100
        else:
            compliance_score = 100.0
        
        last_check = max([c.checked_at for c in self.checks]) if self.checks else datetime.now()
        
        return ComplianceMetrics(
            total_rules=total_rules,
            active_rules=active_rules,
            total_violations=total_violations,
            open_violations=open_violations,
            compliance_score=compliance_score,
            last_check=last_check,
            checks_today=checks_today,
            violations_today=violations_today
        )

# Initialize service
compliance_engine = ComplianceEngine()

# API Endpoints
@router.post("/check")
async def run_compliance_check(
    request: ComplianceCheckRequest,
    current_user = Depends(get_current_user)
):
    """Run a compliance check"""
    check = compliance_engine.run_compliance_check(
        request.check_type,
        request.entity_id,
        request.parameters
    )
    return check

@router.get("/rules")
async def get_compliance_rules(
    rule_type: Optional[str] = None,
    enabled_only: bool = True,
    current_user = Depends(get_current_user)
):
    """Get compliance rules"""
    rules = compliance_engine.get_rules(rule_type, enabled_only)
    return rules

@router.post("/rules")
async def create_compliance_rule(
    rule_data: Dict[str, Any],
    current_user = Depends(get_current_user)
):
    """Create a new compliance rule"""
    rule = compliance_engine.create_rule(rule_data)
    return rule

@router.get("/violations")
async def get_violations(
    status: Optional[str] = None,
    severity: Optional[str] = None,
    current_user = Depends(get_current_user)
):
    """Get compliance violations"""
    violations = compliance_engine.get_violations(status, severity)
    return violations

@router.put("/violations/{violation_id}/resolve")
async def resolve_violation(
    violation_id: str,
    resolution_data: Dict[str, str],
    current_user = Depends(get_current_user)
):
    """Resolve a compliance violation"""
    success = compliance_engine.resolve_violation(
        violation_id,
        resolution_data.get("resolution_notes", "")
    )
    
    if success:
        return {"status": "resolved", "violation_id": violation_id}
    
    raise HTTPException(status_code=404, detail="Violation not found")

@router.post("/policies")
async def create_policy(
    request: PolicyRequest,
    current_user = Depends(get_current_user)
):
    """Create a compliance policy"""
    policy = compliance_engine.create_policy(request.dict())
    return policy

@router.post("/reports")
async def generate_report(
    report_data: Dict[str, Any],
    current_user = Depends(get_current_user)
):
    """Generate a compliance report"""
    report = compliance_engine.generate_compliance_report(
        report_data["report_type"],
        datetime.fromisoformat(report_data["period_start"]),
        datetime.fromisoformat(report_data["period_end"])
    )
    return report

@router.get("/metrics")
async def get_compliance_metrics(current_user = Depends(get_current_user)):
    """Get compliance metrics"""
    metrics = compliance_engine.get_compliance_metrics()
    return metrics

@router.get("/health")
async def compliance_health_check():
    """Compliance engine health check"""
    return {
        "status": "healthy",
        "service": "compliance_engine",
        "timestamp": datetime.now(),
        "features": [
            "rule_management",
            "compliance_checking",
            "violation_tracking",
            "policy_management",
            "compliance_reporting"
        ]
    }

# Export service and router
__all__ = ["ComplianceEngine", "compliance_engine", "router"]