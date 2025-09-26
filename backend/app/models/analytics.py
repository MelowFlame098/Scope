"""Analytics-related database models.

This module contains models for:
- Financial analysis and metrics
- Performance tracking and reporting
- Risk analysis and calculations
- AI-powered insights and recommendations
"""

from typing import Optional, Dict, Any, List
from decimal import Decimal
from sqlalchemy import (
    Column, String, Boolean, DateTime, Text, Integer,
    ForeignKey, Enum as SQLEnum, UniqueConstraint, Index,
    Numeric, CheckConstraint, Date
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import relationship, validates
from datetime import datetime, date
import uuid
from enum import Enum

from app.models.base import BaseModel, TimestampMixin


class AnalysisType(str, Enum):
    """Analysis types."""
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"
    RISK = "risk"
    PERFORMANCE = "performance"
    CORRELATION = "correlation"
    VOLATILITY = "volatility"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    PORTFOLIO_OPTIMIZATION = "portfolio_optimization"


class MetricType(str, Enum):
    """Metric types."""
    RETURN = "return"
    RISK = "risk"
    RATIO = "ratio"
    INDICATOR = "indicator"
    SCORE = "score"
    PERCENTAGE = "percentage"
    CURRENCY = "currency"
    COUNT = "count"
    DURATION = "duration"


class ReportType(str, Enum):
    """Report types."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    CUSTOM = "custom"
    REAL_TIME = "real_time"


class AlertType(str, Enum):
    """Alert types."""
    PRICE = "price"
    VOLUME = "volume"
    TECHNICAL_INDICATOR = "technical_indicator"
    NEWS = "news"
    EARNINGS = "earnings"
    RISK = "risk"
    PERFORMANCE = "performance"
    PORTFOLIO = "portfolio"
    MARKET = "market"


class AlertStatus(str, Enum):
    """Alert status."""
    ACTIVE = "active"
    TRIGGERED = "triggered"
    ACKNOWLEDGED = "acknowledged"
    DISMISSED = "dismissed"
    EXPIRED = "expired"
    DISABLED = "disabled"


class InsightType(str, Enum):
    """AI insight types."""
    RECOMMENDATION = "recommendation"
    WARNING = "warning"
    OPPORTUNITY = "opportunity"
    TREND = "trend"
    ANOMALY = "anomaly"
    PREDICTION = "prediction"
    OPTIMIZATION = "optimization"


class ConfidenceLevel(str, Enum):
    """Confidence levels for AI insights."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class FinancialMetric(BaseModel):
    """Financial metric model."""
    
    __tablename__ = "financial_metrics"
    
    # Foreign keys
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,
        doc="Reference to user (null for system metrics)"
    )
    
    portfolio_id = Column(
        UUID(as_uuid=True),
        ForeignKey("portfolios.id", ondelete="CASCADE"),
        nullable=True,
        doc="Reference to portfolio"
    )
    
    security_id = Column(
        UUID(as_uuid=True),
        ForeignKey("securities.id", ondelete="CASCADE"),
        nullable=True,
        doc="Reference to security"
    )
    
    # Metric information
    metric_name = Column(
        String(100),
        nullable=False,
        doc="Metric name"
    )
    
    metric_type = Column(
        SQLEnum(MetricType),
        nullable=False,
        doc="Metric type"
    )
    
    category = Column(
        String(50),
        nullable=True,
        doc="Metric category"
    )
    
    # Value and metadata
    value = Column(
        Numeric(20, 8),
        nullable=False,
        doc="Metric value"
    )
    
    unit = Column(
        String(20),
        nullable=True,
        doc="Unit of measurement"
    )
    
    # Time period
    period_start = Column(
        Date,
        nullable=True,
        doc="Period start date"
    )
    
    period_end = Column(
        Date,
        nullable=True,
        doc="Period end date"
    )
    
    calculation_date = Column(
        Date,
        nullable=False,
        default=date.today,
        doc="Calculation date"
    )
    
    # Additional data
    benchmark_value = Column(
        Numeric(20, 8),
        nullable=True,
        doc="Benchmark comparison value"
    )
    
    percentile_rank = Column(
        Numeric(5, 2),
        nullable=True,
        doc="Percentile rank"
    )
    
    # Metadata
    calculation_method = Column(
        String(100),
        nullable=True,
        doc="Calculation methodology"
    )
    
    data_source = Column(
        String(50),
        nullable=True,
        doc="Data source"
    )
    
    metadata = Column(
        JSONB,
        nullable=True,
        doc="Additional metric metadata"
    )
    
    # Relationships
    user = relationship("User")
    portfolio = relationship("Portfolio")
    security = relationship("Security")
    
    # Constraints
    __table_args__ = (
        Index('idx_financial_metric_user_name_date', 'user_id', 'metric_name', 'calculation_date'),
        Index('idx_financial_metric_portfolio_name_date', 'portfolio_id', 'metric_name', 'calculation_date'),
        Index('idx_financial_metric_security_name_date', 'security_id', 'metric_name', 'calculation_date'),
        Index('idx_financial_metric_type_category', 'metric_type', 'category'),
    )
    
    def __repr__(self) -> str:
        """String representation.
        
        Returns:
            String representation
        """
        return (
            f"<FinancialMetric(id={self.id}, name={self.metric_name}, "
            f"value={self.value}, date={self.calculation_date})>"
        )


class PerformanceReport(BaseModel):
    """Performance report model."""
    
    __tablename__ = "performance_reports"
    
    # Foreign keys
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        doc="Reference to user"
    )
    
    portfolio_id = Column(
        UUID(as_uuid=True),
        ForeignKey("portfolios.id", ondelete="CASCADE"),
        nullable=True,
        doc="Reference to portfolio"
    )
    
    # Report information
    report_name = Column(
        String(200),
        nullable=False,
        doc="Report name"
    )
    
    report_type = Column(
        SQLEnum(ReportType),
        nullable=False,
        doc="Report type"
    )
    
    # Time period
    period_start = Column(
        Date,
        nullable=False,
        doc="Report period start"
    )
    
    period_end = Column(
        Date,
        nullable=False,
        doc="Report period end"
    )
    
    generated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        doc="Report generation time"
    )
    
    # Summary metrics
    total_return = Column(
        Numeric(20, 8),
        nullable=True,
        doc="Total return for period"
    )
    
    total_return_percentage = Column(
        Numeric(10, 4),
        nullable=True,
        doc="Total return percentage"
    )
    
    benchmark_return = Column(
        Numeric(10, 4),
        nullable=True,
        doc="Benchmark return percentage"
    )
    
    alpha = Column(
        Numeric(10, 4),
        nullable=True,
        doc="Alpha vs benchmark"
    )
    
    beta = Column(
        Numeric(10, 4),
        nullable=True,
        doc="Beta vs benchmark"
    )
    
    sharpe_ratio = Column(
        Numeric(10, 4),
        nullable=True,
        doc="Sharpe ratio"
    )
    
    volatility = Column(
        Numeric(10, 4),
        nullable=True,
        doc="Portfolio volatility"
    )
    
    max_drawdown = Column(
        Numeric(10, 4),
        nullable=True,
        doc="Maximum drawdown"
    )
    
    # Detailed data
    performance_data = Column(
        JSONB,
        nullable=True,
        doc="Detailed performance data"
    )
    
    risk_metrics = Column(
        JSONB,
        nullable=True,
        doc="Risk analysis metrics"
    )
    
    attribution_analysis = Column(
        JSONB,
        nullable=True,
        doc="Performance attribution analysis"
    )
    
    # Report content
    summary = Column(
        Text,
        nullable=True,
        doc="Report summary"
    )
    
    insights = Column(
        JSONB,
        nullable=True,
        doc="Key insights and findings"
    )
    
    recommendations = Column(
        JSONB,
        nullable=True,
        doc="Recommendations"
    )
    
    # Status
    is_published = Column(
        Boolean,
        default=False,
        nullable=False,
        doc="Report is published"
    )
    
    # Metadata
    metadata = Column(
        JSONB,
        nullable=True,
        doc="Additional report metadata"
    )
    
    # Relationships
    user = relationship("User")
    portfolio = relationship("Portfolio")
    
    # Constraints
    __table_args__ = (
        Index('idx_performance_report_user_type_period', 'user_id', 'report_type', 'period_end'),
        Index('idx_performance_report_portfolio_period', 'portfolio_id', 'period_end'),
        Index('idx_performance_report_generated', 'generated_at'),
        CheckConstraint('period_end >= period_start', name='ck_performance_report_period_valid'),
    )
    
    def __repr__(self) -> str:
        """String representation.
        
        Returns:
            String representation
        """
        return (
            f"<PerformanceReport(id={self.id}, name={self.report_name}, "
            f"type={self.report_type.value}, period={self.period_start} to {self.period_end})>"
        )


class RiskAnalysis(BaseModel):
    """Risk analysis model."""
    
    __tablename__ = "risk_analyses"
    
    # Foreign keys
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        doc="Reference to user"
    )
    
    portfolio_id = Column(
        UUID(as_uuid=True),
        ForeignKey("portfolios.id", ondelete="CASCADE"),
        nullable=True,
        doc="Reference to portfolio"
    )
    
    # Analysis information
    analysis_name = Column(
        String(200),
        nullable=False,
        doc="Analysis name"
    )
    
    analysis_type = Column(
        SQLEnum(AnalysisType),
        nullable=False,
        doc="Analysis type"
    )
    
    # Time period
    analysis_date = Column(
        Date,
        nullable=False,
        default=date.today,
        doc="Analysis date"
    )
    
    lookback_period = Column(
        Integer,
        nullable=True,
        doc="Lookback period in days"
    )
    
    # Risk metrics
    value_at_risk_95 = Column(
        Numeric(20, 8),
        nullable=True,
        doc="Value at Risk (95% confidence)"
    )
    
    value_at_risk_99 = Column(
        Numeric(20, 8),
        nullable=True,
        doc="Value at Risk (99% confidence)"
    )
    
    expected_shortfall = Column(
        Numeric(20, 8),
        nullable=True,
        doc="Expected Shortfall (Conditional VaR)"
    )
    
    volatility = Column(
        Numeric(10, 4),
        nullable=True,
        doc="Portfolio volatility"
    )
    
    skewness = Column(
        Numeric(10, 4),
        nullable=True,
        doc="Return distribution skewness"
    )
    
    kurtosis = Column(
        Numeric(10, 4),
        nullable=True,
        doc="Return distribution kurtosis"
    )
    
    # Concentration risk
    concentration_risk = Column(
        Numeric(10, 4),
        nullable=True,
        doc="Concentration risk score"
    )
    
    largest_position_weight = Column(
        Numeric(10, 4),
        nullable=True,
        doc="Largest position weight"
    )
    
    top_10_concentration = Column(
        Numeric(10, 4),
        nullable=True,
        doc="Top 10 positions concentration"
    )
    
    # Correlation analysis
    correlation_data = Column(
        JSONB,
        nullable=True,
        doc="Correlation matrix and analysis"
    )
    
    # Stress testing
    stress_test_results = Column(
        JSONB,
        nullable=True,
        doc="Stress test scenarios and results"
    )
    
    # Risk decomposition
    risk_attribution = Column(
        JSONB,
        nullable=True,
        doc="Risk attribution by asset/sector"
    )
    
    # Analysis results
    risk_score = Column(
        Numeric(5, 2),
        nullable=True,
        doc="Overall risk score (1-10)"
    )
    
    risk_level = Column(
        String(20),
        nullable=True,
        doc="Risk level (low, medium, high)"
    )
    
    findings = Column(
        JSONB,
        nullable=True,
        doc="Key findings and insights"
    )
    
    recommendations = Column(
        JSONB,
        nullable=True,
        doc="Risk management recommendations"
    )
    
    # Metadata
    methodology = Column(
        String(100),
        nullable=True,
        doc="Analysis methodology"
    )
    
    data_source = Column(
        String(50),
        nullable=True,
        doc="Data source"
    )
    
    metadata = Column(
        JSONB,
        nullable=True,
        doc="Additional analysis metadata"
    )
    
    # Relationships
    user = relationship("User")
    portfolio = relationship("Portfolio")
    
    # Constraints
    __table_args__ = (
        Index('idx_risk_analysis_user_type_date', 'user_id', 'analysis_type', 'analysis_date'),
        Index('idx_risk_analysis_portfolio_date', 'portfolio_id', 'analysis_date'),
        Index('idx_risk_analysis_risk_score', 'risk_score'),
    )
    
    def __repr__(self) -> str:
        """String representation.
        
        Returns:
            String representation
        """
        return (
            f"<RiskAnalysis(id={self.id}, name={self.analysis_name}, "
            f"type={self.analysis_type.value}, date={self.analysis_date})>"
        )


class Alert(BaseModel):
    """Alert model for notifications and warnings."""
    
    __tablename__ = "alerts"
    
    # Foreign keys
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        doc="Reference to user"
    )
    
    portfolio_id = Column(
        UUID(as_uuid=True),
        ForeignKey("portfolios.id", ondelete="CASCADE"),
        nullable=True,
        doc="Reference to portfolio"
    )
    
    security_id = Column(
        UUID(as_uuid=True),
        ForeignKey("securities.id", ondelete="CASCADE"),
        nullable=True,
        doc="Reference to security"
    )
    
    # Alert information
    alert_type = Column(
        SQLEnum(AlertType),
        nullable=False,
        doc="Alert type"
    )
    
    title = Column(
        String(200),
        nullable=False,
        doc="Alert title"
    )
    
    message = Column(
        Text,
        nullable=False,
        doc="Alert message"
    )
    
    # Trigger conditions
    trigger_conditions = Column(
        JSONB,
        nullable=True,
        doc="Alert trigger conditions"
    )
    
    trigger_value = Column(
        Numeric(20, 8),
        nullable=True,
        doc="Trigger threshold value"
    )
    
    current_value = Column(
        Numeric(20, 8),
        nullable=True,
        doc="Current value that triggered alert"
    )
    
    # Status and timing
    status = Column(
        SQLEnum(AlertStatus),
        default=AlertStatus.ACTIVE,
        nullable=False,
        doc="Alert status"
    )
    
    priority = Column(
        String(20),
        default="medium",
        nullable=False,
        doc="Alert priority"
    )
    
    triggered_at = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Alert trigger time"
    )
    
    acknowledged_at = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Alert acknowledgment time"
    )
    
    expires_at = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Alert expiration time"
    )
    
    # Notification settings
    notification_sent = Column(
        Boolean,
        default=False,
        nullable=False,
        doc="Notification has been sent"
    )
    
    notification_channels = Column(
        ARRAY(String),
        nullable=True,
        doc="Notification channels used"
    )
    
    # Metadata
    metadata = Column(
        JSONB,
        nullable=True,
        doc="Additional alert metadata"
    )
    
    # Relationships
    user = relationship("User")
    portfolio = relationship("Portfolio")
    security = relationship("Security")
    
    # Constraints
    __table_args__ = (
        Index('idx_alert_user_status', 'user_id', 'status'),
        Index('idx_alert_type_status', 'alert_type', 'status'),
        Index('idx_alert_triggered', 'triggered_at'),
        Index('idx_alert_priority', 'priority'),
    )
    
    @property
    def is_active(self) -> bool:
        """Check if alert is active.
        
        Returns:
            True if alert is active
        """
        return self.status == AlertStatus.ACTIVE
    
    @property
    def is_expired(self) -> bool:
        """Check if alert is expired.
        
        Returns:
            True if alert is expired
        """
        return (
            self.expires_at is not None and
            self.expires_at < datetime.utcnow()
        )
    
    def trigger(self, current_value: Optional[Decimal] = None) -> None:
        """Trigger the alert.
        
        Args:
            current_value: Current value that triggered the alert
        """
        self.status = AlertStatus.TRIGGERED
        self.triggered_at = datetime.utcnow()
        if current_value is not None:
            self.current_value = current_value
    
    def acknowledge(self) -> None:
        """Acknowledge the alert."""
        self.status = AlertStatus.ACKNOWLEDGED
        self.acknowledged_at = datetime.utcnow()
    
    def dismiss(self) -> None:
        """Dismiss the alert."""
        self.status = AlertStatus.DISMISSED
    
    def __repr__(self) -> str:
        """String representation.
        
        Returns:
            String representation
        """
        return (
            f"<Alert(id={self.id}, type={self.alert_type.value}, "
            f"title={self.title}, status={self.status.value})>"
        )


class AIInsight(BaseModel):
    """AI-generated insight model."""
    
    __tablename__ = "ai_insights"
    
    # Foreign keys
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        doc="Reference to user"
    )
    
    portfolio_id = Column(
        UUID(as_uuid=True),
        ForeignKey("portfolios.id", ondelete="CASCADE"),
        nullable=True,
        doc="Reference to portfolio"
    )
    
    security_id = Column(
        UUID(as_uuid=True),
        ForeignKey("securities.id", ondelete="CASCADE"),
        nullable=True,
        doc="Reference to security"
    )
    
    # Insight information
    insight_type = Column(
        SQLEnum(InsightType),
        nullable=False,
        doc="Insight type"
    )
    
    title = Column(
        String(200),
        nullable=False,
        doc="Insight title"
    )
    
    description = Column(
        Text,
        nullable=False,
        doc="Insight description"
    )
    
    # AI model information
    model_name = Column(
        String(100),
        nullable=True,
        doc="AI model used"
    )
    
    model_version = Column(
        String(50),
        nullable=True,
        doc="Model version"
    )
    
    confidence_level = Column(
        SQLEnum(ConfidenceLevel),
        nullable=False,
        doc="Confidence level"
    )
    
    confidence_score = Column(
        Numeric(5, 4),
        nullable=True,
        doc="Confidence score (0-1)"
    )
    
    # Insight data
    key_metrics = Column(
        JSONB,
        nullable=True,
        doc="Key metrics supporting the insight"
    )
    
    supporting_data = Column(
        JSONB,
        nullable=True,
        doc="Supporting data and analysis"
    )
    
    recommendations = Column(
        JSONB,
        nullable=True,
        doc="Actionable recommendations"
    )
    
    # Prediction data (if applicable)
    prediction_horizon = Column(
        Integer,
        nullable=True,
        doc="Prediction horizon in days"
    )
    
    predicted_value = Column(
        Numeric(20, 8),
        nullable=True,
        doc="Predicted value"
    )
    
    prediction_range = Column(
        JSONB,
        nullable=True,
        doc="Prediction confidence intervals"
    )
    
    # Status and feedback
    is_active = Column(
        Boolean,
        default=True,
        nullable=False,
        doc="Insight is active"
    )
    
    user_feedback = Column(
        String(20),
        nullable=True,
        doc="User feedback (helpful, not_helpful, etc.)"
    )
    
    feedback_notes = Column(
        Text,
        nullable=True,
        doc="User feedback notes"
    )
    
    # Timing
    generated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        doc="Insight generation time"
    )
    
    expires_at = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Insight expiration time"
    )
    
    viewed_at = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="First view time"
    )
    
    # Metadata
    tags = Column(
        ARRAY(String),
        nullable=True,
        doc="Insight tags"
    )
    
    metadata = Column(
        JSONB,
        nullable=True,
        doc="Additional insight metadata"
    )
    
    # Relationships
    user = relationship("User")
    portfolio = relationship("Portfolio")
    security = relationship("Security")
    
    # Constraints
    __table_args__ = (
        Index('idx_ai_insight_user_type', 'user_id', 'insight_type'),
        Index('idx_ai_insight_portfolio_type', 'portfolio_id', 'insight_type'),
        Index('idx_ai_insight_security_type', 'security_id', 'insight_type'),
        Index('idx_ai_insight_confidence', 'confidence_level'),
        Index('idx_ai_insight_generated', 'generated_at'),
        Index('idx_ai_insight_active', 'is_active'),
    )
    
    @property
    def is_expired(self) -> bool:
        """Check if insight is expired.
        
        Returns:
            True if insight is expired
        """
        return (
            self.expires_at is not None and
            self.expires_at < datetime.utcnow()
        )
    
    @property
    def is_viewed(self) -> bool:
        """Check if insight has been viewed.
        
        Returns:
            True if insight has been viewed
        """
        return self.viewed_at is not None
    
    def mark_viewed(self) -> None:
        """Mark insight as viewed."""
        if not self.viewed_at:
            self.viewed_at = datetime.utcnow()
    
    def set_feedback(self, feedback: str, notes: Optional[str] = None) -> None:
        """Set user feedback.
        
        Args:
            feedback: Feedback type
            notes: Optional feedback notes
        """
        self.user_feedback = feedback
        self.feedback_notes = notes
    
    def __repr__(self) -> str:
        """String representation.
        
        Returns:
            String representation
        """
        return (
            f"<AIInsight(id={self.id}, type={self.insight_type.value}, "
            f"title={self.title}, confidence={self.confidence_level.value})>"
        )