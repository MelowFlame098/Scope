"""Prompt Engineering Engine for FinScope - Phase 6 Implementation

Provides sophisticated prompt templates and generation for financial explanations,
technical analysis interpretation, and market insights.
"""

import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

from jinja2 import Environment, BaseLoader, Template
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class PromptType(str, Enum):
    """Types of prompts for different use cases"""
    TECHNICAL_ANALYSIS = "technical_analysis"
    MARKET_EXPLANATION = "market_explanation"
    MODEL_INTERPRETATION = "model_interpretation"
    RISK_ASSESSMENT = "risk_assessment"
    PORTFOLIO_ANALYSIS = "portfolio_analysis"
    NEWS_ANALYSIS = "news_analysis"
    GENERAL_CHAT = "general_chat"
    TRADING_STRATEGY = "trading_strategy"

class ExplanationComplexity(str, Enum):
    """Explanation complexity levels"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

@dataclass
class PromptTemplate:
    """Prompt template configuration"""
    name: str
    template: str
    description: str
    required_context: List[str]
    optional_context: List[str]
    complexity_variations: Dict[ExplanationComplexity, str]

class PromptEngine:
    """Advanced prompt engineering engine for financial explanations"""
    
    def __init__(self):
        self.jinja_env = Environment(loader=BaseLoader())
        self.templates: Dict[PromptType, PromptTemplate] = {}
        self._initialize_templates()
    
    def _initialize_templates(self):
        """Initialize all prompt templates"""
        
        # Technical Analysis Explanation Template
        self.templates[PromptType.TECHNICAL_ANALYSIS] = PromptTemplate(
            name="Technical Analysis Explanation",
            template="""You are a financial technical analysis expert. Explain the following technical analysis data in clear, {{ complexity_level }} terms.

**Technical Analysis Data:**
{{ content }}

**Market Context:**
{% if symbol %}Symbol: {{ symbol }}{% endif %}
{% if timeframe %}Timeframe: {{ timeframe }}{% endif %}
{% if current_price %}Current Price: ${{ current_price }}{% endif %}
{% if volume %}Volume: {{ volume }}{% endif %}

**Additional Context:**
{% for key, value in context.items() %}
{{ key }}: {{ value }}
{% endfor %}

{% if conversation_context %}
**Previous Conversation:**
{{ conversation_context }}
{% endif %}

**Instructions:**
1. Explain what each technical indicator means
2. Interpret the signals and patterns
3. Discuss potential implications for price movement
4. Mention any important caveats or limitations
{% if include_examples %}5. Provide relevant examples or analogies{% endif %}

**Complexity Level:** {{ complexity_level }}
{% if complexity_level == 'beginner' %}
- Use simple language and avoid jargon
- Explain basic concepts clearly
- Focus on practical implications
{% elif complexity_level == 'expert' %}
- Use technical terminology appropriately
- Include quantitative details
- Discuss advanced concepts and nuances
{% endif %}

Provide a comprehensive yet concise explanation that helps the user understand the technical analysis.""",
            description="Explains technical indicators and chart patterns",
            required_context=["content"],
            optional_context=["symbol", "timeframe", "current_price", "volume"],
            complexity_variations={
                ExplanationComplexity.BEGINNER: "simple and beginner-friendly",
                ExplanationComplexity.INTERMEDIATE: "moderately detailed",
                ExplanationComplexity.ADVANCED: "detailed and comprehensive",
                ExplanationComplexity.EXPERT: "highly technical and precise"
            }
        )
        
        # Market Data Explanation Template
        self.templates[PromptType.MARKET_EXPLANATION] = PromptTemplate(
            name="Market Data Explanation",
            template="""You are a financial market analyst. Explain the following market data and movements in {{ complexity_level }} terms.

**Market Data:**
{{ content }}

**Market Context:**
{% if symbol %}Asset: {{ symbol }}{% endif %}
{% if sector %}Sector: {{ sector }}{% endif %}
{% if market_cap %}Market Cap: {{ market_cap }}{% endif %}
{% if trading_session %}Trading Session: {{ trading_session }}{% endif %}

**Economic Context:**
{% for key, value in context.items() %}
{{ key }}: {{ value }}
{% endfor %}

{% if conversation_context %}
**Previous Discussion:**
{{ conversation_context }}
{% endif %}

**Analysis Framework:**
1. Describe what the data shows
2. Explain the significance of the movements
3. Identify potential causes or catalysts
4. Discuss broader market implications
5. Highlight any unusual patterns or anomalies
{% if include_examples %}6. Provide historical context or similar examples{% endif %}

**Explanation Style:** {{ complexity_level }}
{% if complexity_level == 'beginner' %}
- Use everyday language and analogies
- Explain financial terms when used
- Focus on what it means for investors
{% elif complexity_level == 'expert' %}
- Use precise financial terminology
- Include statistical significance
- Reference relevant financial theories
{% endif %}

Provide clear insights that help understand the market dynamics.""",
            description="Explains market movements and price action",
            required_context=["content"],
            optional_context=["symbol", "sector", "market_cap", "trading_session"],
            complexity_variations={
                ExplanationComplexity.BEGINNER: "accessible and educational",
                ExplanationComplexity.INTERMEDIATE: "balanced and informative",
                ExplanationComplexity.ADVANCED: "thorough and analytical",
                ExplanationComplexity.EXPERT: "sophisticated and precise"
            }
        )
        
        # AI/ML Model Interpretation Template
        self.templates[PromptType.MODEL_INTERPRETATION] = PromptTemplate(
            name="AI/ML Model Interpretation",
            template="""You are an AI/ML expert specializing in financial models. Interpret the following model output in {{ complexity_level }} terms.

**Model Output:**
{{ content }}

**Model Information:**
{% if model_type %}Model Type: {{ model_type }}{% endif %}
{% if confidence %}Confidence: {{ confidence }}%{% endif %}
{% if prediction_horizon %}Prediction Horizon: {{ prediction_horizon }}{% endif %}
{% if features_used %}Key Features: {{ features_used }}{% endif %}

**Context:**
{% for key, value in context.items() %}
{{ key }}: {{ value }}
{% endfor %}

{% if conversation_context %}
**Previous Context:**
{{ conversation_context }}
{% endif %}

**Interpretation Guidelines:**
1. Explain what the model is predicting
2. Interpret the confidence level and reliability
3. Discuss the key factors influencing the prediction
4. Explain any limitations or uncertainties
5. Provide actionable insights
{% if include_examples %}6. Give examples of how to use this information{% endif %}

**Technical Level:** {{ complexity_level }}
{% if complexity_level == 'beginner' %}
- Avoid technical ML jargon
- Focus on practical implications
- Use simple analogies for complex concepts
{% elif complexity_level == 'expert' %}
- Include technical details about the model
- Discuss statistical significance
- Reference model performance metrics
{% endif %}

Help the user understand and act on the model's insights.""",
            description="Interprets AI/ML model predictions and outputs",
            required_context=["content"],
            optional_context=["model_type", "confidence", "prediction_horizon", "features_used"],
            complexity_variations={
                ExplanationComplexity.BEGINNER: "non-technical and practical",
                ExplanationComplexity.INTERMEDIATE: "moderately technical",
                ExplanationComplexity.ADVANCED: "technically detailed",
                ExplanationComplexity.EXPERT: "highly technical and comprehensive"
            }
        )
        
        # Risk Assessment Template
        self.templates[PromptType.RISK_ASSESSMENT] = PromptTemplate(
            name="Risk Assessment Explanation",
            template="""You are a risk management expert. Explain the following risk analysis in {{ complexity_level }} terms.

**Risk Analysis:**
{{ content }}

**Risk Context:**
{% if portfolio_value %}Portfolio Value: ${{ portfolio_value }}{% endif %}
{% if time_horizon %}Time Horizon: {{ time_horizon }}{% endif %}
{% if risk_tolerance %}Risk Tolerance: {{ risk_tolerance }}{% endif %}
{% if benchmark %}Benchmark: {{ benchmark }}{% endif %}

**Additional Information:**
{% for key, value in context.items() %}
{{ key }}: {{ value }}
{% endfor %}

{% if conversation_context %}
**Previous Discussion:**
{{ conversation_context }}
{% endif %}

**Risk Explanation Framework:**
1. Define the types of risk identified
2. Quantify the risk levels where possible
3. Explain the potential impact on investments
4. Discuss risk mitigation strategies
5. Compare to industry standards or benchmarks
{% if include_examples %}6. Provide real-world examples or scenarios{% endif %}

**Communication Style:** {{ complexity_level }}
{% if complexity_level == 'beginner' %}
- Use clear, non-intimidating language
- Focus on practical risk management
- Avoid complex statistical terms
{% elif complexity_level == 'expert' %}
- Use precise risk management terminology
- Include statistical measures (VaR, CVaR, etc.)
- Reference risk management frameworks
{% endif %}

Help the user understand and manage their investment risks effectively.""",
            description="Explains risk metrics and assessments",
            required_context=["content"],
            optional_context=["portfolio_value", "time_horizon", "risk_tolerance", "benchmark"],
            complexity_variations={
                ExplanationComplexity.BEGINNER: "reassuring and educational",
                ExplanationComplexity.INTERMEDIATE: "balanced and informative",
                ExplanationComplexity.ADVANCED: "comprehensive and detailed",
                ExplanationComplexity.EXPERT: "technical and precise"
            }
        )
        
        # General Financial Chat Template
        self.templates[PromptType.GENERAL_CHAT] = PromptTemplate(
            name="General Financial Chat",
            template="""You are a knowledgeable financial advisor having a conversation. Respond to the following question or comment in a {{ complexity_level }} manner.

**User Question/Comment:**
{{ content }}

**User Context:**
{% if user_profile %}User Profile: {{ user_profile }}{% endif %}
{% if investment_experience %}Investment Experience: {{ investment_experience }}{% endif %}
{% if current_holdings %}Current Holdings: {{ current_holdings }}{% endif %}

**Additional Context:**
{% for key, value in context.items() %}
{{ key }}: {{ value }}
{% endfor %}

{% if conversation_context %}
**Conversation History:**
{{ conversation_context }}
{% endif %}

**Response Guidelines:**
1. Address the user's question directly
2. Provide relevant financial insights
3. Offer practical advice when appropriate
4. Maintain a helpful and professional tone
5. Ask clarifying questions if needed
{% if include_examples %}6. Use examples to illustrate points{% endif %}

**Communication Level:** {{ complexity_level }}
{% if complexity_level == 'beginner' %}
- Use simple, encouraging language
- Explain financial concepts clearly
- Focus on building financial literacy
{% elif complexity_level == 'expert' %}
- Use sophisticated financial terminology
- Provide detailed analytical insights
- Reference advanced financial concepts
{% endif %}

Provide a helpful, informative response that advances the conversation.""",
            description="Handles general financial questions and conversations",
            required_context=["content"],
            optional_context=["user_profile", "investment_experience", "current_holdings"],
            complexity_variations={
                ExplanationComplexity.BEGINNER: "supportive and educational",
                ExplanationComplexity.INTERMEDIATE: "informative and balanced",
                ExplanationComplexity.ADVANCED: "detailed and analytical",
                ExplanationComplexity.EXPERT: "sophisticated and comprehensive"
            }
        )
    
    async def generate_explanation_prompt(
        self,
        content: str,
        context: Dict[str, Any],
        complexity: ExplanationComplexity,
        prompt_type: PromptType = PromptType.GENERAL_CHAT,
        conversation_context: str = "",
        include_examples: bool = True
    ) -> str:
        """Generate a prompt for explanation based on type and context"""
        try:
            template_config = self.templates.get(prompt_type)
            if not template_config:
                raise ValueError(f"Unknown prompt type: {prompt_type}")
            
            # Get complexity level description
            complexity_level = template_config.complexity_variations.get(
                complexity, "balanced"
            )
            
            # Prepare template variables
            template_vars = {
                "content": content,
                "context": context,
                "complexity_level": complexity_level,
                "conversation_context": conversation_context,
                "include_examples": include_examples
            }
            
            # Add context variables directly to template vars
            template_vars.update(context)
            
            # Render the template
            template = self.jinja_env.from_string(template_config.template)
            prompt = template.render(**template_vars)
            
            return prompt.strip()
            
        except Exception as e:
            logger.error(f"Error generating prompt: {str(e)}")
            # Fallback to simple prompt
            return f"Please explain the following in {complexity.value} terms: {content}"
    
    async def generate_technical_analysis_prompt(
        self,
        indicators: Dict[str, Any],
        symbol: str,
        timeframe: str,
        complexity: ExplanationComplexity = ExplanationComplexity.INTERMEDIATE
    ) -> str:
        """Generate specialized prompt for technical analysis"""
        context = {
            "symbol": symbol,
            "timeframe": timeframe,
            "indicators": indicators
        }
        
        # Format indicators for better readability
        content = "Technical Indicators:\n"
        for indicator, value in indicators.items():
            if isinstance(value, dict):
                content += f"\n{indicator}:\n"
                for k, v in value.items():
                    content += f"  {k}: {v}\n"
            else:
                content += f"{indicator}: {value}\n"
        
        return await self.generate_explanation_prompt(
            content=content,
            context=context,
            complexity=complexity,
            prompt_type=PromptType.TECHNICAL_ANALYSIS
        )
    
    async def generate_model_interpretation_prompt(
        self,
        model_output: Dict[str, Any],
        model_info: Dict[str, Any],
        complexity: ExplanationComplexity = ExplanationComplexity.INTERMEDIATE
    ) -> str:
        """Generate specialized prompt for AI/ML model interpretation"""
        context = model_info.copy()
        
        # Format model output
        content = "Model Prediction Results:\n"
        for key, value in model_output.items():
            if isinstance(value, (list, dict)):
                content += f"{key}: {json.dumps(value, indent=2)}\n"
            else:
                content += f"{key}: {value}\n"
        
        return await self.generate_explanation_prompt(
            content=content,
            context=context,
            complexity=complexity,
            prompt_type=PromptType.MODEL_INTERPRETATION
        )
    
    async def generate_risk_assessment_prompt(
        self,
        risk_metrics: Dict[str, Any],
        portfolio_context: Dict[str, Any],
        complexity: ExplanationComplexity = ExplanationComplexity.INTERMEDIATE
    ) -> str:
        """Generate specialized prompt for risk assessment"""
        context = portfolio_context.copy()
        
        # Format risk metrics
        content = "Risk Assessment Results:\n"
        for metric, value in risk_metrics.items():
            if isinstance(value, dict):
                content += f"\n{metric}:\n"
                for k, v in value.items():
                    content += f"  {k}: {v}\n"
            else:
                content += f"{metric}: {value}\n"
        
        return await self.generate_explanation_prompt(
            content=content,
            context=context,
            complexity=complexity,
            prompt_type=PromptType.RISK_ASSESSMENT
        )
    
    async def generate_market_explanation_prompt(
        self,
        market_data: Dict[str, Any],
        market_context: Dict[str, Any],
        complexity: ExplanationComplexity = ExplanationComplexity.INTERMEDIATE
    ) -> str:
        """Generate specialized prompt for market data explanation"""
        context = market_context.copy()
        
        # Format market data
        content = "Market Data Analysis:\n"
        for key, value in market_data.items():
            if isinstance(value, dict):
                content += f"\n{key}:\n"
                for k, v in value.items():
                    content += f"  {k}: {v}\n"
            else:
                content += f"{key}: {value}\n"
        
        return await self.generate_explanation_prompt(
            content=content,
            context=context,
            complexity=complexity,
            prompt_type=PromptType.MARKET_EXPLANATION
        )
    
    def get_available_prompt_types(self) -> List[Dict[str, str]]:
        """Get list of available prompt types with descriptions"""
        return [
            {
                "type": prompt_type.value,
                "name": template.name,
                "description": template.description,
                "required_context": template.required_context,
                "optional_context": template.optional_context
            }
            for prompt_type, template in self.templates.items()
        ]
    
    def validate_context(
        self,
        prompt_type: PromptType,
        context: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Validate that required context is provided"""
        template = self.templates.get(prompt_type)
        if not template:
            return {"errors": [f"Unknown prompt type: {prompt_type}"]}
        
        missing_required = [
            field for field in template.required_context
            if field not in context or context[field] is None
        ]
        
        validation_result = {}
        if missing_required:
            validation_result["missing_required"] = missing_required
        
        available_optional = [
            field for field in template.optional_context
            if field in context and context[field] is not None
        ]
        
        if available_optional:
            validation_result["available_optional"] = available_optional
        
        return validation_result

# Global prompt engine instance
prompt_engine = PromptEngine()