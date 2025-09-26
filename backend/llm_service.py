"""LLM Service for FinScope - Phase 6 Implementation

Provides Large Language Model integration for generating financial explanations,
market analysis, and conversational AI capabilities.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, AsyncGenerator, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

import openai
from anthropic import Anthropic
import httpx
from pydantic import BaseModel, Field

from prompt_engine import PromptEngine
from conversation_manager import ConversationManager
from core.config import settings

logger = logging.getLogger(__name__)

class LLMProvider(str, Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    MISTRAL = "mistral"

class ExplanationComplexity(str, Enum):
    """Explanation complexity levels"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

@dataclass
class LLMConfig:
    """LLM configuration settings"""
    provider: LLMProvider
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 2000
    temperature: float = 0.7
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

class ExplanationRequest(BaseModel):
    """Request model for explanation generation"""
    content: str = Field(..., description="Content to explain")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    complexity: ExplanationComplexity = Field(default=ExplanationComplexity.INTERMEDIATE)
    user_id: Optional[str] = Field(None, description="User ID for personalization")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")
    max_length: int = Field(default=500, description="Maximum explanation length")
    include_examples: bool = Field(default=True, description="Include examples in explanation")

class ExplanationResponse(BaseModel):
    """Response model for explanation generation"""
    explanation: str
    confidence: float
    sources: List[str] = Field(default_factory=list)
    follow_up_questions: List[str] = Field(default_factory=list)
    complexity_level: ExplanationComplexity
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    token_usage: Dict[str, int] = Field(default_factory=dict)

class LLMService:
    """Main LLM service for financial explanations and chat"""
    
    def __init__(self):
        self.configs: Dict[LLMProvider, LLMConfig] = {}
        self.clients: Dict[LLMProvider, Any] = {}
        self.prompt_engine = PromptEngine()
        self.conversation_manager = ConversationManager()
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize LLM providers based on configuration"""
        # OpenAI Configuration
        if hasattr(settings, 'OPENAI_API_KEY') and settings.OPENAI_API_KEY:
            self.configs[LLMProvider.OPENAI] = LLMConfig(
                provider=LLMProvider.OPENAI,
                model_name="gpt-4-turbo-preview",
                api_key=settings.OPENAI_API_KEY,
                max_tokens=2000,
                temperature=0.7
            )
            self.clients[LLMProvider.OPENAI] = openai.AsyncOpenAI(
                api_key=settings.OPENAI_API_KEY
            )
        
        # Anthropic Configuration
        if hasattr(settings, 'ANTHROPIC_API_KEY') and settings.ANTHROPIC_API_KEY:
            self.configs[LLMProvider.ANTHROPIC] = LLMConfig(
                provider=LLMProvider.ANTHROPIC,
                model_name="claude-3-sonnet-20240229",
                api_key=settings.ANTHROPIC_API_KEY,
                max_tokens=2000,
                temperature=0.7
            )
            self.clients[LLMProvider.ANTHROPIC] = Anthropic(
                api_key=settings.ANTHROPIC_API_KEY
            )
        
        # Ollama Configuration (local deployment)
        if hasattr(settings, 'OLLAMA_BASE_URL') and settings.OLLAMA_BASE_URL:
            self.configs[LLMProvider.OLLAMA] = LLMConfig(
                provider=LLMProvider.OLLAMA,
                model_name="llama2:13b",
                base_url=settings.OLLAMA_BASE_URL,
                max_tokens=2000,
                temperature=0.7
            )
    
    async def generate_explanation(
        self,
        request: ExplanationRequest,
        provider: LLMProvider = LLMProvider.OPENAI
    ) -> ExplanationResponse:
        """Generate financial explanation using specified LLM provider"""
        try:
            # Get conversation context if available
            conversation_context = ""
            if request.conversation_id:
                context = await self.conversation_manager.get_context(
                    request.conversation_id
                )
                conversation_context = context.get("summary", "")
            
            # Generate prompt using prompt engine
            prompt = await self.prompt_engine.generate_explanation_prompt(
                content=request.content,
                context=request.context,
                complexity=request.complexity,
                conversation_context=conversation_context,
                include_examples=request.include_examples
            )
            
            # Generate explanation using selected provider
            if provider == LLMProvider.OPENAI:
                response = await self._generate_openai_explanation(prompt, request)
            elif provider == LLMProvider.ANTHROPIC:
                response = await self._generate_anthropic_explanation(prompt, request)
            elif provider == LLMProvider.OLLAMA:
                response = await self._generate_ollama_explanation(prompt, request)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            
            # Update conversation context
            if request.conversation_id:
                await self.conversation_manager.add_interaction(
                    request.conversation_id,
                    user_message=request.content,
                    assistant_response=response.explanation,
                    context=request.context
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            raise
    
    async def _generate_openai_explanation(
        self,
        prompt: str,
        request: ExplanationRequest
    ) -> ExplanationResponse:
        """Generate explanation using OpenAI"""
        client = self.clients[LLMProvider.OPENAI]
        config = self.configs[LLMProvider.OPENAI]
        
        response = await client.chat.completions.create(
            model=config.model_name,
            messages=[
                {"role": "system", "content": "You are a financial expert providing clear, accurate explanations."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            frequency_penalty=config.frequency_penalty,
            presence_penalty=config.presence_penalty
        )
        
        explanation = response.choices[0].message.content
        
        # Generate follow-up questions
        follow_up_questions = await self._generate_follow_up_questions(
            explanation, request.complexity
        )
        
        return ExplanationResponse(
            explanation=explanation,
            confidence=0.85,  # Could be calculated based on model confidence
            follow_up_questions=follow_up_questions,
            complexity_level=request.complexity,
            token_usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        )
    
    async def _generate_anthropic_explanation(
        self,
        prompt: str,
        request: ExplanationRequest
    ) -> ExplanationResponse:
        """Generate explanation using Anthropic Claude"""
        client = self.clients[LLMProvider.ANTHROPIC]
        config = self.configs[LLMProvider.ANTHROPIC]
        
        response = await client.messages.create(
            model=config.model_name,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        explanation = response.content[0].text
        
        # Generate follow-up questions
        follow_up_questions = await self._generate_follow_up_questions(
            explanation, request.complexity
        )
        
        return ExplanationResponse(
            explanation=explanation,
            confidence=0.85,
            follow_up_questions=follow_up_questions,
            complexity_level=request.complexity,
            token_usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            }
        )
    
    async def _generate_ollama_explanation(
        self,
        prompt: str,
        request: ExplanationRequest
    ) -> ExplanationResponse:
        """Generate explanation using Ollama (local LLM)"""
        config = self.configs[LLMProvider.OLLAMA]
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{config.base_url}/api/generate",
                json={
                    "model": config.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": config.temperature,
                        "top_p": config.top_p,
                        "num_predict": config.max_tokens
                    }
                },
                timeout=60.0
            )
            response.raise_for_status()
            
            result = response.json()
            explanation = result.get("response", "")
            
            # Generate follow-up questions
            follow_up_questions = await self._generate_follow_up_questions(
                explanation, request.complexity
            )
            
            return ExplanationResponse(
                explanation=explanation,
                confidence=0.80,  # Local models might have lower confidence
                follow_up_questions=follow_up_questions,
                complexity_level=request.complexity,
                token_usage={
                    "total_tokens": result.get("eval_count", 0)
                }
            )
    
    async def _generate_follow_up_questions(
        self,
        explanation: str,
        complexity: ExplanationComplexity
    ) -> List[str]:
        """Generate relevant follow-up questions based on explanation"""
        # This could be enhanced with another LLM call or rule-based generation
        base_questions = [
            "Can you explain this in simpler terms?",
            "What are the practical implications?",
            "How does this relate to current market conditions?"
        ]
        
        if complexity == ExplanationComplexity.BEGINNER:
            return [
                "What does this mean for my investments?",
                "Should I be concerned about this?",
                "How can I learn more about this topic?"
            ]
        elif complexity == ExplanationComplexity.EXPERT:
            return [
                "What are the quantitative models behind this?",
                "How does this compare to historical patterns?",
                "What are the risk-adjusted implications?"
            ]
        
        return base_questions
    
    async def stream_explanation(
        self,
        request: ExplanationRequest,
        provider: LLMProvider = LLMProvider.OPENAI
    ) -> AsyncGenerator[str, None]:
        """Stream explanation generation for real-time responses"""
        try:
            # Get conversation context
            conversation_context = ""
            if request.conversation_id:
                context = await self.conversation_manager.get_context(
                    request.conversation_id
                )
                conversation_context = context.get("summary", "")
            
            # Generate prompt
            prompt = await self.prompt_engine.generate_explanation_prompt(
                content=request.content,
                context=request.context,
                complexity=request.complexity,
                conversation_context=conversation_context,
                include_examples=request.include_examples
            )
            
            # Stream response based on provider
            if provider == LLMProvider.OPENAI:
                async for chunk in self._stream_openai_explanation(prompt, request):
                    yield chunk
            elif provider == LLMProvider.ANTHROPIC:
                async for chunk in self._stream_anthropic_explanation(prompt, request):
                    yield chunk
            else:
                # For non-streaming providers, yield the complete response
                response = await self.generate_explanation(request, provider)
                yield response.explanation
                
        except Exception as e:
            logger.error(f"Error streaming explanation: {str(e)}")
            yield f"Error: {str(e)}"
    
    async def _stream_openai_explanation(
        self,
        prompt: str,
        request: ExplanationRequest
    ) -> AsyncGenerator[str, None]:
        """Stream explanation from OpenAI"""
        client = self.clients[LLMProvider.OPENAI]
        config = self.configs[LLMProvider.OPENAI]
        
        stream = await client.chat.completions.create(
            model=config.model_name,
            messages=[
                {"role": "system", "content": "You are a financial expert providing clear, accurate explanations."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            stream=True
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    async def _stream_anthropic_explanation(
        self,
        prompt: str,
        request: ExplanationRequest
    ) -> AsyncGenerator[str, None]:
        """Stream explanation from Anthropic Claude"""
        client = self.clients[LLMProvider.ANTHROPIC]
        config = self.configs[LLMProvider.ANTHROPIC]
        
        async with client.messages.stream(
            model=config.model_name,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            messages=[
                {"role": "user", "content": prompt}
            ]
        ) as stream:
            async for text in stream.text_stream:
                yield text
    
    async def get_available_models(self) -> Dict[LLMProvider, List[str]]:
        """Get list of available models for each provider"""
        models = {}
        
        for provider in self.configs.keys():
            if provider == LLMProvider.OPENAI:
                models[provider] = [
                    "gpt-4-turbo-preview",
                    "gpt-4",
                    "gpt-3.5-turbo",
                    "gpt-3.5-turbo-16k"
                ]
            elif provider == LLMProvider.ANTHROPIC:
                models[provider] = [
                    "claude-3-opus-20240229",
                    "claude-3-sonnet-20240229",
                    "claude-3-haiku-20240307"
                ]
            elif provider == LLMProvider.OLLAMA:
                # Could query Ollama API for available models
                models[provider] = ["llama2:13b", "mistral:7b", "codellama:13b"]
        
        return models
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get LLM service status and health information"""
        status = {
            "providers": {},
            "total_providers": len(self.configs),
            "active_conversations": await self.conversation_manager.get_active_count(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        for provider, config in self.configs.items():
            try:
                # Test provider connectivity
                if provider == LLMProvider.OPENAI:
                    client = self.clients[provider]
                    models = await client.models.list()
                    status["providers"][provider.value] = {
                        "status": "healthy",
                        "model": config.model_name,
                        "available_models": len(models.data)
                    }
                elif provider == LLMProvider.ANTHROPIC:
                    status["providers"][provider.value] = {
                        "status": "healthy",
                        "model": config.model_name
                    }
                elif provider == LLMProvider.OLLAMA:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(f"{config.base_url}/api/tags")
                        if response.status_code == 200:
                            status["providers"][provider.value] = {
                                "status": "healthy",
                                "model": config.model_name,
                                "base_url": config.base_url
                            }
                        else:
                            status["providers"][provider.value] = {
                                "status": "unhealthy",
                                "error": "Connection failed"
                            }
            except Exception as e:
                status["providers"][provider.value] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        
        return status

# Global LLM service instance
llm_service = LLMService()