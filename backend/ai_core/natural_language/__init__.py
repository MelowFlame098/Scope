# Natural Language Interface Module
# Phase 9: AI-First Platform Implementation

"""
Natural Language Interface for FinScope AI-First Platform

This module provides conversational AI capabilities allowing users to:
- Query portfolio and market data using natural language
- Execute trades through voice/text commands
- Get AI-powered financial insights and recommendations
- Interact with the system through intuitive conversations
"""

__version__ = "9.0.0"
__author__ = "FinScope AI Team"

# Import natural language components
from .conversational_ai import ConversationalAI
from .query_processor import QueryProcessor
from .response_generator import ResponseGenerator
from .intent_classifier import IntentClassifier
from .entity_extractor import EntityExtractor
from .context_manager import ContextManager

__all__ = [
    "ConversationalAI",
    "QueryProcessor",
    "ResponseGenerator",
    "IntentClassifier",
    "EntityExtractor",
    "ContextManager"
]