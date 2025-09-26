"""LangChain Integration for FinScope - Phase 6 Implementation

Provides advanced LLM capabilities using LangChain framework including
chains, agents, memory, and retrieval-augmented generation (RAG).
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# LangChain imports
try:
    from langchain.llms import OpenAI, Anthropic
    from langchain.chat_models import ChatOpenAI, ChatAnthropic
    from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
    from langchain.vectorstores import Chroma, FAISS, Pinecone
    from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
    from langchain.chains import (
        LLMChain, ConversationChain, RetrievalQA, 
        AnalyzeDocumentChain, MapReduceDocumentsChain
    )
    from langchain.agents import (
        initialize_agent, AgentType, Tool, 
        create_react_agent, create_openai_functions_agent
    )
    from langchain.prompts import (
        PromptTemplate, ChatPromptTemplate, 
        SystemMessagePromptTemplate, HumanMessagePromptTemplate
    )
    from langchain.schema import Document, BaseMessage, HumanMessage, AIMessage
    from langchain.callbacks import AsyncCallbackHandler
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.document_loaders import TextLoader, PDFLoader
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain not available. Install with: pip install langchain")

class ChainType(str, Enum):
    """Types of LangChain chains"""
    CONVERSATION = "conversation"
    RETRIEVAL_QA = "retrieval_qa"
    ANALYSIS = "analysis"
    SUMMARIZATION = "summarization"
    FINANCIAL_ADVISOR = "financial_advisor"
    RISK_ASSESSMENT = "risk_assessment"
    MARKET_ANALYSIS = "market_analysis"
    PORTFOLIO_OPTIMIZATION = "portfolio_optimization"

class AgentType(str, Enum):
    """Types of LangChain agents"""
    REACT = "react"
    OPENAI_FUNCTIONS = "openai_functions"
    CONVERSATIONAL = "conversational"
    FINANCIAL_ANALYST = "financial_analyst"
    PORTFOLIO_MANAGER = "portfolio_manager"
    RISK_MANAGER = "risk_manager"

class MemoryType(str, Enum):
    """Types of conversation memory"""
    BUFFER = "buffer"
    SUMMARY = "summary"
    ENTITY = "entity"
    KNOWLEDGE_GRAPH = "knowledge_graph"

@dataclass
class ChainConfig:
    """Configuration for LangChain chains"""
    chain_type: ChainType = ChainType.FINANCIAL_ADVISOR
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 1000
    memory_type: MemoryType = MemoryType.BUFFER
    retrieval_enabled: bool = False
    vector_store_path: Optional[str] = None
    custom_prompts: Optional[Dict[str, str]] = None

class LangChainRequest(BaseModel):
    """Request for LangChain processing"""
    query: str
    chain_type: ChainType = ChainType.CONVERSATION
    agent_type: Optional[AgentType] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None
    config: Optional[ChainConfig] = None
    tools: List[str] = Field(default_factory=list)
    documents: List[str] = Field(default_factory=list)

class LangChainResponse(BaseModel):
    """Response from LangChain processing"""
    response: str
    chain_type: ChainType
    confidence: float
    sources: List[str] = Field(default_factory=list)
    intermediate_steps: List[Dict[str, Any]] = Field(default_factory=list)
    memory_summary: Optional[str] = None
    tokens_used: int = 0
    processing_time: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

if LANGCHAIN_AVAILABLE:
    class FinancialStreamingCallback(AsyncCallbackHandler):
        """Custom callback for streaming financial responses"""
        
        def __init__(self, callback_func: Optional[Callable] = None):
            self.callback_func = callback_func
            self.tokens = []
            self.start_time = datetime.utcnow()
        
        async def on_llm_new_token(self, token: str, **kwargs) -> None:
            """Handle new token from LLM"""
            self.tokens.append(token)
            if self.callback_func:
                await self.callback_func(token)
        
        async def on_llm_end(self, response, **kwargs) -> None:
            """Handle LLM completion"""
            end_time = datetime.utcnow()
            processing_time = (end_time - self.start_time).total_seconds()
            
            if self.callback_func:
                await self.callback_func({
                    "type": "completion",
                    "tokens_generated": len(self.tokens),
                    "processing_time": processing_time
                })
else:
    class FinancialStreamingCallback:
        """Placeholder callback when LangChain is not available"""
        
        def __init__(self, callback_func: Optional[Callable] = None):
            self.callback_func = callback_func
            self.tokens = []
            self.start_time = datetime.utcnow()
        
        async def on_llm_new_token(self, token: str, **kwargs) -> None:
            """Handle new token from LLM"""
            self.tokens.append(token)
            if self.callback_func:
                await self.callback_func(token)
        
        async def on_llm_end(self, response, **kwargs) -> None:
            """Handle LLM completion"""
            end_time = datetime.utcnow()
            processing_time = (end_time - self.start_time).total_seconds()
            
            if self.callback_func:
                await self.callback_func({
                    "type": "completion",
                    "tokens_generated": len(self.tokens),
                    "processing_time": processing_time
                })

class LangChainIntegration:
    """Advanced LangChain integration for financial AI"""
    
    def __init__(self):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is required but not installed")
        
        # Initialize models
        self.chat_models = {}
        self.embeddings = {}
        self.vector_stores = {}
        self.chains = {}
        self.agents = {}
        self.memories = {}
        
        # Initialize default configurations
        self._initialize_models()
        self._initialize_chains()
        self._initialize_agents()
        self._initialize_tools()
    
    def _initialize_models(self):
        """Initialize LLM models and embeddings"""
        try:
            # Chat models
            self.chat_models["openai"] = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0.7,
                streaming=True
            )
            
            self.chat_models["openai-4"] = ChatOpenAI(
                model_name="gpt-4",
                temperature=0.7,
                streaming=True
            )
            
            # Embeddings
            self.embeddings["openai"] = OpenAIEmbeddings()
            self.embeddings["huggingface"] = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
        except Exception as e:
            logger.warning(f"Could not initialize some models: {e}")
    
    def _initialize_chains(self):
        """Initialize predefined chains"""
        # Financial advisor chain
        financial_advisor_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            You are an expert financial advisor with deep knowledge of markets, investments, and financial planning.
            
            Context: {context}
            
            Question: {question}
            
            Provide a comprehensive, actionable response that:
            1. Addresses the specific question
            2. Considers risk factors
            3. Provides practical recommendations
            4. Explains the reasoning behind your advice
            5. Mentions any important disclaimers
            
            Response:
            """
        )
        
        self.chains[ChainType.FINANCIAL_ADVISOR] = LLMChain(
            llm=self.chat_models.get("openai"),
            prompt=financial_advisor_prompt,
            verbose=True
        )
        
        # Market analysis chain
        market_analysis_prompt = PromptTemplate(
            input_variables=["market_data", "technical_indicators", "news", "question"],
            template="""
            You are a senior market analyst with expertise in technical and fundamental analysis.
            
            Market Data: {market_data}
            Technical Indicators: {technical_indicators}
            Recent News: {news}
            
            Analysis Request: {question}
            
            Provide a detailed market analysis that includes:
            1. Current market conditions assessment
            2. Technical analysis insights
            3. Fundamental factors
            4. Risk assessment
            5. Short-term and long-term outlook
            6. Trading/investment implications
            
            Analysis:
            """
        )
        
        self.chains[ChainType.MARKET_ANALYSIS] = LLMChain(
            llm=self.chat_models.get("openai"),
            prompt=market_analysis_prompt,
            verbose=True
        )
        
        # Risk assessment chain
        risk_assessment_prompt = PromptTemplate(
            input_variables=["portfolio_data", "market_conditions", "risk_tolerance", "question"],
            template="""
            You are a risk management expert specializing in portfolio risk assessment.
            
            Portfolio Data: {portfolio_data}
            Market Conditions: {market_conditions}
            Risk Tolerance: {risk_tolerance}
            
            Risk Assessment Request: {question}
            
            Provide a comprehensive risk assessment that covers:
            1. Portfolio risk metrics analysis
            2. Concentration risk evaluation
            3. Market risk exposure
            4. Liquidity risk assessment
            5. Stress testing scenarios
            6. Risk mitigation recommendations
            
            Risk Assessment:
            """
        )
        
        self.chains[ChainType.RISK_ASSESSMENT] = LLMChain(
            llm=self.chat_models.get("openai"),
            prompt=risk_assessment_prompt,
            verbose=True
        )
    
    def _initialize_agents(self):
        """Initialize specialized financial agents"""
        # Financial analyst agent tools
        financial_tools = self._get_financial_tools()
        
        if financial_tools:
            try:
                self.agents[AgentType.FINANCIAL_ANALYST] = initialize_agent(
                    tools=financial_tools,
                    llm=self.chat_models.get("openai"),
                    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    verbose=True,
                    max_iterations=3
                )
            except Exception as e:
                logger.warning(f"Could not initialize financial analyst agent: {e}")
    
    def _initialize_tools(self):
        """Initialize tools for agents"""
        if LANGCHAIN_AVAILABLE:
            self.tools = {
                "market_data": Tool(
                    name="Market Data",
                    description="Get current market data for stocks, indices, and other securities",
                    func=self._get_market_data_tool
                ),
                "technical_analysis": Tool(
                    name="Technical Analysis",
                    description="Perform technical analysis on price data and indicators",
                    func=self._get_technical_analysis_tool
                ),
                "news_search": Tool(
                    name="News Search",
                    description="Search for relevant financial news and analysis",
                    func=self._get_news_search_tool
                ),
                "portfolio_analysis": Tool(
                    name="Portfolio Analysis",
                    description="Analyze portfolio composition, performance, and risk metrics",
                    func=self._get_portfolio_analysis_tool
                ),
                "economic_data": Tool(
                    name="Economic Data",
                    description="Get economic indicators and macroeconomic data",
                    func=self._get_economic_data_tool
                )
            }
        else:
            self.tools = {}
    
    async def process_request(
        self,
        request: LangChainRequest,
        streaming_callback: Optional[Callable] = None
    ) -> LangChainResponse:
        """Process request using appropriate chain or agent"""
        start_time = datetime.utcnow()
        
        try:
            # Setup callback if streaming
            callback = None
            if streaming_callback:
                callback = FinancialStreamingCallback(streaming_callback)
            
            # Get or create memory for conversation
            memory = self._get_or_create_memory(
                request.conversation_id,
                request.config.memory_type if request.config else MemoryType.BUFFER
            )
            
            # Process based on request type
            if request.agent_type:
                response_text = await self._process_with_agent(
                    request, memory, callback
                )
            else:
                response_text = await self._process_with_chain(
                    request, memory, callback
                )
            
            # Calculate processing time
            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds()
            
            # Create response
            response = LangChainResponse(
                response=response_text,
                chain_type=request.chain_type,
                confidence=0.85,  # Mock confidence - could be calculated
                processing_time=processing_time,
                memory_summary=self._get_memory_summary(memory),
                metadata={
                    "model_used": request.config.model_name if request.config else "gpt-3.5-turbo",
                    "conversation_id": request.conversation_id,
                    "user_id": request.user_id
                }
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing LangChain request: {e}")
            return LangChainResponse(
                response=f"Error processing request: {str(e)}",
                chain_type=request.chain_type,
                confidence=0.0,
                processing_time=(datetime.utcnow() - start_time).total_seconds()
            )
    
    async def _process_with_chain(
        self,
        request: LangChainRequest,
        memory: Any,
        callback: Optional[FinancialStreamingCallback]
    ) -> str:
        """Process request using appropriate chain"""
        chain = self.chains.get(request.chain_type)
        
        if not chain:
            # Create default conversation chain
            chain = ConversationChain(
                llm=self.chat_models.get("openai"),
                memory=memory,
                verbose=True
            )
        
        # Prepare input based on chain type
        if request.chain_type == ChainType.FINANCIAL_ADVISOR:
            result = await chain.arun(
                context=str(request.context),
                question=request.query,
                callbacks=[callback] if callback else None
            )
        elif request.chain_type == ChainType.MARKET_ANALYSIS:
            result = await chain.arun(
                market_data=request.context.get("market_data", ""),
                technical_indicators=request.context.get("technical_indicators", ""),
                news=request.context.get("news", ""),
                question=request.query,
                callbacks=[callback] if callback else None
            )
        elif request.chain_type == ChainType.RISK_ASSESSMENT:
            result = await chain.arun(
                portfolio_data=request.context.get("portfolio_data", ""),
                market_conditions=request.context.get("market_conditions", ""),
                risk_tolerance=request.context.get("risk_tolerance", "moderate"),
                question=request.query,
                callbacks=[callback] if callback else None
            )
        else:
            # Default conversation
            result = await chain.arun(
                input=request.query,
                callbacks=[callback] if callback else None
            )
        
        return result
    
    async def _process_with_agent(
        self,
        request: LangChainRequest,
        memory: Any,
        callback: Optional[FinancialStreamingCallback]
    ) -> str:
        """Process request using appropriate agent"""
        agent = self.agents.get(request.agent_type)
        
        if not agent:
            raise ValueError(f"Agent type {request.agent_type} not available")
        
        # Run agent
        result = await agent.arun(
            input=request.query,
            callbacks=[callback] if callback else None
        )
        
        return result
    
    def _get_or_create_memory(
        self,
        conversation_id: Optional[str],
        memory_type: MemoryType
    ) -> Any:
        """Get or create conversation memory"""
        if not conversation_id:
            # Create temporary memory
            if memory_type == MemoryType.SUMMARY:
                return ConversationSummaryMemory(
                    llm=self.chat_models.get("openai")
                )
            else:
                return ConversationBufferMemory()
        
        # Get existing memory or create new one
        if conversation_id not in self.memories:
            if memory_type == MemoryType.SUMMARY:
                self.memories[conversation_id] = ConversationSummaryMemory(
                    llm=self.chat_models.get("openai")
                )
            else:
                self.memories[conversation_id] = ConversationBufferMemory()
        
        return self.memories[conversation_id]
    
    def _get_memory_summary(self, memory: Any) -> Optional[str]:
        """Get summary of conversation memory"""
        try:
            if hasattr(memory, 'buffer'):
                return memory.buffer[:200] + "..." if len(memory.buffer) > 200 else memory.buffer
            elif hasattr(memory, 'summary'):
                return memory.summary
            return None
        except Exception:
            return None
    
    def _get_financial_tools(self) -> List:
        """Get financial analysis tools for agents"""
        if not LANGCHAIN_AVAILABLE:
            return []
        return [
            self.tools["market_data"],
            self.tools["technical_analysis"],
            self.tools["news_search"],
            self.tools["portfolio_analysis"],
            self.tools["economic_data"]
        ]
    
    def _get_market_data_tool(self, query: str) -> str:
        """Tool function for market data"""
        # Mock implementation - replace with real market data service
        return f"Market data for {query}: Current price $150.25, Volume 1.2M, Change +2.5%"
    
    def _get_technical_analysis_tool(self, query: str) -> str:
        """Tool function for technical analysis"""
        # Mock implementation - replace with real technical analysis
        return f"Technical analysis for {query}: RSI 65.4 (neutral), MACD bullish crossover, Support at $145"
    
    def _get_news_search_tool(self, query: str) -> str:
        """Tool function for news search"""
        # Mock implementation - replace with real news search
        return f"Recent news for {query}: Strong earnings report, analyst upgrade, positive market sentiment"
    
    def _get_portfolio_analysis_tool(self, query: str) -> str:
        """Tool function for portfolio analysis"""
        # Mock implementation - replace with real portfolio analysis
        return f"Portfolio analysis: Total value $125K, YTD return +12.5%, Beta 1.15, Sharpe ratio 1.25"
    
    def _get_economic_data_tool(self, query: str) -> str:
        """Tool function for economic data"""
        # Mock implementation - replace with real economic data
        return f"Economic data: Fed rate 5.25%, Unemployment 3.8%, CPI 3.2%, GDP growth 2.1%"
    
    async def create_vector_store(
        self,
        documents: List[str],
        store_name: str,
        embedding_model: str = "openai"
    ) -> bool:
        """Create vector store from documents"""
        try:
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            
            docs = []
            for doc_text in documents:
                chunks = text_splitter.split_text(doc_text)
                docs.extend([Document(page_content=chunk) for chunk in chunks])
            
            # Create embeddings
            embeddings = self.embeddings.get(embedding_model)
            if not embeddings:
                raise ValueError(f"Embedding model {embedding_model} not available")
            
            # Create vector store
            vector_store = FAISS.from_documents(docs, embeddings)
            self.vector_stores[store_name] = vector_store
            
            logger.info(f"Created vector store '{store_name}' with {len(docs)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            return False
    
    async def query_vector_store(
        self,
        store_name: str,
        query: str,
        k: int = 5
    ) -> List:
        """Query vector store for relevant documents"""
        vector_store = self.vector_stores.get(store_name)
        if not vector_store:
            raise ValueError(f"Vector store '{store_name}' not found")
        
        return vector_store.similarity_search(query, k=k)
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """Get available models and capabilities"""
        return {
            "chat_models": list(self.chat_models.keys()),
            "embeddings": list(self.embeddings.keys()),
            "chains": [chain_type.value for chain_type in ChainType],
            "agents": [agent_type.value for agent_type in AgentType],
            "vector_stores": list(self.vector_stores.keys())
        }
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get service status and health"""
        return {
            "langchain_available": LANGCHAIN_AVAILABLE,
            "models_loaded": len(self.chat_models),
            "chains_available": len(self.chains),
            "agents_available": len(self.agents),
            "vector_stores": len(self.vector_stores),
            "active_conversations": len(self.memories),
            "tools_available": len(self.tools)
        }

# Global LangChain integration instance
langchain_integration = LangChainIntegration() if LANGCHAIN_AVAILABLE else None