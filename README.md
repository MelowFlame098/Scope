# Scope - Financial Analysis Platform

## Overview
Scope is a comprehensive financial analysis platform combining real-time stock tracking, AI-driven news summarization, and quantitative modeling.

## Development Plan

### Phase 1: Foundation & Infrastructure (Current Focus)
- [ ] **Infrastructure**: Docker Compose setup for PostgreSQL, Redis, and MongoDB.
- [ ] **Backend Core**: Golang service setup with basic routing and database connections.
- [ ] **Frontend Core**: React + TypeScript setup with a basic shell.

### Phase 2: Core Features (MVP)
- [ ] **Authentication**: User Signup/Login (PostgreSQL).
- [ ] **Market Data**: Integration with stock APIs, storing hot data in Redis.
- [ ] **Visualization**: Candlestick charts using a library like Recharts or Lightweight Charts.
- [ ] **Stock Watchlist**: "My Stocks" feature.

### Phase 3: Advanced Intelligence
- [ ] **Data Collection**: Web crawlers for news (Python/Go).
- [ ] **AI Service**: Python service for news summarization and sentiment analysis.
- [ ] **Storage**: Storing news/summaries in MongoDB.

### Phase 4: Quantitative Analysis
- [ ] **Quant Engine**: C++ modules for high-performance mathematical modeling.
- [ ] **Integration**: Interfacing C++ models with the Go/Python backend.

## Architecture & Tech Stack

The system is designed as a polyglot microservices-inspired architecture to leverage the best tools for each domain.

### 1. Backend API (Golang)
- **Role**: The central nervous system. Handles client requests, authentication, and orchestrates data flow.
- **Why**: High concurrency, type safety, and speed. Perfect for financial transactions and API gateway logic.
- **Dependencies**: `Gin` or `Echo` (Router), `Gorm` or `Sqlx` (DB).

### 2. Frontend (React + TypeScript)
- **Role**: User Interface.
- **Why**: Type safety matches the backend; React ecosystem offers rich charting libraries.
- **State Management**: React Query (Server state), Zustand/Redux (Client state).

### 3. AI & Data Science Service (Python)
- **Role**: News crawling, NLP (Sentiment Analysis), AI Summarization.
- **Why**: Unrivaled ML ecosystem (PyTorch, TensorFlow, HuggingFace, NLTK).
- **Communication**: gRPC or REST to the Main Backend.

### 4. Quantitative Engine (C++)
- **Role**: Heavy mathematical computation, Order Flow analysis, Quant Models.
- **Why**: Maximum performance and low latency for math operations.
- **Integration**: Exposed via a thin wrapper or Foreign Function Interface (FFI) if tightly coupled, or a separate microservice.

### 5. Data Storage
- **PostgreSQL**: Relational data (Users, User Portfolios, Ledger/Transactions). **Strict ACID compliance.**
- **Redis**: Volatile/Cache data (Real-time stock prices, Order Book snapshots, Session tokens).
- **MongoDB**: Unstructured data (News articles, Social media sentiment logs, JSON blobs).

## Directory Structure
```
/scope
  /backend       # Go API
  /frontend      # React App
  /ai-service    # Python AI/ML
  /quant-engine  # C++ Models
  /infra         # Docker & Configs
```
