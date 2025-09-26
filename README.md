# FinScope - Advanced Financial Analysis Platform

FinScope is a comprehensive financial analysis platform that combines real-time market data, AI-powered insights, news aggregation, and community features to provide traders and investors with powerful tools for making informed decisions.

## 🚀 Features

### Frontend (Next.js + React)
- **Modern Dashboard**: Clean, responsive interface with dark mode support
- **Real-time Market Data**: Live price updates and charts
- **Interactive Charts**: Multiple chart types with technical indicators
- **AI-Powered Insights**: Machine learning analysis and predictions
- **News Aggregation**: Financial news with sentiment analysis
- **Community Forum**: Discussion platform for traders
- **Asset Research**: Detailed fundamental and technical analysis
- **Watchlist Management**: Track your favorite assets
- **Model Selection**: Choose from various trading models

### Backend (FastAPI + Python)
- **RESTful API**: Comprehensive API for all platform features
- **WebSocket Support**: Real-time data streaming
- **JWT Authentication**: Secure user authentication
- **Database Integration**: PostgreSQL with SQLAlchemy ORM
- **Market Data Integration**: Multiple data sources (Yahoo Finance, CoinGecko, etc.)
- **AI/ML Services**: Technical analysis and price predictions
- **News Aggregation**: Multi-source news collection with sentiment analysis
- **Rate Limiting**: API protection and usage control

## 🛠️ Technology Stack

### Frontend
- **Framework**: Next.js 14 with App Router
- **UI Library**: React 18 with TypeScript
- **Styling**: Tailwind CSS
- **State Management**: Zustand
- **Charts**: Recharts
- **Icons**: Heroicons
- **HTTP Client**: Axios
- **Real-time**: Socket.io Client

### Backend
- **Framework**: FastAPI
- **Language**: Python 3.9+
- **Database**: PostgreSQL (SQLite for development)
- **ORM**: SQLAlchemy
- **Authentication**: JWT with bcrypt
- **Real-time**: WebSockets
- **Data Sources**: Yahoo Finance, CoinGecko, NewsAPI
- **AI/ML**: scikit-learn, pandas, numpy
- **NLP**: TextBlob for sentiment analysis

## 📋 Prerequisites

- **Node.js** 18+ and npm/yarn
- **Python** 3.9+
- **PostgreSQL** (optional, SQLite works for development)
- **Git**

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/finscope.git
cd finscope
```

### 2. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp ../.env.example .env

# Edit .env file with your API keys and configuration
# At minimum, set:
# - SECRET_KEY (generate a secure random string)
# - DATABASE_URL (or use SQLite default)

# Run database migrations
python -c "from database import init_db; init_db()"

# Start the backend server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Frontend Setup

```bash
# Open new terminal and navigate to project root
cd finscope

# Install dependencies
npm install
# or
yarn install

# Start the development server
npm run dev
# or
yarn dev
```

### 4. Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 🔧 Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure the following:

#### Required
- `SECRET_KEY`: JWT secret key (generate a secure random string)
- `DATABASE_URL`: Database connection string

#### Optional (for full functionality)
- `ALPHA_VANTAGE_API_KEY`: For stock market data
- `FINNHUB_API_KEY`: For financial data
- `COINGECKO_API_KEY`: For cryptocurrency data
- `NEWSAPI_KEY`: For news aggregation
- `OPENAI_API_KEY`: For AI-powered explanations

### API Keys Setup

1. **Alpha Vantage**: Get free API key at https://www.alphavantage.co/support/#api-key
2. **Finnhub**: Register at https://finnhub.io/
3. **NewsAPI**: Get key at https://newsapi.org/
4. **OpenAI**: Get API key at https://platform.openai.com/

## 📚 API Documentation

Once the backend is running, visit http://localhost:8000/docs for interactive API documentation powered by Swagger UI.

### Key Endpoints

- `POST /auth/register` - User registration
- `POST /auth/login` - User login
- `GET /market/assets` - Get available assets
- `GET /market/chart/{symbol}` - Get chart data
- `GET /news` - Get financial news
- `POST /ai/analyze` - AI analysis request
- `WebSocket /ws` - Real-time data stream

## 🏗️ Project Structure

```
finscope/
├── src/                          # Frontend source code
│   ├── app/                      # Next.js app directory
│   │   ├── components/           # React components
│   │   │   ├── Sidebar.tsx
│   │   │   ├── Header.tsx
│   │   │   ├── Dashboard.tsx
│   │   │   └── ...
│   │   ├── globals.css           # Global styles
│   │   ├── layout.tsx            # Root layout
│   │   └── page.tsx              # Home page
│   └── ...
├── backend/                      # Backend source code
│   ├── main.py                   # FastAPI application
│   ├── database.py               # Database configuration
│   ├── models.py                 # SQLAlchemy models
│   ├── schemas.py                # Pydantic schemas
│   ├── auth.py                   # Authentication logic
│   ├── market_data.py            # Market data service
│   ├── ai_service.py             # AI/ML service
│   ├── news_service.py           # News aggregation service
│   ├── websocket_manager.py      # WebSocket management
│   └── requirements.txt          # Python dependencies
├── package.json                  # Frontend dependencies
├── tailwind.config.ts            # Tailwind configuration
├── next.config.js                # Next.js configuration
├── .env.example                  # Environment variables template
└── README.md                     # This file
```

## 🔄 Development Workflow

### Running in Development Mode

1. **Backend**: `uvicorn main:app --reload --port 8000`
2. **Frontend**: `npm run dev`

### Building for Production

```bash
# Frontend
npm run build
npm start

# Backend
pip install gunicorn
gunicorn main:app --host 0.0.0.0 --port 8000
```

## 🧪 Testing

### Backend Testing

```bash
cd backend
pytest tests/
```

### Frontend Testing

```bash
npm test
# or
yarn test
```

## 🚀 Deployment

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build
```

### Manual Deployment

1. **Frontend**: Deploy to Vercel, Netlify, or similar
2. **Backend**: Deploy to Heroku, AWS, DigitalOcean, or similar
3. **Database**: Use managed PostgreSQL service

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/yourusername/finscope/issues) page
2. Create a new issue with detailed information
3. Join our community discussions

## 🙏 Acknowledgments

- Market data provided by Yahoo Finance, Alpha Vantage, and CoinGecko
- News data from various financial news sources
- UI components inspired by modern financial platforms
- Community contributions and feedback

## 🔮 Roadmap

- [ ] Mobile app development
- [ ] Advanced portfolio analytics
- [ ] Social trading features
- [ ] Options and derivatives support
- [ ] Advanced AI models
- [ ] Multi-language support
- [ ] Dark/Light theme customization
- [ ] Advanced charting tools
- [ ] Backtesting capabilities
- [ ] API rate limiting dashboard

---

**Built with ❤️ for the trading and investment community**