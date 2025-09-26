# FinScope Development Setup Guide

## Concurrent Frontend & Backend Development

This guide explains how to run both frontend and backend servers concurrently for development.

## Prerequisites

- Node.js (v18 or higher)
- Python 3.11+
- Virtual environment activated (`.venv`)

## Quick Start

### Option 1: Concurrent Development (Recommended)

Run both frontend and backend together:

```bash
npm run dev:full
```

This will start:
- **Backend API**: http://localhost:8000 (FastAPI with SQLite)
- **Frontend**: http://localhost:3000 (Next.js)

### Option 2: Individual Services

Run services separately:

```bash
# Backend only
npm run dev:backend

# Frontend only
npm run dev:frontend
```

### Option 3: Docker Compose

For production-like environment:

```bash
# Start all services (PostgreSQL, Redis, Backend, Frontend, Nginx)
npm run docker:up

# Stop all services
npm run docker:down

# Rebuild containers
npm run docker:build
```

## Available Scripts

| Script | Description |
|--------|-------------|
| `npm run dev:full` | Run both frontend and backend concurrently |
| `npm run dev:backend` | Run backend API server only |
| `npm run dev:frontend` | Run frontend development server only |
| `npm run docker:up` | Start Docker Compose services |
| `npm run docker:down` | Stop Docker Compose services |
| `npm run docker:build` | Build Docker containers |

## Development URLs

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Backend Health Check**: http://localhost:8000/health

## Backend Configuration

### Database
- **Development**: SQLite (`finscope.db`)
- **Production**: PostgreSQL (via Docker)

### Environment Variables
Backend uses `.env` file in the `backend/` directory:

```env
DATABASE_URL=sqlite:///./finscope.db
SECRET_KEY=your-secret-key
ENVIRONMENT=development
DEBUG=true
```

## Frontend Configuration

### Environment Variables
Frontend uses environment variables for API connection:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

## Troubleshooting

### Backend Issues
1. **Database Connection**: Ensure SQLite database is accessible
2. **Missing Dependencies**: Run `pip install -r backend/requirements.txt`
3. **Port Conflicts**: Backend runs on port 8000 by default

### Frontend Issues
1. **Missing UI Components**: Some components reference `./ui/*` modules that may need to be created
2. **API Connection**: Ensure backend is running on port 8000
3. **Port Conflicts**: Frontend runs on port 3000 by default

### Concurrent Development
1. **Process Management**: Uses `concurrently` package to manage both processes
2. **Logs**: Color-coded logs (blue for backend, green for frontend)
3. **Stopping**: Use `Ctrl+C` to stop both processes

## Development Workflow

1. **Start Development**:
   ```bash
   npm run dev:full
   ```

2. **Make Changes**:
   - Backend changes auto-reload (uvicorn with `--reload`)
   - Frontend changes auto-reload (Next.js hot reload)

3. **Test API**:
   - Visit http://localhost:8000/docs for interactive API documentation
   - Use http://localhost:8000/health for health checks

4. **Test Frontend**:
   - Visit http://localhost:3000 for the web application
   - Check browser console for any errors

## Production Deployment

For production deployment, use Docker Compose:

```bash
docker-compose -f docker-compose.prod.yml up -d
```

This includes:
- PostgreSQL database
- Redis for caching
- Backend API server
- Frontend application
- Nginx reverse proxy with SSL

## Next Steps

1. **UI Components**: Create missing UI components in `src/components/ui/`
2. **API Integration**: Connect frontend components to backend endpoints
3. **Authentication**: Implement user authentication flow
4. **Real-time Features**: Set up WebSocket connections
5. **Testing**: Add unit and integration tests