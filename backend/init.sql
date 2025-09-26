-- FinScope Database Initialization Script
-- This script sets up the initial database schema and sample data

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create enum types
CREATE TYPE user_role AS ENUM ('user', 'admin', 'moderator');
CREATE TYPE asset_type AS ENUM ('crypto', 'stock', 'forex', 'commodity', 'index');
CREATE TYPE post_category AS ENUM ('general', 'analysis', 'news', 'question', 'strategy');
CREATE TYPE insight_type AS ENUM ('analysis', 'prediction', 'alert', 'recommendation');
CREATE TYPE model_category AS ENUM ('crypto', 'stocks', 'forex', 'cross_asset');
CREATE TYPE model_type AS ENUM ('technical', 'fundamental', 'ml', 'sentiment', 'statistical');

-- Insert sample assets
INSERT INTO assets (symbol, name, asset_type, current_price, market_cap, volume_24h, price_change_24h, description) VALUES
('BTC', 'Bitcoin', 'crypto', 45000.00, 850000000000, 25000000000, 2.5, 'The first and largest cryptocurrency by market capitalization'),
('ETH', 'Ethereum', 'crypto', 3200.00, 380000000000, 15000000000, 1.8, 'Decentralized platform for smart contracts and DApps'),
('AAPL', 'Apple Inc.', 'stock', 175.50, 2800000000000, 50000000, 0.8, 'Technology company known for iPhone, iPad, and Mac products'),
('GOOGL', 'Alphabet Inc.', 'stock', 142.30, 1800000000000, 25000000, -0.5, 'Multinational technology company specializing in Internet services'),
('TSLA', 'Tesla Inc.', 'stock', 248.75, 790000000000, 75000000, 3.2, 'Electric vehicle and clean energy company'),
('EUR/USD', 'Euro to US Dollar', 'forex', 1.0850, NULL, 1500000000, 0.2, 'Major currency pair representing Euro against US Dollar'),
('GBP/USD', 'British Pound to US Dollar', 'forex', 1.2650, NULL, 800000000, -0.3, 'Major currency pair representing British Pound against US Dollar'),
('GOLD', 'Gold Futures', 'commodity', 2025.50, NULL, 5000000, 1.1, 'Precious metal commodity futures'),
('OIL', 'Crude Oil Futures', 'commodity', 78.25, NULL, 8000000, -1.5, 'Energy commodity futures');

-- Insert sample trading models
INSERT INTO trading_models (name, description, category, model_type, accuracy, risk_level, created_by) VALUES
('Crypto Momentum Scanner', 'Technical analysis model for cryptocurrency momentum trading', 'crypto', 'technical', 0.72, 'medium', 1),
('Stock Value Analyzer', 'Fundamental analysis model for undervalued stocks', 'stocks', 'fundamental', 0.68, 'low', 1),
('Forex Sentiment Tracker', 'Sentiment analysis model for major forex pairs', 'forex', 'sentiment', 0.65, 'medium', 1),
('ML Price Predictor', 'Machine learning model for cross-asset price prediction', 'cross_asset', 'ml', 0.75, 'high', 1),
('Statistical Arbitrage', 'Statistical model for pairs trading opportunities', 'stocks', 'statistical', 0.70, 'medium', 1);

-- Insert sample news articles
INSERT INTO news_articles (title, content, source, url, published_at, category, sentiment, tags, symbols) VALUES
('Bitcoin Reaches New Monthly High', 'Bitcoin has surged to a new monthly high as institutional adoption continues...', 'CryptoNews', 'https://example.com/btc-high', NOW() - INTERVAL '2 hours', 'crypto', 'positive', ARRAY['bitcoin', 'price', 'institutional'], ARRAY['BTC']),
('Federal Reserve Hints at Rate Changes', 'The Federal Reserve has indicated potential changes to interest rates...', 'FinancialTimes', 'https://example.com/fed-rates', NOW() - INTERVAL '4 hours', 'economics', 'neutral', ARRAY['fed', 'rates', 'policy'], ARRAY['USD']),
('Tesla Reports Strong Q4 Earnings', 'Tesla has reported stronger than expected Q4 earnings...', 'TechCrunch', 'https://example.com/tsla-earnings', NOW() - INTERVAL '6 hours', 'earnings', 'positive', ARRAY['tesla', 'earnings', 'ev'], ARRAY['TSLA']),
('Gold Prices Stabilize Amid Market Uncertainty', 'Gold prices have found stability as markets navigate uncertainty...', 'MarketWatch', 'https://example.com/gold-stable', NOW() - INTERVAL '8 hours', 'commodities', 'neutral', ARRAY['gold', 'stability', 'uncertainty'], ARRAY['GOLD']);

-- Insert sample forum posts
INSERT INTO forum_posts (title, content, author_id, category, tags, likes, dislikes, view_count) VALUES
('Best Strategy for Crypto Bull Market?', 'What are your thoughts on the best strategies for the current crypto bull market? I''ve been focusing on DCA but wondering about other approaches...', 1, 'strategy', ARRAY['crypto', 'bull-market', 'strategy'], 15, 2, 234),
('Technical Analysis: BTC Support Levels', 'Looking at the current BTC chart, I see strong support at $42k. What do you think about the next resistance levels?', 1, 'analysis', ARRAY['bitcoin', 'technical-analysis', 'support'], 23, 1, 456),
('Fed Rate Decision Impact on Markets', 'How do you think the upcoming Fed rate decision will impact both crypto and traditional markets?', 1, 'general', ARRAY['fed', 'rates', 'market-impact'], 8, 0, 123),
('New to Trading - Need Advice', 'I''m new to trading and looking for advice on where to start. Any recommended resources or strategies for beginners?', 1, 'question', ARRAY['beginner', 'advice', 'resources'], 12, 0, 189);

-- Insert sample AI insights
INSERT INTO ai_insights (title, content, insight_type, confidence, asset_symbol, model_id, sentiment, tags) VALUES
('BTC Bullish Momentum Detected', 'Technical indicators suggest strong bullish momentum for Bitcoin with RSI showing oversold recovery and volume increasing significantly.', 'analysis', 0.85, 'BTC', 1, 'bullish', ARRAY['bitcoin', 'bullish', 'momentum']),
('AAPL Undervalued Based on Fundamentals', 'Fundamental analysis indicates Apple stock is currently undervalued based on P/E ratio, revenue growth, and market position.', 'analysis', 0.78, 'AAPL', 2, 'bullish', ARRAY['apple', 'undervalued', 'fundamentals']),
('EUR/USD Bearish Sentiment Alert', 'Market sentiment analysis shows increasing bearish sentiment for EUR/USD pair due to economic uncertainty in Europe.', 'alert', 0.72, 'EUR/USD', 3, 'bearish', ARRAY['eurusd', 'bearish', 'sentiment']),
('Gold Price Prediction: Upward Trend', 'ML model predicts gold prices will trend upward over the next 30 days with 75% confidence based on historical patterns and current market conditions.', 'prediction', 0.75, 'GOLD', 4, 'bullish', ARRAY['gold', 'prediction', 'upward']);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_assets_symbol ON assets(symbol);
CREATE INDEX IF NOT EXISTS idx_assets_type ON assets(asset_type);
CREATE INDEX IF NOT EXISTS idx_watchlist_user_asset ON watchlist(user_id, asset_id);
CREATE INDEX IF NOT EXISTS idx_portfolio_user ON portfolio(user_id);
CREATE INDEX IF NOT EXISTS idx_forum_posts_category ON forum_posts(category);
CREATE INDEX IF NOT EXISTS idx_forum_posts_created ON forum_posts(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_news_published ON news_articles(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_news_category ON news_articles(category);
CREATE INDEX IF NOT EXISTS idx_ai_insights_asset ON ai_insights(asset_symbol);
CREATE INDEX IF NOT EXISTS idx_ai_insights_type ON ai_insights(insight_type);
CREATE INDEX IF NOT EXISTS idx_trading_models_category ON trading_models(category);

-- Create full-text search indexes
CREATE INDEX IF NOT EXISTS idx_forum_posts_search ON forum_posts USING gin(to_tsvector('english', title || ' ' || content));
CREATE INDEX IF NOT EXISTS idx_news_search ON news_articles USING gin(to_tsvector('english', title || ' ' || content));
CREATE INDEX IF NOT EXISTS idx_assets_search ON assets USING gin(to_tsvector('english', name || ' ' || symbol || ' ' || COALESCE(description, '')));

-- Insert sample user (password is 'password123' hashed)
INSERT INTO users (username, email, hashed_password, full_name, role) VALUES
('admin', 'admin@finscope.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj3bp.Gm.F5e', 'Admin User', 'admin'),
('demo_user', 'demo@finscope.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj3bp.Gm.F5e', 'Demo User', 'user');

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO finscope_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO finscope_user;

-- Update sequences to start from appropriate values
SELECT setval('users_id_seq', (SELECT MAX(id) FROM users));
SELECT setval('assets_id_seq', (SELECT MAX(id) FROM assets));
SELECT setval('trading_models_id_seq', (SELECT MAX(id) FROM trading_models));
SELECT setval('forum_posts_id_seq', (SELECT MAX(id) FROM forum_posts));
SELECT setval('news_articles_id_seq', (SELECT MAX(id) FROM news_articles));
SELECT setval('ai_insights_id_seq', (SELECT MAX(id) FROM ai_insights));

COMMIT;