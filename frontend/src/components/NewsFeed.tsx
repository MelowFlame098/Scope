import React, { useEffect, useState } from 'react';
import { getLatestNews, NewsArticle } from '../api/client';

const NewsFeed: React.FC = () => {
  const [news, setNews] = useState<NewsArticle[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    const fetchNews = async () => {
      try {
        const data = await getLatestNews(10);
        setNews(data);
        setLoading(false);
      } catch (error) {
        console.error("Failed to fetch news", error);
        setLoading(false);
      }
    };

    fetchNews();
    const interval = setInterval(fetchNews, 30000); // Update every 30 seconds

    return () => clearInterval(interval);
  }, []);

  const getSentimentColor = (score: number) => {
    if (score > 0.2) return 'green';
    if (score < -0.2) return 'red';
    return 'gray';
  };

  if (loading) {
    return <div>Loading News...</div>;
  }

  return (
    <div style={{ padding: '20px', border: '1px solid #ddd', borderRadius: '8px', marginTop: '20px' }}>
      <h2>Market News & AI Insights</h2>
      <ul style={{ listStyle: 'none', padding: 0 }}>
        {news.map((article, index) => (
          <li key={index} style={{ marginBottom: '15px', paddingBottom: '15px', borderBottom: '1px solid #eee' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
              <a href={article.url} target="_blank" rel="noopener noreferrer" style={{ textDecoration: 'none', color: '#333', fontWeight: 'bold', fontSize: '1.1em' }}>
                {article.title}
              </a>
              <span style={{ 
                backgroundColor: getSentimentColor(article.sentiment), 
                color: 'white', 
                padding: '2px 8px', 
                borderRadius: '4px', 
                fontSize: '0.8em',
                marginLeft: '10px'
              }}>
                {article.sentiment > 0.2 ? 'Bullish' : article.sentiment < -0.2 ? 'Bearish' : 'Neutral'}
              </span>
            </div>
            <div style={{ fontSize: '0.9em', color: '#666', marginTop: '5px' }}>
              <span>{article.source}</span> • <span>{new Date(article.timestamp).toLocaleString()}</span>
            </div>
            <div style={{ marginTop: '5px' }}>
              {article.tags?.map(tag => (
                <span key={tag} style={{ backgroundColor: '#f0f0f0', color: '#555', padding: '2px 6px', borderRadius: '4px', fontSize: '0.75em', marginRight: '5px' }}>
                  {tag}
                </span>
              ))}
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default NewsFeed;
