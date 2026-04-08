import React, { useCallback, useEffect, useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Dashboard from './components/Dashboard';
import Navbar from './components/Navbar';
import Login from './components/Login';
import Register from './components/Register';
import StockBackground from './components/StockBackground';
import { getScopeMonitorFeed, MonitorFeedItem } from './api/client';

const ScopeMonitorPage: React.FC = () => {
  const [items, setItems] = useState<MonitorFeedItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stream, setStream] = useState<'news' | 'markets' | 'events' | 'watchlist'>('news');

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await getScopeMonitorFeed(stream, 30);
      setItems(res.items || []);
    } catch (e: any) {
      setItems([]);
      setError(e?.message || 'Failed to load feed');
    } finally {
      setLoading(false);
    }
  }, [stream]);

  useEffect(() => {
    load();
  }, [load]);

  return (
    <div style={{ padding: '0 16px 16px 16px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <div>
          <h2 style={{ color: '#fff', margin: 0 }}>Scope Monitor</h2>
          <div style={{ color: '#a1a1aa', marginTop: 6 }}>Blank template for a global intelligence dashboard.</div>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button style={{ background: 'transparent', color: '#e5e7eb', border: '1px solid #374151', borderRadius: 6, padding: '8px 12px' }}>Settings</button>
          <button onClick={load} style={{ background: '#2563eb', color: '#fff', border: '1px solid #1d4ed8', borderRadius: 6, padding: '8px 12px', cursor: 'pointer' }}>Refresh</button>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '280px 1fr', gap: 12 }}>
        <div style={{ border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: 10, background: 'rgba(0,0,0,0.35)', padding: 12 }}>
          <div style={{ color: '#e5e7eb', fontWeight: 600, marginBottom: 10 }}>Streams</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              <button onClick={() => setStream('news')} style={{ width: '100%', textAlign: 'left', background: stream === 'news' ? '#111827' : 'transparent', color: '#e5e7eb', border: '1px solid #374151', borderRadius: 8, padding: '10px 12px', cursor: 'pointer' }}>News</button>
              <button onClick={() => setStream('markets')} style={{ width: '100%', textAlign: 'left', background: stream === 'markets' ? '#111827' : 'transparent', color: '#e5e7eb', border: '1px solid #374151', borderRadius: 8, padding: '10px 12px', cursor: 'pointer' }}>Markets</button>
              <button onClick={() => setStream('events')} style={{ width: '100%', textAlign: 'left', background: stream === 'events' ? '#111827' : 'transparent', color: '#e5e7eb', border: '1px solid #374151', borderRadius: 8, padding: '10px 12px', cursor: 'pointer' }}>Events</button>
              <button onClick={() => setStream('watchlist')} style={{ width: '100%', textAlign: 'left', background: stream === 'watchlist' ? '#111827' : 'transparent', color: '#e5e7eb', border: '1px solid #374151', borderRadius: 8, padding: '10px 12px', cursor: 'pointer' }}>Watchlist</button>
          </div>

          <div style={{ height: 12 }} />
          <div style={{ color: '#e5e7eb', fontWeight: 600, marginBottom: 10 }}>Filters</div>
          <div style={{ display: 'grid', gap: 8 }}>
            <select style={{ width: '100%', padding: 10, background: '#0b0f17', color: '#e5e7eb', border: '1px solid #374151', borderRadius: 8 }}>
              <option>All regions</option>
            </select>
            <select style={{ width: '100%', padding: 10, background: '#0b0f17', color: '#e5e7eb', border: '1px solid #374151', borderRadius: 8 }}>
              <option>All categories</option>
            </select>
            <input placeholder="Search…" style={{ width: '100%', padding: 10, background: '#0b0f17', color: '#e5e7eb', border: '1px solid #374151', borderRadius: 8 }} />
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateRows: '320px 1fr', gap: 12 }}>
          <div style={{ border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: 10, background: 'rgba(0,0,0,0.35)', padding: 12 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
              <div style={{ color: '#e5e7eb', fontWeight: 600 }}>Map</div>
              <div style={{ display: 'flex', gap: 8 }}>
                <button style={{ background: 'transparent', color: '#e5e7eb', border: '1px solid #374151', borderRadius: 6, padding: '6px 10px' }}>Layers</button>
                <button style={{ background: 'transparent', color: '#e5e7eb', border: '1px solid #374151', borderRadius: 6, padding: '6px 10px' }}>Reset</button>
              </div>
            </div>
            <div style={{ height: 260, borderRadius: 8, border: '1px dashed rgba(255,255,255,0.18)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#a1a1aa' }}>
              Map placeholder
            </div>
          </div>

          <div style={{ border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: 10, background: 'rgba(0,0,0,0.35)', padding: 12 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
              <div style={{ color: '#e5e7eb', fontWeight: 600 }}>Feed</div>
              <div style={{ color: '#a1a1aa', fontSize: '0.85rem' }}>{loading ? 'Loading…' : error ? error : `${items.length} items`}</div>
            </div>
            <div style={{ display: 'grid', gap: 10 }}>
              {items.map(it => (
                <div key={it.id} style={{ border: '1px solid rgba(255,255,255,0.1)', borderRadius: 10, padding: 12 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12 }}>
                    <div style={{ color: '#e5e7eb', fontWeight: 600 }}>{it.title}</div>
                    <div style={{ color: '#a1a1aa', fontSize: '0.85rem', whiteSpace: 'nowrap' }}>
                      {it.published_at ? new Date(it.published_at).toLocaleString() : ''}
                    </div>
                  </div>
                  <div style={{ color: '#a1a1aa', marginTop: 6 }}>{it.summary}</div>
                  <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 10, color: '#a1a1aa', fontSize: '0.85rem' }}>
                    <span>{it.kind}</span>
                    {it.category ? <span>· {it.category}</span> : null}
                    {it.region ? <span>· {it.region}</span> : null}
                    {it.source ? <span>· {it.source}</span> : null}
                    {it.url ? (
                      <a href={it.url} target="_blank" rel="noreferrer" style={{ color: '#8b5cf6', textDecoration: 'none' }}>
                        Open
                      </a>
                    ) : null}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const App: React.FC = () => {
  return (
    <Router>
      <div className="App">
        <StockBackground />
        <Navbar />
        <div style={{ position: 'relative', zIndex: 1 }}>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/scope-monitor" element={<ScopeMonitorPage />} />
            <Route path="/login" element={<Login />} />
            <Route path="/register" element={<Register />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
};

export default App;
