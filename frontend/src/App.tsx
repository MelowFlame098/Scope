import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import 'leaflet/dist/leaflet.css';
import Dashboard from './components/Dashboard';
import Navbar from './components/Navbar';
import Login from './components/Login';
import Register from './components/Register';
import StockBackground from './components/StockBackground';
import { getScopeMonitorFeed, MonitorFeedItem } from './api/client';

const { CircleMarker, LayerGroup, LayersControl, MapContainer, Polyline, Popup, TileLayer, useMapEvents } = require('react-leaflet') as any;

type FeedStream = 'news' | 'markets' | 'events' | 'cyber' | 'watchlist';
type DateRange = '24h' | '7d' | '30d';

type LatLon = { lat: number; lon: number };

type Hotspot = {
  id: string;
  name: string;
  kind: 'conflict' | 'tension' | 'protest' | 'military' | 'critical_infra' | 'cyber' | 'market' | 'news';
  lat: number;
  lon: number;
  severity: number;
  details?: string;
};

type TravelRoute = {
  id: string;
  name: string;
  mode: 'ship' | 'flight';
  points: [number, number][];
};

type StrategicLine = {
  id: string;
  name: string;
  kind: 'cable' | 'pipeline';
  points: [number, number][];
};

const kmBetween = (a: LatLon, b: LatLon) => {
  const R = 6371;
  const dLat = ((b.lat - a.lat) * Math.PI) / 180;
  const dLon = ((b.lon - a.lon) * Math.PI) / 180;
  const lat1 = (a.lat * Math.PI) / 180;
  const lat2 = (b.lat * Math.PI) / 180;
  const s1 = Math.sin(dLat / 2);
  const s2 = Math.sin(dLon / 2);
  const h = s1 * s1 + Math.cos(lat1) * Math.cos(lat2) * s2 * s2;
  return 2 * R * Math.asin(Math.min(1, Math.sqrt(h)));
};

const clamp = (n: number, min: number, max: number) => Math.max(min, Math.min(max, n));

const STATIC_HOTSPOTS: Hotspot[] = [
  { id: 'war-ukraine', name: 'Ukraine War', kind: 'conflict', lat: 49.0, lon: 31.0, severity: 10, details: 'Active conflict and restrictions.' },
  { id: 'war-sudan', name: 'Sudan Conflict', kind: 'conflict', lat: 15.5, lon: 32.5, severity: 9, details: 'Active conflict and instability.' },
  { id: 'war-myanmar', name: 'Myanmar Civil War', kind: 'conflict', lat: 21.0, lon: 96.0, severity: 8, details: 'Active conflict and unrest.' },
  { id: 'tension-taiwan', name: 'Taiwan Strait Tensions', kind: 'tension', lat: 24.0, lon: 120.8, severity: 7, details: 'Strategic chokepoint; elevated geopolitical risk.' },
  { id: 'tension-gaza', name: 'Gaza / Israel Tensions', kind: 'tension', lat: 31.5, lon: 34.5, severity: 8, details: 'Regional instability and restrictions.' },
  { id: 'tension-iran-us', name: 'Iran–US Tensions', kind: 'tension', lat: 26.9, lon: 52.6, severity: 8, details: 'Elevated risk in Gulf; sanctions and maritime security concerns.' },
  { id: 'tension-iran-israel', name: 'Iran–Israel Shadow Conflict', kind: 'tension', lat: 32.0, lon: 53.0, severity: 7, details: 'Escalation risk, proxy activity, and regional spillover.' },
  { id: 'protest-tehran', name: 'Protest Activity', kind: 'protest', lat: 35.6892, lon: 51.389, severity: 6, details: 'Social unrest signals.' },
  { id: 'protest-lagos', name: 'Protest Activity', kind: 'protest', lat: 6.5244, lon: 3.3792, severity: 5, details: 'Social unrest signals.' },
  { id: 'mil-base-djibouti', name: 'Military Base Cluster (Djibouti)', kind: 'military', lat: 11.588, lon: 43.145, severity: 6, details: 'Strategic military logistics hub.' },
  { id: 'mil-base-guam', name: 'Military Base Cluster (Guam)', kind: 'military', lat: 13.4443, lon: 144.7937, severity: 6, details: 'Pacific projection and air/naval facilities.' },
  { id: 'mil-base-ramstein', name: 'Air Base (Ramstein)', kind: 'military', lat: 49.4369, lon: 7.6003, severity: 4, details: 'Major air logistics node.' },
  { id: 'mil-fleet-scs', name: 'Naval Activity (South China Sea)', kind: 'military', lat: 13.5, lon: 114.0, severity: 7, details: 'Elevated naval activity and disputes.' },
  { id: 'mil-fleet-blacksea', name: 'Naval Activity (Black Sea)', kind: 'military', lat: 44.5, lon: 33.5, severity: 7, details: 'Restricted waters and conflict adjacency.' },
  { id: 'cyber-apt-eurasia', name: 'APT Activity (OSINT)', kind: 'cyber', lat: 55.75, lon: 37.62, severity: 6, details: 'Regional cyber threat concentration.' },
  { id: 'cyber-apt-asia', name: 'APT Activity (OSINT)', kind: 'cyber', lat: 39.9042, lon: 116.4074, severity: 6, details: 'Regional cyber threat concentration.' },
  { id: 'infra-cable-ashburn', name: 'Data Center Corridor (Ashburn, VA)', kind: 'critical_infra', lat: 39.0438, lon: -77.4874, severity: 5, details: 'High concentration of cloud and backbone infrastructure.' },
  { id: 'infra-cable-marseille', name: 'Subsea Cable Landing (Marseille)', kind: 'critical_infra', lat: 43.2965, lon: 5.3698, severity: 4, details: 'Major European subsea cable landing hub.' },
  { id: 'infra-cable-singapore', name: 'Subsea Cable Landing (Singapore)', kind: 'critical_infra', lat: 1.3521, lon: 103.8198, severity: 4, details: 'APAC subsea cable and exchange hub.' },
  { id: 'infra-nuclear-zap', name: 'Nuclear Facility (Zaporizhzhia)', kind: 'critical_infra', lat: 47.511, lon: 34.585, severity: 7, details: 'Strategic energy infrastructure; conflict sensitivity.' },
  { id: 'infra-nuclear-fukushima', name: 'Nuclear Facility (Fukushima)', kind: 'critical_infra', lat: 37.421, lon: 141.032, severity: 4, details: 'Strategic energy infrastructure.' },
  { id: 'infra-ground-svalbard', name: 'Satellite Ground Station (Svalbard)', kind: 'critical_infra', lat: 78.2298, lon: 15.4078, severity: 5, details: 'Polar downlink and EO satellite relay.' },
  { id: 'infra-minerals-congo', name: 'Critical Minerals (DRC Copper/Cobalt)', kind: 'critical_infra', lat: -10.5, lon: 26.5, severity: 5, details: 'High strategic mineral concentration.' },
  { id: 'infra-port-rotterdam', name: 'Major Port (Rotterdam)', kind: 'critical_infra', lat: 51.95, lon: 4.14, severity: 3, details: 'Major European port and logistics hub.' },
  { id: 'infra-port-shanghai', name: 'Major Port (Shanghai)', kind: 'critical_infra', lat: 31.2304, lon: 121.4737, severity: 3, details: 'Major global container port.' },
  { id: 'infra-port-jebel', name: 'Major Port (Jebel Ali)', kind: 'critical_infra', lat: 25.011, lon: 55.061, severity: 4, details: 'Strategic regional port and transshipment hub.' },
  { id: 'infra-hormuz', name: 'Strait of Hormuz', kind: 'critical_infra', lat: 26.56, lon: 56.25, severity: 7, details: 'Energy chokepoint.' },
  { id: 'infra-bab', name: 'Bab el-Mandeb', kind: 'critical_infra', lat: 12.62, lon: 43.33, severity: 7, details: 'Suez access chokepoint.' },
  { id: 'infra-suez', name: 'Suez Canal', kind: 'critical_infra', lat: 30.0, lon: 32.58, severity: 6, details: 'Global shipping chokepoint.' },
  { id: 'infra-malacca', name: 'Strait of Malacca', kind: 'critical_infra', lat: 2.5, lon: 101.0, severity: 6, details: 'Asia shipping chokepoint.' },
  { id: 'infra-panama', name: 'Panama Canal', kind: 'critical_infra', lat: 9.08, lon: -79.68, severity: 5, details: 'Interocean transit chokepoint.' },
  { id: 'infra-gibraltar', name: 'Strait of Gibraltar', kind: 'critical_infra', lat: 35.98, lon: -5.6, severity: 5, details: 'Atlantic–Mediterranean chokepoint.' },
];

const STATIC_ROUTES: TravelRoute[] = [
  { id: 'ship-asia-europe', name: 'Asia ↔ Europe (Suez)', mode: 'ship', points: [[1.29, 103.85], [12.62, 43.33], [30.0, 32.58], [35.98, -5.6], [51.5, 0.0]] },
  { id: 'ship-gulf-asia', name: 'Gulf ↔ Asia (Hormuz → Malacca)', mode: 'ship', points: [[26.56, 56.25], [12.62, 43.33], [1.29, 103.85], [2.5, 101.0]] },
  { id: 'ship-atlantic-pacific', name: 'Atlantic ↔ Pacific (Panama)', mode: 'ship', points: [[25.77, -80.19], [9.08, -79.68], [34.05, -118.25]] },
  { id: 'flight-natl', name: 'Transatlantic Corridor', mode: 'flight', points: [[40.71, -74.0], [51.5, 0.0]] },
  { id: 'flight-eurasia', name: 'Europe ↔ East Asia Corridor', mode: 'flight', points: [[48.86, 2.35], [55.75, 37.62], [39.9, 116.4], [35.68, 139.69]] },
  { id: 'flight-me-apac', name: 'Middle East ↔ APAC Corridor', mode: 'flight', points: [[25.20, 55.27], [19.08, 72.88], [1.29, 103.85]] },
];

const STATIC_INFRA_LINES: StrategicLine[] = [
  { id: 'cable-transatlantic', name: 'Subsea Cable Corridor (North Atlantic)', kind: 'cable', points: [[40.71, -74.0], [48.86, 2.35], [51.5, 0.0]] },
  { id: 'cable-eu-me-apac', name: 'Subsea Cable Corridor (Europe → ME → APAC)', kind: 'cable', points: [[43.2965, 5.3698], [30.0, 32.58], [25.20, 55.27], [19.08, 72.88], [1.3521, 103.8198]] },
  { id: 'cable-pacific', name: 'Subsea Cable Corridor (Transpacific)', kind: 'cable', points: [[35.6762, 139.6503], [21.3069, -157.8583], [34.0522, -118.2437]] },
  { id: 'pipe-northsea', name: 'Pipeline Corridor (North Sea)', kind: 'pipeline', points: [[54.6, 8.5], [55.7, 12.6], [57.0, 18.0]] },
  { id: 'pipe-europe', name: 'Pipeline Corridor (Eastern Europe)', kind: 'pipeline', points: [[55.75, 37.62], [52.23, 21.01], [50.45, 30.52], [48.2, 16.37]] },
  { id: 'pipe-middleeast', name: 'Pipeline Corridor (Gulf)', kind: 'pipeline', points: [[26.56, 56.25], [25.2, 55.27], [24.71, 46.67]] },
];

const hashToUnit = (s: string) => {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return ((h >>> 0) % 10000) / 10000;
};

const offsetLatLon = (lat: number, lon: number, idx: number, count: number, seed: number) => {
  if (count <= 1) return { lat, lon };
  const slots = 8;
  const ring = Math.floor(idx / slots) + 1;
  const pos = idx % slots;
  const angle = (2 * Math.PI * pos) / slots + seed * 0.8;
  const dist = ring * 0.008;
  const dLat = dist * Math.cos(angle);
  const dLon = (dist * Math.sin(angle)) / Math.max(0.3, Math.cos((lat * Math.PI) / 180));
  return { lat: lat + dLat, lon: lon + dLon };
};

type CountryInfo = {
  name: string;
  official: string;
  cca2: string;
  capital: string;
  region: string;
  subregion: string;
  population: number;
  flagPng: string;
};

const ScopeMonitorPage: React.FC = () => {
  const [newsItems, setNewsItems] = useState<MonitorFeedItem[]>([]);
  const [eventItems, setEventItems] = useState<MonitorFeedItem[]>([]);
  const [cyberItems, setCyberItems] = useState<MonitorFeedItem[]>([]);
  const [watchlistItems, setWatchlistItems] = useState<MonitorFeedItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdatedAt, setLastUpdatedAt] = useState<string | null>(null);

  const [keyword, setKeyword] = useState('');
  const [dateRange, setDateRange] = useState<DateRange>('30d');
  const [minSeverity, setMinSeverity] = useState(0);

  const [showNews, setShowNews] = useState(true);
  const [showEarthquakes, setShowEarthquakes] = useState(true);
  const [showDisasters, setShowDisasters] = useState(true);
  const [showCyber, setShowCyber] = useState(true);
  const [showGdelt, setShowGdelt] = useState(true);
  const [showAviation, setShowAviation] = useState(true);
  const [showMaritime, setShowMaritime] = useState(true);
  const [showSummaries, setShowSummaries] = useState(true);
  const [alertsEnabled, setAlertsEnabled] = useState(true);
  const [mapStyle, setMapStyle] = useState<'dark' | 'light' | 'satellite'>('dark');

  const [weatherLat, setWeatherLat] = useState(40.7128);
  const [weatherLon, setWeatherLon] = useState(-74.0060);
  const [weather, setWeather] = useState<{ temperature: number; windspeed: number; time: string } | null>(null);
  const [weatherError, setWeatherError] = useState<string | null>(null);

  const [countryQuery, setCountryQuery] = useState('United States');
  const [countryInfo, setCountryInfo] = useState<CountryInfo | null>(null);
  const [countryLoading, setCountryLoading] = useState(false);
  const [countryError, setCountryError] = useState<string | null>(null);

  const [selectedPoint, setSelectedPoint] = useState<LatLon | null>(null);
  const [selectedWeather, setSelectedWeather] = useState<{ temperature: number; windspeed: number; time: string } | null>(null);
  const [selectedPlace, setSelectedPlace] = useState<{ label: string; countryName: string; countryCode: string } | null>(null);
  const [selectedPlaceError, setSelectedPlaceError] = useState<string | null>(null);

  const loadAll = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const days = dateRange === '24h' ? 1 : dateRange === '7d' ? 7 : 30;
      const afterISO = new Date(Date.now() - days * 24 * 60 * 60 * 1000).toISOString();
      const [news, events, cyber, watchlist] = await Promise.all([
        getScopeMonitorFeed('news', 300, afterISO),
        getScopeMonitorFeed('events', 3000, afterISO),
        getScopeMonitorFeed('cyber', 2000, afterISO),
        getScopeMonitorFeed('watchlist', 5000, afterISO),
      ]);

      setNewsItems(news.items || []);
      setEventItems(events.items || []);
      setCyberItems(cyber.items || []);
      setWatchlistItems(watchlist.items || []);

      const candidates = [news.generated_at, events.generated_at, cyber.generated_at, watchlist.generated_at].filter(Boolean);
      candidates.sort((a, b) => new Date(b).getTime() - new Date(a).getTime());
      setLastUpdatedAt(candidates[0] || null);
    } catch (e: any) {
      setError(e?.message || 'Failed to load feed');
    } finally {
      setLoading(false);
    }
  }, [dateRange]);

  const loadWeather = useCallback(async () => {
    setWeatherError(null);
    try {
      const url = `https://api.open-meteo.com/v1/forecast?latitude=${weatherLat}&longitude=${weatherLon}&current_weather=true`;
      const res = await fetch(url);
      if (!res.ok) {
        throw new Error(`Open-Meteo error ${res.status}`);
      }
      const data = await res.json();
      const cw = data?.current_weather;
      if (!cw) {
        setWeather(null);
        return;
      }
      setWeather({ temperature: cw.temperature, windspeed: cw.windspeed, time: cw.time });
    } catch (e: any) {
      setWeather(null);
      setWeatherError(e?.message || 'Failed to load weather');
    }
  }, [weatherLat, weatherLon]);

  const loadCountry = useCallback(async () => {
    const q = countryQuery.trim();
    if (!q) {
      setCountryInfo(null);
      return;
    }
    setCountryLoading(true);
    setCountryError(null);
    try {
      const url = `https://restcountries.com/v3.1/name/${encodeURIComponent(q)}?fields=name,cca2,capital,region,subregion,population,flags`;
      const res = await fetch(url);
      if (!res.ok) {
        throw new Error(`REST Countries error ${res.status}`);
      }
      const data = await res.json();
      const c = Array.isArray(data) ? data[0] : null;
      if (!c) {
        throw new Error('No country match');
      }
      setCountryInfo({
        name: c?.name?.common || q,
        official: c?.name?.official || q,
        cca2: c?.cca2 || '',
        capital: Array.isArray(c?.capital) ? (c.capital[0] || '') : '',
        region: c?.region || '',
        subregion: c?.subregion || '',
        population: typeof c?.population === 'number' ? c.population : 0,
        flagPng: c?.flags?.png || '',
      });
    } catch (e: any) {
      setCountryInfo(null);
      setCountryError(e?.message || 'Failed to load country info');
    } finally {
      setCountryLoading(false);
    }
  }, [countryQuery]);

  useEffect(() => {
    loadAll();
    loadWeather();
    const id = window.setInterval(() => {
      loadAll();
      loadWeather();
    }, 2 * 60 * 1000);
    return () => window.clearInterval(id);
  }, [loadAll, loadWeather]);

  useEffect(() => {
    loadCountry();
  }, [loadCountry]);

  useEffect(() => {
    if (!selectedPoint) {
      setSelectedWeather(null);
      setSelectedPlace(null);
      setSelectedPlaceError(null);
      return;
    }

    (async () => {
      try {
        const url = `https://api.open-meteo.com/v1/forecast?latitude=${selectedPoint.lat}&longitude=${selectedPoint.lon}&current_weather=true`;
        const res = await fetch(url);
        if (!res.ok) throw new Error(`Open-Meteo error ${res.status}`);
        const data = await res.json();
        const cw = data?.current_weather;
        if (cw) {
          setSelectedWeather({ temperature: cw.temperature, windspeed: cw.windspeed, time: cw.time });
        } else {
          setSelectedWeather(null);
        }
      } catch {
        setSelectedWeather(null);
      }
    })();

    (async () => {
      setSelectedPlaceError(null);
      try {
        const url = `https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat=${selectedPoint.lat}&lon=${selectedPoint.lon}`;
        const res = await fetch(url, { headers: { Accept: 'application/json' } });
        if (!res.ok) throw new Error(`Reverse geocode error ${res.status}`);
        const data = await res.json();
        const address = data?.address || {};
        const countryName = String(address?.country || '');
        const countryCode = String(address?.country_code || '').toUpperCase();
        const label = String(data?.display_name || countryName || `${selectedPoint.lat.toFixed(3)}, ${selectedPoint.lon.toFixed(3)}`);
        setSelectedPlace({ label, countryName, countryCode });
      } catch (e: any) {
        setSelectedPlace(null);
        setSelectedPlaceError(e?.message || 'Failed to resolve region');
      }
    })();
  }, [selectedPoint]);

  const allowedKind = useCallback(
    (kind: string) => {
      if (kind === 'news') return showNews;
      if (kind === 'earthquake') return showEarthquakes;
      if (kind === 'disaster') return showDisasters;
      if (kind === 'cyber') return showCyber;
      if (kind === 'gdelt') return showGdelt;
      if (kind === 'aviation') return showAviation;
      return true;
    },
    [showAviation, showCyber, showDisasters, showEarthquakes, showGdelt, showNews]
  );

  const rangeCutoff = useMemo(() => {
    const now = Date.now();
    if (dateRange === '24h') return now - 24 * 60 * 60 * 1000;
    if (dateRange === '7d') return now - 7 * 24 * 60 * 60 * 1000;
    return now - 30 * 24 * 60 * 60 * 1000;
  }, [dateRange]);

  const liveNewsCutoff = useMemo(() => Date.now() - 30 * 24 * 60 * 60 * 1000, []);

  type NewsGeoScope = 'global' | 'continent' | 'regional' | 'local';
  const [newsGeoScope, setNewsGeoScope] = useState<NewsGeoScope>('global');
  const [newsContinent, setNewsContinent] = useState<'Africa' | 'Asia' | 'Europe' | 'North America' | 'South America' | 'Oceania' | 'Middle East'>('North America');
  const [newsRegionalCountries, setNewsRegionalCountries] = useState('Germany, France, Italy');
  const [newsLocalCountry, setNewsLocalCountry] = useState('United States');

  useEffect(() => {
    if (!countryInfo?.name) return;
    if (newsGeoScope === 'local') setNewsLocalCountry(countryInfo.name);
  }, [countryInfo?.name, newsGeoScope]);

  const liveNewsItems = useMemo(() => {
    const normalize = (s: string) =>
      s
        .toLowerCase()
        .replace(/[\u2019']/g, '')
        .replace(/[^a-z0-9\s]/g, ' ')
        .replace(/\s+/g, ' ')
        .trim();

    const allowedCategories = new Set(['finance', 'macro', 'technology', 'science', 'markets', 'business', 'research']);
    const financeTechKeywords = [
      'inflation',
      'cpi',
      'gdp',
      'rates',
      'interest rate',
      'central bank',
      'fed',
      'ecb',
      'boj',
      'pbo',
      'treasury',
      'bond',
      'yield',
      'stock',
      'equity',
      'ipo',
      'earnings',
      'investment',
      'fund',
      'bank',
      'fintech',
      'currency',
      'fx',
      'forex',
      'dollar',
      'euro',
      'yen',
      'pound',
      'ai',
      'semiconductor',
      'chip',
      'research',
      'study',
      'trial',
      'breakthrough',
      'quantum',
      'biotech',
      'vaccine',
    ];

    const continentTerms: Record<string, string[]> = {
      Africa: ['africa', 'nigeria', 'south africa', 'egypt', 'kenya', 'ethiopia', 'algeria', 'morocco', 'ghana'],
      Asia: ['asia', 'china', 'beijing', 'japan', 'tokyo', 'india', 'delhi', 'south korea', 'seoul', 'taiwan', 'singapore', 'hong kong', 'indonesia', 'vietnam'],
      Europe: ['europe', 'eu', 'eurozone', 'uk', 'united kingdom', 'london', 'france', 'paris', 'germany', 'berlin', 'italy', 'rome', 'spain', 'madrid', 'poland', 'ukraine'],
      'North America': ['north america', 'united states', 'u s', 'us', 'usa', 'canada', 'mexico', 'washington', 'new york'],
      'South America': ['south america', 'brazil', 'argentina', 'chile', 'colombia', 'peru'],
      Oceania: ['oceania', 'australia', 'sydney', 'melbourne', 'new zealand', 'wellington'],
      'Middle East': ['middle east', 'gulf', 'iran', 'tehran', 'israel', 'saudi', 'riyadh', 'uae', 'dubai', 'qatar', 'iraq', 'turkey'],
    };

    const parseCountries = (raw: string) =>
      raw
        .split(',')
        .map(s => normalize(s))
        .map(s => s.replace(/\bthe\b/g, '').trim())
        .filter(Boolean)
        .slice(0, 24);

    const scopeTerms = (() => {
      if (newsGeoScope === 'continent') return (continentTerms[newsContinent] || []).map(normalize);
      if (newsGeoScope === 'regional') return parseCountries(newsRegionalCountries);
      if (newsGeoScope === 'local') return parseCountries(newsLocalCountry);
      return [];
    })();

    const matchesScope = (hay: string) => {
      if (scopeTerms.length === 0) return true;
      for (const t of scopeTerms) {
        if (!t) continue;
        if (hay.includes(` ${t} `) || hay.startsWith(`${t} `) || hay.endsWith(` ${t}`) || hay.includes(t)) return true;
      }
      return false;
    };

    const matchesTopic = (it: MonitorFeedItem) => {
      const cat = normalize(it.category || '');
      if (cat && allowedCategories.has(cat)) return true;
      const hay = normalize(`${it.title || ''} ${it.summary || ''}`);
      return financeTechKeywords.some(k => hay.includes(normalize(k)));
    };

    return newsItems
      .filter(it => it.kind === 'news')
      .filter(it => {
        const t = new Date(it.published_at).getTime();
        if (!Number.isNaN(t) && t < liveNewsCutoff) return false;
        return true;
      })
      .filter(it => {
        const hay = normalize(`${it.title || ''} ${it.summary || ''} ${it.source || ''} ${it.region || ''} ${it.category || ''}`);
        return matchesScope(hay);
      })
      .filter(matchesTopic)
      .sort((a, b) => new Date(b.published_at).getTime() - new Date(a.published_at).getTime())
      .slice(0, 120);
  }, [liveNewsCutoff, newsContinent, newsGeoScope, newsItems, newsLocalCountry, newsRegionalCountries]);

  const geoItems = useMemo(() => {
    const all = [...watchlistItems, ...eventItems, ...cyberItems];
    const seen = new Set<string>();
    const out: MonitorFeedItem[] = [];
    for (const it of all) {
      if (!it?.id) continue;
      if (seen.has(it.id)) continue;
      seen.add(it.id);
      if (!it.geo || typeof it.geo.lat !== 'number' || typeof it.geo.lon !== 'number') continue;
      out.push(it);
    }
    return out;
  }, [cyberItems, eventItems, watchlistItems]);

  const mapPoints = useMemo(() => {
    const q = keyword.trim().toLowerCase();
    const base = geoItems
      .filter(it => allowedKind(it.kind))
      .filter(it => (it.severity || 0) >= minSeverity)
      .filter(it => {
        if (it.published_at) {
          const t = new Date(it.published_at).getTime();
          if (!Number.isNaN(t) && t < rangeCutoff) return false;
        }
        return true;
      })
      .filter(it => {
        if (!q) return true;
        const hay = `${it.title} ${it.summary} ${it.source || ''} ${it.region || ''} ${it.category || ''}`.toLowerCase();
        return hay.includes(q);
      });
    const groups = new Map<string, MonitorFeedItem[]>();
    for (const it of base) {
      const lat = it.geo!.lat;
      const lon = it.geo!.lon;
      const k = `${it.kind}:${Math.round(lat * 100) / 100}:${Math.round(lon * 100) / 100}`;
      const arr = groups.get(k);
      if (arr) arr.push(it);
      else groups.set(k, [it]);
    }
    const out: MonitorFeedItem[] = [];
    groups.forEach((arr: MonitorFeedItem[]) => {
      arr.sort((a: MonitorFeedItem, b: MonitorFeedItem) => (b.severity || 0) - (a.severity || 0));
      for (let i = 0; i < arr.length; i++) {
        const it = arr[i];
        const seed = hashToUnit(it.id);
        const p = offsetLatLon(it.geo!.lat, it.geo!.lon, i, arr.length, seed);
        out.push({ ...it, geo: { lat: p.lat, lon: p.lon } });
      }
    });
    return out;
  }, [allowedKind, geoItems, keyword, minSeverity, rangeCutoff]);

  const aviationPoints = useMemo(() => {
    return geoItems
      .filter(it => it.kind === 'aviation')
      .filter(() => showAviation)
      .filter(it => (it.severity || 0) >= minSeverity);
  }, [geoItems, minSeverity, showAviation]);

  const maritimePoints = useMemo(() => {
    const maritimeCats = ['hurricane', 'cyclone', 'storm', 'flood', 'sea', 'marine', 'typhoon'];
    return geoItems
      .filter(it => it.kind === 'disaster' && typeof it.category === 'string' && maritimeCats.some(k => it.category!.toLowerCase().includes(k)))
      .filter(() => showMaritime)
      .filter(it => (it.severity || 0) >= minSeverity);
  }, [geoItems, minSeverity, showMaritime]);

  const conflictHotspots = useMemo(() => {
    return STATIC_HOTSPOTS.filter(h => h.kind === 'conflict' || h.kind === 'tension');
  }, []);

  const unrestHotspots = useMemo(() => {
    return STATIC_HOTSPOTS.filter(h => h.kind === 'protest');
  }, []);

  const militaryHotspots = useMemo(() => {
    return STATIC_HOTSPOTS.filter(h => h.kind === 'military');
  }, []);

  const infraHotspots = useMemo(() => {
    return STATIC_HOTSPOTS.filter(h => h.kind === 'critical_infra');
  }, []);

  const cyberHotspots = useMemo(() => {
    return STATIC_HOTSPOTS.filter(h => h.kind === 'cyber');
  }, []);

  const marketHotspots = useMemo(() => {
    const hubs: Array<{ id: string; name: string; lat: number; lon: number; keywords: string[] }> = [
      { id: 'mkt-nyc', name: 'New York', lat: 40.7128, lon: -74.006, keywords: ['us', 'fed', 'treasury', 'nasdaq', 's&p', 'dow', 'nyse', 'wall street'] },
      { id: 'mkt-lon', name: 'London', lat: 51.5074, lon: -0.1278, keywords: ['uk', 'boe', 'ftse', 'london', 'europe'] },
      { id: 'mkt-fra', name: 'Frankfurt', lat: 50.1109, lon: 8.6821, keywords: ['ecb', 'eurozone', 'bund', 'germany'] },
      { id: 'mkt-dub', name: 'Dubai', lat: 25.2048, lon: 55.2708, keywords: ['opec', 'oil', 'gulf', 'middle east'] },
      { id: 'mkt-sin', name: 'Singapore', lat: 1.3521, lon: 103.8198, keywords: ['singapore', 'shipping', 'malacca', 'asia'] },
      { id: 'mkt-hkg', name: 'Hong Kong', lat: 22.3193, lon: 114.1694, keywords: ['china', 'hong kong', 'hang seng', 'yuan'] },
      { id: 'mkt-tok', name: 'Tokyo', lat: 35.6762, lon: 139.6503, keywords: ['japan', 'boj', 'nikkei', 'yen'] },
    ];

    const corpus = [...newsItems];
    const text = corpus.map(it => `${it.title} ${it.summary} ${it.region || ''} ${it.category || ''}`).join(' ').toLowerCase();
    return hubs
      .map(h => {
        const score = h.keywords.reduce((acc, k) => acc + (text.includes(k) ? 1 : 0), 0);
        return { id: h.id, name: `${h.name} Market Pulse`, kind: 'market' as const, lat: h.lat, lon: h.lon, severity: clamp(score * 2, 0, 10), details: 'Trending news that may affect markets.' };
      })
      .filter(h => h.severity > 0);
  }, [newsItems]);

  const breakingNewsHotspots = useMemo(() => {
    const hubs: Array<{ id: string; name: string; lat: number; lon: number; keywords: string[]; categories: string[] }> = [
      { id: 'news-wdc', name: 'Washington', lat: 38.9072, lon: -77.0369, categories: ['geopolitics', 'macro', 'finance'], keywords: ['white house', 'pentagon', 'congress', 'treasury', 'sanction', 'tariff', 'trade', 'fed', 'sec', 'doj', 'us '] },
      { id: 'news-brx', name: 'Brussels', lat: 50.8503, lon: 4.3517, categories: ['geopolitics', 'macro', 'finance'], keywords: ['eu', 'european union', 'brussels', 'ecb', 'nato', 'sanction'] },
      { id: 'news-bjs', name: 'Beijing', lat: 39.9042, lon: 116.4074, categories: ['geopolitics', 'macro', 'technology'], keywords: ['china', 'beijing', 'pbo', 'yuan', 'taiwan', 'chip', 'export control', 'rare earth'] },
      { id: 'news-mow', name: 'Moscow', lat: 55.7558, lon: 37.6173, categories: ['geopolitics', 'macro'], keywords: ['russia', 'moscow', 'kremlin', 'ukraine'] },
      { id: 'news-thr', name: 'Tehran', lat: 35.6892, lon: 51.389, categories: ['geopolitics', 'natural resources'], keywords: ['iran', 'tehran', 'hormuz', 'sanction', 'missile', 'ceasefire'] },
      { id: 'news-jrs', name: 'Levant', lat: 31.7683, lon: 35.2137, categories: ['geopolitics', 'natural resources'], keywords: ['israel', 'gaza', 'hamas', 'hezbollah', 'lebanon', 'syria'] },
      { id: 'news-ruh', name: 'Riyadh', lat: 24.7136, lon: 46.6753, categories: ['natural resources', 'macro'], keywords: ['opec', 'oil', 'brent', 'wti', 'saudi', 'production cut'] },
      { id: 'news-sf', name: 'Silicon Valley', lat: 37.3875, lon: -122.0575, categories: ['technology', 'finance'], keywords: ['ai', 'chip', 'semiconductor', 'nvidia', 'openai', 'microsoft', 'apple', 'google', 'amazon', 'quantum', 'breakthrough'] },
      { id: 'news-sin', name: 'Supply Chain', lat: 1.3521, lon: 103.8198, categories: ['natural resources', 'macro', 'geopolitics'], keywords: ['shipping', 'container', 'port', 'strait', 'malacca', 'supply chain'] },
    ];

    const wantedCats = new Set(['geopolitics', 'technology', 'macro', 'finance', 'natural resources', 'crypto']);
    const corpus = newsItems
      .filter(it => {
        const t = new Date(it.published_at).getTime();
        if (!Number.isNaN(t) && t < rangeCutoff) return false;
        const impact = it.relevance ?? 0;
        if (impact >= 12) return true;
        const cat = (it.category || '').toLowerCase();
        return wantedCats.has(cat);
      })
      .slice(0, 600);

    const compact = (s: string) => s.replace(/\s+/g, ' ').trim();
    const short = (s: string, n: number) => (s.length > n ? `${s.slice(0, n - 1)}…` : s);

    return hubs
      .map(h => {
        let weight = 0;
        const matchedTitles: string[] = [];
        for (const it of corpus) {
          const text = `${it.title} ${it.summary || ''} ${it.category || ''} ${it.region || ''} ${it.source || ''}`.toLowerCase();
          const cat = (it.category || '').toLowerCase();
          const hit = h.keywords.some(k => text.includes(k)) || (cat && h.categories.includes(cat));
          if (!hit) continue;

          let w = it.relevance ?? 0;
          if (!w) w = it.source === 'Finviz Aggregated' ? 2 : 8;
          if (cat && wantedCats.has(cat)) w += 2;
          w = clamp(w, 0, 20);

          weight += w;
          if (matchedTitles.length < 3) matchedTitles.push(short(compact(it.title || ''), 110));
        }
        const severity = clamp(Math.round((weight / 20) * 10), 0, 10);
        return {
          id: h.id,
          name: `${h.name} Breaking`,
          kind: 'news' as const,
          lat: h.lat,
          lon: h.lon,
          severity,
          details: matchedTitles.length ? matchedTitles.join(' • ') : 'High-impact headlines cluster',
        };
      })
      .filter(h => h.severity >= 3);
  }, [newsItems, rangeCutoff]);

  const routeStatuses = useMemo(() => {
    const riskAt = (p: LatLon) => {
      let minKm = Number.POSITIVE_INFINITY;
      let maxSev = 0;
      for (const h of conflictHotspots) {
        const d = kmBetween(p, { lat: h.lat, lon: h.lon });
        if (d < minKm) minKm = d;
        if (h.severity > maxSev) maxSev = h.severity;
      }
      for (const h of unrestHotspots) {
        const d = kmBetween(p, { lat: h.lat, lon: h.lon });
        if (d < minKm) minKm = d;
        maxSev = Math.max(maxSev, Math.max(0, h.severity - 2));
      }
      return { minKm, maxSev };
    };

    const scoreRoute = (r: TravelRoute) => {
      let worst = { minKm: Number.POSITIVE_INFINITY, maxSev: 0 };
      for (const [lat, lon] of r.points) {
        const cur = riskAt({ lat, lon });
        if (cur.minKm < worst.minKm) worst = cur;
      }
      const risk = worst.maxSev * (1 / clamp(worst.minKm / 800, 0.2, 10));
      const status = risk >= 18 ? 'Closed' : risk >= 10 ? 'Restricted' : risk >= 6 ? 'Caution' : 'Normal';
      return { status, risk: clamp(risk, 0, 100), minKm: worst.minKm };
    };

    const out: Record<string, { status: string; risk: number; minKm: number }> = {};
    for (const r of STATIC_ROUTES) out[r.id] = scoreRoute(r);
    return out;
  }, [conflictHotspots, unrestHotspots]);

  const infraLineStatuses = useMemo(() => {
    const riskAt = (p: LatLon) => {
      let minKm = Number.POSITIVE_INFINITY;
      let maxSev = 0;
      for (const h of conflictHotspots) {
        const d = kmBetween(p, { lat: h.lat, lon: h.lon });
        if (d < minKm) minKm = d;
        if (h.severity > maxSev) maxSev = h.severity;
      }
      for (const h of unrestHotspots) {
        const d = kmBetween(p, { lat: h.lat, lon: h.lon });
        if (d < minKm) minKm = d;
        maxSev = Math.max(maxSev, Math.max(0, h.severity - 3));
      }
      return { minKm, maxSev };
    };

    const scoreLine = (l: StrategicLine) => {
      let worst = { minKm: Number.POSITIVE_INFINITY, maxSev: 0 };
      for (const [lat, lon] of l.points) {
        const cur = riskAt({ lat, lon });
        if (cur.minKm < worst.minKm) worst = cur;
      }
      const risk = worst.maxSev * (1 / clamp(worst.minKm / 900, 0.25, 10));
      const status = risk >= 18 ? 'Disrupted' : risk >= 10 ? 'At Risk' : risk >= 6 ? 'Watch' : 'Normal';
      return { status, risk: clamp(risk, 0, 100), minKm: worst.minKm };
    };

    const out: Record<string, { status: string; risk: number; minKm: number }> = {};
    for (const l of STATIC_INFRA_LINES) out[l.id] = scoreLine(l);
    return out;
  }, [conflictHotspots, unrestHotspots]);

  const severityColor = (s: number) => {
    if (s >= 7) return '#ef4444';
    if (s >= 4) return '#f59e0b';
    return '#22c55e';
  };

  const highSeverityAlerts = useMemo(() => {
    return watchlistItems
      .filter(it => (it.severity || 0) >= 6)
      .sort((a, b) => new Date(b.published_at).getTime() - new Date(a.published_at).getTime())
      .slice(0, 12);
  }, [watchlistItems]);

  const countryMentions = useMemo(() => {
    const needle = (countryInfo?.name || countryQuery).trim().toLowerCase();
    if (!needle) return 0;
    return watchlistItems.filter(it => {
      const hay = `${it.title} ${it.summary} ${it.region || ''}`.toLowerCase();
      return hay.includes(needle);
    }).length;
  }, [countryInfo?.name, countryQuery, watchlistItems]);

  const selectedBrief = useMemo(() => {
    if (!selectedPoint) return null;
    const nearbyConflict = conflictHotspots
      .map(h => ({ h, km: kmBetween(selectedPoint, { lat: h.lat, lon: h.lon }) }))
      .filter(x => x.km <= 1200)
      .sort((a, b) => a.km - b.km)
      .slice(0, 8);

    const nearbyUnrest = unrestHotspots
      .map(h => ({ h, km: kmBetween(selectedPoint, { lat: h.lat, lon: h.lon }) }))
      .filter(x => x.km <= 1200)
      .sort((a, b) => a.km - b.km)
      .slice(0, 8);

    const nearbyMilitary = militaryHotspots
      .map(h => ({ h, km: kmBetween(selectedPoint, { lat: h.lat, lon: h.lon }) }))
      .filter(x => x.km <= 1500)
      .sort((a, b) => a.km - b.km)
      .slice(0, 8);

    const nearbyInfra = infraHotspots
      .map(h => ({ h, km: kmBetween(selectedPoint, { lat: h.lat, lon: h.lon }) }))
      .filter(x => x.km <= 1500)
      .sort((a, b) => a.km - b.km)
      .slice(0, 8);

    const nearbyDisasters = mapPoints
      .filter(it => it.kind === 'earthquake' || it.kind === 'disaster')
      .map(it => ({ it, km: kmBetween(selectedPoint, { lat: it.geo!.lat, lon: it.geo!.lon }) }))
      .filter(x => x.km <= 1200)
      .sort((a, b) => a.km - b.km)
      .slice(0, 10);

    const nearbyAviation = aviationPoints
      .map(it => ({ it, km: kmBetween(selectedPoint, { lat: it.geo!.lat, lon: it.geo!.lon }) }))
      .filter(x => x.km <= 600)
      .sort((a, b) => a.km - b.km)
      .slice(0, 10);

    const nearbyRoutes = STATIC_ROUTES.map(r => {
      const minKm = Math.min(...r.points.map(([lat, lon]) => kmBetween(selectedPoint, { lat, lon })));
      return { r, minKm, status: routeStatuses[r.id]?.status || 'Normal', risk: routeStatuses[r.id]?.risk || 0 };
    })
      .filter(x => x.minKm <= 1800)
      .sort((a, b) => a.minKm - b.minKm)
      .slice(0, 8);

    const instabilityRaw =
      nearbyConflict.reduce((acc, x) => acc + (x.h.severity * 18) / clamp(x.km, 150, 4000), 0) +
      nearbyUnrest.reduce((acc, x) => acc + (x.h.severity * 8) / clamp(x.km, 250, 4000), 0) +
      nearbyMilitary.reduce((acc, x) => acc + (x.h.severity * 10) / clamp(x.km, 250, 4000), 0) +
      nearbyDisasters.reduce((acc, x) => acc + ((x.it.severity || 0) * 10) / clamp(x.km, 200, 4000), 0) +
      (nearbyAviation.length >= 6 ? 4 : 0);

    const instability = Math.round(clamp(instabilityRaw * 12, 0, 100));

    const countryNeedle = (selectedPlace?.countryName || '').trim().toLowerCase();
    const recentForCountry =
      countryNeedle.length > 0
        ? watchlistItems
            .filter(it => (`${it.title} ${it.summary} ${it.region || ''}`).toLowerCase().includes(countryNeedle))
            .slice(0, 10)
        : [];

    return { nearbyConflict, nearbyUnrest, nearbyMilitary, nearbyInfra, nearbyDisasters, nearbyAviation, nearbyRoutes, instability, recentForCountry };
  }, [aviationPoints, conflictHotspots, infraHotspots, mapPoints, militaryHotspots, routeStatuses, selectedPlace?.countryName, selectedPoint, unrestHotspots, watchlistItems]);

  const [toastQueue, setToastQueue] = useState<{ id: string; title: string }[]>([]);
  const prevHighRef = React.useRef<Set<string>>(new Set());
  useEffect(() => {
    if (!alertsEnabled) return;
    const curr = new Set(highSeverityAlerts.map(it => it.id));
    const prev = prevHighRef.current;
    const newOnes = highSeverityAlerts.filter(it => !prev.has(it.id));
    if (newOnes.length > 0) {
      setToastQueue(q => [...q, ...newOnes.slice(0, 3).map(it => ({ id: it.id, title: it.title }))]);
    }
    prevHighRef.current = curr;
  }, [alertsEnabled, highSeverityAlerts]);

  useEffect(() => {
    if (toastQueue.length === 0) return;
    const next = toastQueue[0];
    const t = window.setTimeout(() => {
      setToastQueue(q => q.filter(x => x.id !== next.id));
    }, 6000);
    return () => window.clearTimeout(t);
  }, [toastQueue]);

  const MapClickHandler: React.FC<{ onPick: (p: LatLon) => void }> = ({ onPick }) => {
    useMapEvents({
      click: (e: any) => {
        const lat = Number(e?.latlng?.lat);
        const lon = Number(e?.latlng?.lng);
        if (Number.isFinite(lat) && Number.isFinite(lon)) onPick({ lat, lon });
      },
    });
    return null;
  };

  return (
    <div style={{ padding: '0 16px 16px 16px' }}>
      <div style={{ border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: 12, background: 'rgba(0,0,0,0.35)', padding: 12, marginBottom: 12 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 12, flexWrap: 'wrap' }}>
          <div>
            <h2 style={{ color: '#fff', margin: 0 }}>Scope Monitor</h2>
            <div style={{ color: '#a1a1aa', marginTop: 6 }}>OSINT dashboard: map + feeds + panels with real-time refresh.</div>
          </div>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <div style={{ color: '#a1a1aa', fontSize: '0.85rem' }}>
              {loading ? 'Updating…' : error ? error : lastUpdatedAt ? `Last update: ${new Date(lastUpdatedAt).toLocaleString()}` : ''}
            </div>
            <button onClick={loadAll} style={{ background: '#2563eb', color: '#fff', border: '1px solid #1d4ed8', borderRadius: 6, padding: '8px 12px', cursor: 'pointer' }}>Refresh</button>
          </div>
        </div>

        <div style={{ height: 10 }} />

        <div style={{ display: 'grid', gridTemplateColumns: '1.6fr 0.8fr 0.8fr', gap: 10, alignItems: 'end' }}>
          <div>
            <div style={{ color: '#a1a1aa', fontSize: '0.8rem', marginBottom: 6 }}>Search</div>
            <input value={keyword} onChange={e => setKeyword(e.target.value)} placeholder="keywords, country, region, sector" style={{ width: '100%', padding: 10, background: '#0b0f17', color: '#e5e7eb', border: '1px solid #374151', borderRadius: 8 }} />
          </div>
          <div>
            <div style={{ color: '#a1a1aa', fontSize: '0.8rem', marginBottom: 6 }}>Date Range</div>
            <select value={dateRange} onChange={e => setDateRange(e.target.value as DateRange)} style={{ width: '100%', padding: 10, background: '#0b0f17', color: '#e5e7eb', border: '1px solid #374151', borderRadius: 8 }}>
              <option value="24h">Last 24h</option>
              <option value="7d">Last 7d</option>
              <option value="30d">Last 30d</option>
            </select>
          </div>
          <div>
            <div style={{ color: '#a1a1aa', fontSize: '0.8rem', marginBottom: 6 }}>Min Severity</div>
            <input type="number" min={0} max={10} value={minSeverity} onChange={e => setMinSeverity(Number(e.target.value) || 0)} style={{ width: '100%', padding: 10, background: '#0b0f17', color: '#e5e7eb', border: '1px solid #374151', borderRadius: 8 }} />
          </div>
        </div>

        <div style={{ height: 10 }} />

        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
          <label style={{ display: 'flex', gap: 8, alignItems: 'center', color: '#e5e7eb' }}>
            <input type="checkbox" checked={showNews} onChange={e => setShowNews(e.target.checked)} />
            News
          </label>
          <label style={{ display: 'flex', gap: 8, alignItems: 'center', color: '#e5e7eb' }}>
            <input type="checkbox" checked={showEarthquakes} onChange={e => setShowEarthquakes(e.target.checked)} />
            Earthquakes
          </label>
          <label style={{ display: 'flex', gap: 8, alignItems: 'center', color: '#e5e7eb' }}>
            <input type="checkbox" checked={showDisasters} onChange={e => setShowDisasters(e.target.checked)} />
            Disasters
          </label>
          <label style={{ display: 'flex', gap: 8, alignItems: 'center', color: '#e5e7eb' }}>
            <input type="checkbox" checked={showCyber} onChange={e => setShowCyber(e.target.checked)} />
            Cyber
          </label>
          <label style={{ display: 'flex', gap: 8, alignItems: 'center', color: '#e5e7eb' }}>
            <input type="checkbox" checked={showGdelt} onChange={e => setShowGdelt(e.target.checked)} />
            GDELT
          </label>
          <label style={{ display: 'flex', gap: 8, alignItems: 'center', color: '#e5e7eb' }}>
            <input type="checkbox" checked={showAviation} onChange={e => setShowAviation(e.target.checked)} />
            Aviation
          </label>
          <label style={{ display: 'flex', gap: 8, alignItems: 'center', color: '#e5e7eb' }}>
            <input type="checkbox" checked={showMaritime} onChange={e => setShowMaritime(e.target.checked)} />
            Maritime
          </label>
          <label style={{ display: 'flex', gap: 8, alignItems: 'center', color: '#e5e7eb' }}>
            <input type="checkbox" checked={showSummaries} onChange={e => setShowSummaries(e.target.checked)} />
            Summaries
          </label>
          <label style={{ display: 'flex', gap: 8, alignItems: 'center', color: '#e5e7eb' }}>
            <input type="checkbox" checked={alertsEnabled} onChange={e => setAlertsEnabled(e.target.checked)} />
            Alerts
          </label>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 380px', gap: 12 }}>
        <div style={{ display: 'grid', gap: 12 }}>
          <div style={{ border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: 12, background: 'rgba(0,0,0,0.35)', padding: 12 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
              <div style={{ color: '#e5e7eb', fontWeight: 600 }}>Global Map</div>
              <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
                <select value={mapStyle} onChange={e => setMapStyle(e.target.value as any)} style={{ padding: '6px 10px', background: '#0b0f17', color: '#e5e7eb', border: '1px solid rgba(255,255,255,0.12)', borderRadius: 10 }}>
                  <option value="dark">Dark</option>
                  <option value="light">Light</option>
                  <option value="satellite">Satellite</option>
                </select>
                <div style={{ color: '#a1a1aa', fontSize: '0.85rem' }}>
                  {mapPoints.length} events • {conflictHotspots.length} conflict • {unrestHotspots.length} unrest • {infraHotspots.length} infra
                </div>
              </div>
            </div>
            <div style={{ height: 460, borderRadius: 14, overflow: 'hidden', border: '1px solid rgba(255,255,255,0.10)', boxShadow: '0 0 0 1px rgba(139,92,246,0.15), 0 18px 80px rgba(0,0,0,0.6), 0 0 60px rgba(56,189,248,0.10)', position: 'relative' }}>
              <MapContainer center={[20, 0]} zoom={2} style={{ height: '100%', width: '100%' }} scrollWheelZoom preferCanvas>
                <MapClickHandler onPick={(p: LatLon) => setSelectedPoint(p)} />
                <TileLayer
                  attribution={
                    mapStyle === 'satellite'
                      ? 'Tiles &copy; Esri'
                      : 'Tiles &copy; OpenStreetMap contributors, &copy; CARTO'
                  }
                  url={
                    mapStyle === 'satellite'
                      ? 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
                      : mapStyle === 'light'
                        ? 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png'
                        : 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png'
                  }
                />

                <LayersControl position="topright">
                  <LayersControl.Overlay checked name="Natural Disasters & Events">
                    <LayerGroup>
                      {mapPoints
                        .filter(it => it.kind !== 'aviation')
                        .map(it => {
                          const lat = it.geo?.lat as number;
                          const lon = it.geo?.lon as number;
                          const s = it.severity || 0;
                          const color = it.kind === 'cyber' ? '#a78bfa' : severityColor(s);
                          const r = Math.max(4, Math.min(14, 4 + s));
                          return (
                            <React.Fragment key={it.id}>
                              <CircleMarker center={[lat, lon]} radius={r * 2.6} pathOptions={{ color: 'transparent', fillColor: color, fillOpacity: 0.06, weight: 0 }} />
                              <CircleMarker center={[lat, lon]} radius={r * 1.7} pathOptions={{ color: 'transparent', fillColor: color, fillOpacity: 0.11, weight: 0 }} />
                              <CircleMarker center={[lat, lon]} radius={r} pathOptions={{ color, fillColor: color, fillOpacity: 0.7, weight: 1 }}>
                                <Popup>
                                  <div style={{ minWidth: 240 }}>
                                    <div style={{ fontWeight: 700, marginBottom: 6 }}>{it.title}</div>
                                    <div style={{ color: '#555', marginBottom: 8 }}>{showSummaries ? (it.summary || it.title) : it.title}</div>
                                    <div style={{ fontSize: 12, color: '#777' }}>
                                      {it.kind}
                                      {it.category ? ` • ${it.category}` : ''}
                                      {it.region ? ` • ${it.region}` : ''}
                                      {it.severity != null ? ` • severity ${it.severity}` : ''}
                                    </div>
                                    {it.url ? (
                                      <div style={{ marginTop: 8 }}>
                                        <a href={it.url} target="_blank" rel="noreferrer">Open source</a>
                                      </div>
                                    ) : null}
                                  </div>
                                </Popup>
                              </CircleMarker>
                            </React.Fragment>
                          );
                        })}
                    </LayerGroup>
                  </LayersControl.Overlay>

                  <LayersControl.Overlay checked={showAviation} name="Military Activity (Air) / ADS-B">
                    <LayerGroup>
                      {aviationPoints.map(it => {
                        const lat = it.geo?.lat as number;
                        const lon = it.geo?.lon as number;
                        const r = Math.max(3, Math.min(10, 3 + (it.severity || 0)));
                        const color = '#38bdf8';
                        return (
                          <React.Fragment key={`av-${it.id}`}>
                            <CircleMarker center={[lat, lon]} radius={r * 2.4} pathOptions={{ color: 'transparent', fillColor: color, fillOpacity: 0.06, weight: 0 }} />
                            <CircleMarker center={[lat, lon]} radius={r * 1.5} pathOptions={{ color: 'transparent', fillColor: color, fillOpacity: 0.10, weight: 0 }} />
                            <CircleMarker center={[lat, lon]} radius={r} pathOptions={{ color, fillColor: color, fillOpacity: 0.65, weight: 1 }}>
                              <Popup>
                                <div style={{ minWidth: 240 }}>
                                  <div style={{ fontWeight: 700, marginBottom: 6 }}>{it.title}</div>
                                  <div style={{ color: '#555', marginBottom: 8 }}>{showSummaries ? (it.summary || it.title) : it.title}</div>
                                  <div style={{ fontSize: 12, color: '#777' }}>
                                    {it.kind}
                                    {it.region ? ` • ${it.region}` : ''}
                                    {it.severity != null ? ` • severity ${it.severity}` : ''}
                                  </div>
                                </div>
                              </Popup>
                            </CircleMarker>
                          </React.Fragment>
                        );
                      })}
                    </LayerGroup>
                  </LayersControl.Overlay>

                  <LayersControl.Overlay checked={showMaritime} name="Natural Marine Hazards">
                    <LayerGroup>
                      {maritimePoints.map(it => {
                        const lat = it.geo?.lat as number;
                        const lon = it.geo?.lon as number;
                        const r = Math.max(3, Math.min(12, 3 + (it.severity || 0)));
                        const color = '#60a5fa';
                        return (
                          <React.Fragment key={`mt-${it.id}`}>
                            <CircleMarker center={[lat, lon]} radius={r * 2.5} pathOptions={{ color: 'transparent', fillColor: color, fillOpacity: 0.05, weight: 0 }} />
                            <CircleMarker center={[lat, lon]} radius={r * 1.6} pathOptions={{ color: 'transparent', fillColor: color, fillOpacity: 0.09, weight: 0 }} />
                            <CircleMarker center={[lat, lon]} radius={r} pathOptions={{ color, fillColor: color, fillOpacity: 0.55, weight: 1 }}>
                              <Popup>
                                <div style={{ minWidth: 240 }}>
                                  <div style={{ fontWeight: 700, marginBottom: 6 }}>{it.title}</div>
                                  <div style={{ color: '#555', marginBottom: 8 }}>{showSummaries ? (it.summary || it.title) : it.title}</div>
                                  <div style={{ fontSize: 12, color: '#777' }}>
                                    Maritime
                                    {it.category ? ` • ${it.category}` : ''}
                                    {it.region ? ` • ${it.region}` : ''}
                                  </div>
                                </div>
                              </Popup>
                            </CircleMarker>
                          </React.Fragment>
                        );
                      })}
                    </LayerGroup>
                  </LayersControl.Overlay>

                  <LayersControl.Overlay checked name="Conflict Zones / Tensions / Restrictions">
                    <LayerGroup>
                      {conflictHotspots.map(h => {
                        const color = h.kind === 'conflict' ? '#ef4444' : '#fb7185';
                        const r = clamp(8 + h.severity, 10, 22);
                        return (
                          <CircleMarker key={h.id} center={[h.lat, h.lon]} radius={r} pathOptions={{ color, fillColor: color, fillOpacity: 0.18, weight: 2 }}>
                            <Popup>
                              <div style={{ minWidth: 240 }}>
                                <div style={{ fontWeight: 700, marginBottom: 6 }}>{h.name}</div>
                                <div style={{ color: '#555', marginBottom: 8 }}>{h.details || 'Risk hotspot'}</div>
                                <div style={{ fontSize: 12, color: '#777' }}>severity {h.severity}</div>
                              </div>
                            </Popup>
                          </CircleMarker>
                        );
                      })}
                    </LayerGroup>
                  </LayersControl.Overlay>

                  <LayersControl.Overlay checked name="Protests & Social Unrest">
                    <LayerGroup>
                      {unrestHotspots.map(h => {
                        const color = '#f59e0b';
                        const r = clamp(7 + h.severity, 9, 18);
                        return (
                          <CircleMarker key={h.id} center={[h.lat, h.lon]} radius={r} pathOptions={{ color, fillColor: color, fillOpacity: 0.16, weight: 2 }}>
                            <Popup>
                              <div style={{ minWidth: 240 }}>
                                <div style={{ fontWeight: 700, marginBottom: 6 }}>{h.name}</div>
                                <div style={{ color: '#555', marginBottom: 8 }}>{h.details || 'Unrest hotspot'}</div>
                                <div style={{ fontSize: 12, color: '#777' }}>severity {h.severity}</div>
                              </div>
                            </Popup>
                          </CircleMarker>
                        );
                      })}
                    </LayerGroup>
                  </LayersControl.Overlay>

                  <LayersControl.Overlay checked name="Military Bases & Fleets">
                    <LayerGroup>
                      {militaryHotspots.map(h => {
                        const color = '#fbbf24';
                        const r = clamp(7 + h.severity, 9, 20);
                        return (
                          <CircleMarker key={h.id} center={[h.lat, h.lon]} radius={r} pathOptions={{ color, fillColor: color, fillOpacity: 0.14, weight: 2 }}>
                            <Popup>
                              <div style={{ minWidth: 240 }}>
                                <div style={{ fontWeight: 700, marginBottom: 6 }}>{h.name}</div>
                                <div style={{ color: '#555', marginBottom: 8 }}>{h.details || ''}</div>
                                <div style={{ fontSize: 12, color: '#777' }}>severity {h.severity}</div>
                              </div>
                            </Popup>
                          </CircleMarker>
                        );
                      })}
                    </LayerGroup>
                  </LayersControl.Overlay>

                  <LayersControl.Overlay checked name="Market News Hotspots">
                    <LayerGroup>
                      {marketHotspots.map(h => {
                        const color = '#60a5fa';
                        const r = clamp(8 + h.severity, 10, 24);
                        return (
                          <CircleMarker key={h.id} center={[h.lat, h.lon]} radius={r} pathOptions={{ color, fillColor: color, fillOpacity: 0.16, weight: 2 }}>
                            <Popup>
                              <div style={{ minWidth: 240 }}>
                                <div style={{ fontWeight: 700, marginBottom: 6 }}>{h.name}</div>
                                <div style={{ color: '#555', marginBottom: 8 }}>{h.details || ''}</div>
                                <div style={{ fontSize: 12, color: '#777' }}>trend {h.severity}/10</div>
                              </div>
                            </Popup>
                          </CircleMarker>
                        );
                      })}
                    </LayerGroup>
                  </LayersControl.Overlay>

                  <LayersControl.Overlay checked={showNews} name="Breaking News Hotspots">
                    <LayerGroup>
                      {breakingNewsHotspots.map(h => {
                        const color = '#38bdf8';
                        const r = clamp(8 + h.severity, 10, 24);
                        return (
                          <CircleMarker key={h.id} center={[h.lat, h.lon]} radius={r} pathOptions={{ color, fillColor: color, fillOpacity: 0.14, weight: 2 }}>
                            <Popup>
                              <div style={{ minWidth: 260 }}>
                                <div style={{ fontWeight: 700, marginBottom: 6 }}>{h.name}</div>
                                <div style={{ color: '#555', marginBottom: 8 }}>{h.details || ''}</div>
                                <div style={{ fontSize: 12, color: '#777' }}>impact {h.severity}/10</div>
                              </div>
                            </Popup>
                          </CircleMarker>
                        );
                      })}
                    </LayerGroup>
                  </LayersControl.Overlay>

                  <LayersControl.Overlay checked name="Critical Geography (Chokepoints)">
                    <LayerGroup>
                      {infraHotspots.map(h => {
                        const color = '#a78bfa';
                        const r = clamp(7 + h.severity, 9, 20);
                        return (
                          <CircleMarker key={h.id} center={[h.lat, h.lon]} radius={r} pathOptions={{ color, fillColor: color, fillOpacity: 0.18, weight: 2 }}>
                            <Popup>
                              <div style={{ minWidth: 260 }}>
                                <div style={{ fontWeight: 700, marginBottom: 6 }}>{h.name}</div>
                                <div style={{ color: '#555', marginBottom: 8 }}>{h.details || ''}</div>
                                <div style={{ display: 'grid', gap: 6, fontSize: 12, color: '#777' }}>
                                  {(() => {
                                    const map: Record<string, string[]> = {
                                      'infra-hormuz': ['ship-gulf-asia', 'flight-me-apac'],
                                      'infra-bab': ['ship-asia-europe'],
                                      'infra-suez': ['ship-asia-europe'],
                                      'infra-malacca': ['ship-gulf-asia', 'ship-asia-europe'],
                                      'infra-panama': ['ship-atlantic-pacific', 'flight-natl'],
                                      'infra-gibraltar': ['ship-asia-europe', 'flight-natl'],
                                      'tension-taiwan': ['flight-eurasia'],
                                    };
                                    const ids = map[h.id] || [];
                                    const routes = ids.length ? STATIC_ROUTES.filter(r => ids.includes(r.id)) : STATIC_ROUTES.slice(0, 3);
                                    return routes.slice(0, 5).map(r => (
                                      <div key={r.id}>
                                        {r.mode.toUpperCase()} • {r.name} • {routeStatuses[r.id]?.status || 'Normal'}
                                      </div>
                                    ));
                                  })()}
                                </div>
                              </div>
                            </Popup>
                          </CircleMarker>
                        );
                      })}
                    </LayerGroup>
                  </LayersControl.Overlay>

                  <LayersControl.Overlay checked name="Critical Infrastructure Lines (Cables / Pipelines)">
                    <LayerGroup>
                      {STATIC_INFRA_LINES.map(l => {
                        const st = infraLineStatuses[l.id]?.status || 'Normal';
                        const riskColor = st === 'Disrupted' ? '#ef4444' : st === 'At Risk' ? '#fb7185' : st === 'Watch' ? '#f59e0b' : l.kind === 'cable' ? '#a78bfa' : '#22c55e';
                        const dash = l.kind === 'cable' ? '4 9' : undefined;
                        return (
                          <Polyline key={l.id} positions={l.points.map(([lat, lon]) => [lat, lon])} pathOptions={{ color: riskColor, weight: 2, opacity: 0.7, dashArray: dash }}>
                            <Popup>
                              <div style={{ minWidth: 260 }}>
                                <div style={{ fontWeight: 700, marginBottom: 6 }}>{l.name}</div>
                                <div style={{ color: '#555', marginBottom: 8 }}>{l.kind === 'cable' ? 'Undersea/Backbone cable' : 'Energy pipeline'} • {st}</div>
                                <div style={{ fontSize: 12, color: '#777' }}>risk {infraLineStatuses[l.id]?.risk?.toFixed?.(1) || '0'}</div>
                              </div>
                            </Popup>
                          </Polyline>
                        );
                      })}
                    </LayerGroup>
                  </LayersControl.Overlay>

                  <LayersControl.Overlay checked name="Travel Routes (Ships / Flights)">
                    <LayerGroup>
                      {STATIC_ROUTES.map(r => {
                        const st = routeStatuses[r.id]?.status || 'Normal';
                        const color = st === 'Closed' ? '#b91c1c' : st === 'Restricted' ? '#ef4444' : st === 'Caution' ? '#f59e0b' : '#22c55e';
                        const dash = r.mode === 'flight' ? '6 10' : undefined;
                        return (
                          <Polyline
                            key={r.id}
                            positions={r.points.map(([lat, lon]) => [lat, lon])}
                            pathOptions={{ color, weight: 2, opacity: 0.7, dashArray: dash }}
                          >
                            <Popup>
                              <div style={{ minWidth: 260 }}>
                                <div style={{ fontWeight: 700, marginBottom: 6 }}>{r.name}</div>
                                <div style={{ color: '#555', marginBottom: 8 }}>
                                  {r.mode === 'ship' ? 'Ship lane' : 'Flight corridor'} • {st}
                                </div>
                                <div style={{ fontSize: 12, color: '#777' }}>risk {routeStatuses[r.id]?.risk?.toFixed?.(1) || '0'}</div>
                              </div>
                            </Popup>
                          </Polyline>
                        );
                      })}
                    </LayerGroup>
                  </LayersControl.Overlay>

                  <LayersControl.Overlay checked name="Cyber & Threat Intelligence">
                    <LayerGroup>
                      {cyberHotspots.map(h => {
                        const color = '#a78bfa';
                        const r = clamp(7 + h.severity, 9, 20);
                        return (
                          <CircleMarker key={h.id} center={[h.lat, h.lon]} radius={r} pathOptions={{ color, fillColor: color, fillOpacity: 0.14, weight: 2 }}>
                            <Popup>
                              <div style={{ minWidth: 240 }}>
                                <div style={{ fontWeight: 700, marginBottom: 6 }}>{h.name}</div>
                                <div style={{ color: '#555', marginBottom: 8 }}>{h.details || ''}</div>
                                <div style={{ fontSize: 12, color: '#777' }}>severity {h.severity}</div>
                              </div>
                            </Popup>
                          </CircleMarker>
                        );
                      })}
                    </LayerGroup>
                  </LayersControl.Overlay>
                </LayersControl>
              </MapContainer>

              <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none', background: 'radial-gradient(800px 320px at 14% 12%, rgba(139,92,246,0.14), transparent 60%), radial-gradient(700px 300px at 78% 24%, rgba(56,189,248,0.10), transparent 58%), radial-gradient(700px 280px at 55% 90%, rgba(34,197,94,0.06), transparent 60%)' }} />
              <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none', background: 'repeating-linear-gradient(180deg, rgba(255,255,255,0.03) 0px, rgba(255,255,255,0.03) 1px, transparent 2px, transparent 6px)', opacity: 0.25, mixBlendMode: 'overlay' as any }} />

              <div style={{ position: 'absolute', left: 12, bottom: 12, background: 'rgba(0,0,0,0.55)', border: '1px solid rgba(255,255,255,0.12)', borderRadius: 12, padding: '10px 12px', color: '#e5e7eb', fontSize: 12, backdropFilter: 'blur(8px)', maxWidth: 340 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10, marginBottom: 8 }}>
                  <div style={{ fontWeight: 700 }}>Legend</div>
                  <div style={{ color: '#a1a1aa' }}>Click map for region brief</div>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, color: '#cbd5e1' }}>
                  <div><span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: 999, background: '#22c55e', marginRight: 8 }} />Low severity</div>
                  <div><span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: 999, background: '#f59e0b', marginRight: 8 }} />Medium severity</div>
                  <div><span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: 999, background: '#ef4444', marginRight: 8 }} />High severity</div>
                  <div><span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: 999, background: '#38bdf8', marginRight: 8 }} />Aircraft</div>
                  <div><span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: 999, background: '#fb7185', marginRight: 8 }} />Conflict/tension</div>
                  <div><span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: 999, background: '#60a5fa', marginRight: 8 }} />Market hotspot</div>
                </div>
              </div>
            </div>
          </div>

          <div style={{ border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: 12, background: 'rgba(0,0,0,0.35)', padding: 12 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
              <div style={{ color: '#e5e7eb', fontWeight: 600 }}>Live News Feed</div>
              <div style={{ color: '#a1a1aa', fontSize: '0.85rem' }}>{liveNewsItems.length} items</div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '0.8fr 1fr', gap: 10, alignItems: 'end', marginBottom: 10 }}>
              <div>
                <div style={{ color: '#a1a1aa', fontSize: '0.8rem', marginBottom: 6 }}>Scope</div>
                <select value={newsGeoScope} onChange={e => setNewsGeoScope(e.target.value as any)} style={{ width: '100%', padding: 10, background: '#0b0f17', color: '#e5e7eb', border: '1px solid #374151', borderRadius: 8 }}>
                  <option value="global">Global</option>
                  <option value="continent">Continental</option>
                  <option value="regional">Regional (countries)</option>
                  <option value="local">Local (one country)</option>
                </select>
              </div>
              <div>
                <div style={{ color: '#a1a1aa', fontSize: '0.8rem', marginBottom: 6 }}>
                  {newsGeoScope === 'continent' ? 'Continent' : newsGeoScope === 'regional' ? 'Countries (comma-separated)' : newsGeoScope === 'local' ? 'Country' : 'Filter'}
                </div>
                {newsGeoScope === 'continent' ? (
                  <select value={newsContinent} onChange={e => setNewsContinent(e.target.value as any)} style={{ width: '100%', padding: 10, background: '#0b0f17', color: '#e5e7eb', border: '1px solid #374151', borderRadius: 8 }}>
                    <option value="Africa">Africa</option>
                    <option value="Asia">Asia</option>
                    <option value="Europe">Europe</option>
                    <option value="North America">North America</option>
                    <option value="South America">South America</option>
                    <option value="Oceania">Oceania</option>
                    <option value="Middle East">Middle East</option>
                  </select>
                ) : newsGeoScope === 'regional' ? (
                  <input value={newsRegionalCountries} onChange={e => setNewsRegionalCountries(e.target.value)} placeholder="e.g. Germany, France, Italy" style={{ width: '100%', padding: 10, background: '#0b0f17', color: '#e5e7eb', border: '1px solid #374151', borderRadius: 8 }} />
                ) : newsGeoScope === 'local' ? (
                  <input value={newsLocalCountry} onChange={e => setNewsLocalCountry(e.target.value)} placeholder="e.g. United States" style={{ width: '100%', padding: 10, background: '#0b0f17', color: '#e5e7eb', border: '1px solid #374151', borderRadius: 8 }} />
                ) : (
                  <input value={keyword} onChange={e => setKeyword(e.target.value)} placeholder="keywords" style={{ width: '100%', padding: 10, background: '#0b0f17', color: '#e5e7eb', border: '1px solid #374151', borderRadius: 8 }} />
                )}
              </div>
            </div>

            <div style={{ display: 'grid', gap: 10 }}>
              {liveNewsItems.slice(0, 60).map((it: MonitorFeedItem) => (
                <div key={it.id} style={{ border: '1px solid rgba(255,255,255,0.1)', borderRadius: 12, padding: 12 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12 }}>
                    <div style={{ color: '#e5e7eb', fontWeight: 600 }}>{it.title}</div>
                    <div style={{ color: '#a1a1aa', fontSize: '0.85rem', whiteSpace: 'nowrap' }}>
                      {it.published_at ? new Date(it.published_at).toLocaleString() : ''}
                    </div>
                  </div>
                  <div style={{ color: '#a1a1aa', marginTop: 6 }}>{it.summary}</div>
                  <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 10, color: '#a1a1aa', fontSize: '0.85rem' }}>
                    {it.category ? <span>{it.category}</span> : <span>News</span>}
                    {it.source ? <span>· {it.source}</span> : null}
                    {it.relevance != null ? <span>· impact {it.relevance}</span> : null}
                    {it.url ? (
                      <a href={it.url} target="_blank" rel="noreferrer" style={{ color: '#8b5cf6', textDecoration: 'none' }}>
                        Open
                      </a>
                    ) : null}
                  </div>
                </div>
              ))}
              {liveNewsItems.length === 0 ? <div style={{ color: '#a1a1aa' }}>No matching news found in the last 30 days.</div> : null}
            </div>
          </div>

          <div style={{ border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: 12, background: 'rgba(0,0,0,0.35)', padding: 12, display: 'flex', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap' }}>
            <div style={{ color: '#a1a1aa' }}>
              Sources: USGS, NASA EONET, URLhaus, GDELT, internal news ingest
            </div>
            <div style={{ color: '#a1a1aa' }}>
              {lastUpdatedAt ? `Last update: ${new Date(lastUpdatedAt).toLocaleString()}` : ''}
            </div>
          </div>
        </div>

        <div style={{ display: 'grid', gap: 12, alignContent: 'start' }}>
          <div style={{ border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: 12, background: 'rgba(0,0,0,0.35)', padding: 12 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 10, marginBottom: 10 }}>
              <div style={{ color: '#e5e7eb', fontWeight: 600 }}>Region Intelligence</div>
              <button
                onClick={() => {
                  if (!selectedPoint) return;
                  setWeatherLat(selectedPoint.lat);
                  setWeatherLon(selectedPoint.lon);
                  loadWeather();
                }}
                disabled={!selectedPoint}
                style={{ background: 'transparent', color: '#e5e7eb', border: '1px solid rgba(255,255,255,0.14)', borderRadius: 10, padding: '8px 10px', cursor: selectedPoint ? 'pointer' : 'default', opacity: selectedPoint ? 1 : 0.5 }}
              >
                Use for Weather
              </button>
            </div>
            {!selectedPoint ? (
              <div style={{ color: '#a1a1aa' }}>Click a country/region on the map to see conditions, risk, chokepoints, and routes.</div>
            ) : (
              <div style={{ display: 'grid', gap: 10 }}>
                <div style={{ color: '#a1a1aa' }}>
                  {selectedPlace?.label ? selectedPlace.label : `${selectedPoint.lat.toFixed(3)}, ${selectedPoint.lon.toFixed(3)}`}
                </div>
                {selectedPlaceError ? <div style={{ color: '#fca5a5' }}>{selectedPlaceError}</div> : null}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
                  <div style={{ border: '1px solid rgba(255,255,255,0.1)', borderRadius: 12, padding: 10 }}>
                    <div style={{ color: '#a1a1aa', fontSize: 12, marginBottom: 6 }}>Instability Score</div>
                    <div style={{ color: '#e5e7eb', fontWeight: 700, fontSize: 20 }}>{selectedBrief?.instability ?? 0}/100</div>
                    <div style={{ color: '#a1a1aa', fontSize: 12, marginTop: 6 }}>
                      {selectedPlace?.countryName ? `${selectedPlace.countryName}${selectedPlace.countryCode ? ` (${selectedPlace.countryCode})` : ''}` : '—'}
                    </div>
                  </div>
                  <div style={{ border: '1px solid rgba(255,255,255,0.1)', borderRadius: 12, padding: 10 }}>
                    <div style={{ color: '#a1a1aa', fontSize: 12, marginBottom: 6 }}>Conditions</div>
                    <div style={{ color: '#e5e7eb', fontWeight: 700, fontSize: 14 }}>
                      {selectedWeather ? `Temp ${selectedWeather.temperature}°C • Wind ${selectedWeather.windspeed} km/h` : 'No weather data'}
                    </div>
                    <div style={{ color: '#a1a1aa', fontSize: 12, marginTop: 6 }}>{selectedWeather?.time ? new Date(selectedWeather.time).toLocaleString() : ''}</div>
                  </div>
                </div>

                <div style={{ border: '1px solid rgba(255,255,255,0.1)', borderRadius: 12, padding: 10 }}>
                  <div style={{ color: '#e5e7eb', fontWeight: 600, marginBottom: 8 }}>Key Straits & Chokepoints (Nearby)</div>
                  {selectedBrief?.nearbyInfra?.length ? (
                    <div style={{ display: 'grid', gap: 6 }}>
                      {selectedBrief.nearbyInfra.slice(0, 5).map(x => (
                        <div key={x.h.id} style={{ display: 'flex', justifyContent: 'space-between', gap: 10, color: '#a1a1aa' }}>
                          <span>{x.h.name}</span>
                          <span style={{ whiteSpace: 'nowrap' }}>{Math.round(x.km)} km</span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div style={{ color: '#a1a1aa' }}>No chokepoints nearby.</div>
                  )}
                </div>

                <div style={{ border: '1px solid rgba(255,255,255,0.1)', borderRadius: 12, padding: 10 }}>
                  <div style={{ color: '#e5e7eb', fontWeight: 600, marginBottom: 8 }}>Travel Routes (Availability)</div>
                  {selectedBrief?.nearbyRoutes?.length ? (
                    <div style={{ display: 'grid', gap: 6 }}>
                      {selectedBrief.nearbyRoutes.map(x => (
                        <div key={x.r.id} style={{ display: 'flex', justifyContent: 'space-between', gap: 10, color: '#a1a1aa' }}>
                          <span>{x.r.mode.toUpperCase()} • {x.r.name}</span>
                          <span style={{ whiteSpace: 'nowrap', color: x.status === 'Closed' ? '#ef4444' : x.status === 'Restricted' ? '#fb7185' : x.status === 'Caution' ? '#f59e0b' : '#22c55e' }}>
                            {x.status}
                          </span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div style={{ color: '#a1a1aa' }}>No major routes nearby.</div>
                  )}
                </div>

                <div style={{ border: '1px solid rgba(255,255,255,0.1)', borderRadius: 12, padding: 10 }}>
                  <div style={{ color: '#e5e7eb', fontWeight: 600, marginBottom: 8 }}>Nearby Signals</div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, color: '#a1a1aa' }}>
                    <div>Conflict/tension: {selectedBrief?.nearbyConflict?.length ?? 0}</div>
                    <div>Unrest: {selectedBrief?.nearbyUnrest?.length ?? 0}</div>
                    <div>Military: {selectedBrief?.nearbyMilitary?.length ?? 0}</div>
                    <div>Disasters: {selectedBrief?.nearbyDisasters?.length ?? 0}</div>
                    <div>Aircraft: {selectedBrief?.nearbyAviation?.length ?? 0}</div>
                    <div>Market hotspots: {marketHotspots.length}</div>
                  </div>
                </div>
              </div>
            )}
          </div>

          <div style={{ border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: 12, background: 'rgba(0,0,0,0.35)', padding: 12 }}>
            <div style={{ color: '#e5e7eb', fontWeight: 600, marginBottom: 10 }}>Cyber Threats</div>
            <div style={{ display: 'grid', gap: 10 }}>
              {cyberItems.slice(0, 10).map(it => (
                <div key={it.id} style={{ border: '1px solid rgba(255,255,255,0.1)', borderRadius: 10, padding: 10 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10 }}>
                    <div style={{ color: '#e5e7eb', fontWeight: 600 }}>{it.title}</div>
                    <div style={{ color: severityColor(it.severity || 0), fontSize: '0.85rem', whiteSpace: 'nowrap' }}>sev {it.severity ?? 0}</div>
                  </div>
                  <div style={{ color: '#a1a1aa', marginTop: 6 }}>{it.summary}</div>
                  {it.url ? (
                    <div style={{ marginTop: 8 }}>
                      <a href={it.url} target="_blank" rel="noreferrer" style={{ color: '#8b5cf6', textDecoration: 'none' }}>Open</a>
                    </div>
                  ) : null}
                </div>
              ))}
            </div>
          </div>

          <div style={{ border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: 12, background: 'rgba(0,0,0,0.35)', padding: 12 }}>
            <div style={{ color: '#e5e7eb', fontWeight: 600, marginBottom: 10 }}>Weather & Climate</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, marginBottom: 10 }}>
              <input type="number" value={weatherLat} onChange={e => setWeatherLat(Number(e.target.value))} style={{ width: '100%', padding: 10, background: '#0b0f17', color: '#e5e7eb', border: '1px solid #374151', borderRadius: 8 }} />
              <input type="number" value={weatherLon} onChange={e => setWeatherLon(Number(e.target.value))} style={{ width: '100%', padding: 10, background: '#0b0f17', color: '#e5e7eb', border: '1px solid #374151', borderRadius: 8 }} />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 10 }}>
              <div style={{ color: '#a1a1aa' }}>
                {weather ? `Temp ${weather.temperature}°C • Wind ${weather.windspeed} km/h` : weatherError ? weatherError : 'No data'}
              </div>
              <button onClick={loadWeather} style={{ background: 'transparent', color: '#e5e7eb', border: '1px solid #374151', borderRadius: 8, padding: '8px 10px', cursor: 'pointer' }}>Refresh</button>
            </div>
          </div>

          <div style={{ border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: 12, background: 'rgba(0,0,0,0.35)', padding: 12 }}>
            <div style={{ color: '#e5e7eb', fontWeight: 600, marginBottom: 10 }}>Country / Demographics</div>
            <div style={{ display: 'flex', gap: 8, marginBottom: 10 }}>
              <input value={countryQuery} onChange={e => setCountryQuery(e.target.value)} placeholder="Country name" style={{ flex: 1, padding: 10, background: '#0b0f17', color: '#e5e7eb', border: '1px solid #374151', borderRadius: 8 }} />
              <button onClick={loadCountry} disabled={countryLoading} style={{ background: 'transparent', color: '#e5e7eb', border: '1px solid #374151', borderRadius: 8, padding: '8px 10px', cursor: 'pointer', opacity: countryLoading ? 0.6 : 1 }}>
                {countryLoading ? 'Loading…' : 'Lookup'}
              </button>
            </div>
            {countryError ? <div style={{ color: '#fca5a5', marginBottom: 10 }}>{countryError}</div> : null}
            {countryInfo ? (
              <div style={{ display: 'grid', gap: 8 }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10 }}>
                  <div style={{ color: '#e5e7eb', fontWeight: 600 }}>{countryInfo.name}{countryInfo.cca2 ? ` (${countryInfo.cca2})` : ''}</div>
                  {countryInfo.flagPng ? <img src={countryInfo.flagPng} alt="" style={{ width: 34, height: 22, borderRadius: 4, border: '1px solid rgba(255,255,255,0.15)' }} /> : null}
                </div>
                <div style={{ color: '#a1a1aa' }}>{countryInfo.official}</div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, color: '#a1a1aa' }}>
                  <div>Capital: {countryInfo.capital || '—'}</div>
                  <div>Population: {countryInfo.population ? countryInfo.population.toLocaleString() : '—'}</div>
                  <div>Region: {countryInfo.region || '—'}</div>
                  <div>Subregion: {countryInfo.subregion || '—'}</div>
                </div>
                <div style={{ color: '#a1a1aa' }}>Mentions in watchlist: {countryMentions}</div>
              </div>
            ) : (
              <div style={{ color: '#a1a1aa' }}>Enter a country name to fetch demographics.</div>
            )}
          </div>

          <div style={{ border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: 12, background: 'rgba(0,0,0,0.35)', padding: 12 }}>
            <div style={{ color: '#e5e7eb', fontWeight: 600, marginBottom: 10 }}>Notifications / Alerts</div>
            <div style={{ display: 'grid', gap: 10 }}>
              {highSeverityAlerts.length === 0 ? (
                <div style={{ color: '#a1a1aa' }}>No high-severity alerts in the current window.</div>
              ) : (
                highSeverityAlerts.map(it => (
                  <div key={it.id} style={{ border: '1px solid rgba(255,255,255,0.1)', borderRadius: 10, padding: 10 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10 }}>
                      <div style={{ color: '#e5e7eb', fontWeight: 600 }}>{it.title}</div>
                      <div style={{ color: severityColor(it.severity || 0), fontSize: '0.85rem', whiteSpace: 'nowrap' }}>sev {it.severity ?? 0}</div>
                    </div>
                    <div style={{ color: '#a1a1aa', marginTop: 6 }}>{it.region || it.category || ''}</div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>
      <div style={{ position: 'fixed', top: 70, right: 16, display: 'grid', gap: 8, zIndex: 50 }}>
        {toastQueue.slice(0, 3).map(t => (
          <div key={t.id} style={{ background: 'rgba(0,0,0,0.8)', color: '#fff', border: '1px solid rgba(255,255,255,0.2)', borderRadius: 8, padding: '8px 12px', minWidth: 240 }}>
            {t.title}
          </div>
        ))}
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
