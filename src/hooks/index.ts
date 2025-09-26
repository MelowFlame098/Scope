import { useState, useEffect, useCallback, useRef } from 'react';
import { useStore } from '../store/useStore';
import { useWebSocket } from '../services/websocket';
import { debounce, throttle } from '../utils';
import type { Asset, Model, NewsItem, ForumPost, ChartData } from '../store/useStore';

// Custom hook for API data fetching with loading and error states
export const useApi = <T>(
  apiFunction: () => Promise<T>,
  dependencies: any[] = [],
  options: {
    immediate?: boolean;
    onSuccess?: (data: T) => void;
    onError?: (error: Error) => void;
  } = {}
) => {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const { immediate = true, onSuccess, onError } = options;

  const execute = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const result = await apiFunction();
      setData(result);
      onSuccess?.(result);
      return result;
    } catch (err) {
      const error = err as Error;
      setError(error);
      onError?.(error);
      throw error;
    } finally {
      setLoading(false);
    }
  }, dependencies);

  useEffect(() => {
    if (immediate) {
      execute();
    }
  }, [execute, immediate]);

  return { data, loading, error, execute, refetch: execute };
};

// Hook for managing local storage with React state
export const useLocalStorage = <T>(
  key: string,
  initialValue: T
): [T, (value: T | ((val: T) => T)) => void] => {
  const [storedValue, setStoredValue] = useState<T>(() => {
    try {
      if (typeof window === 'undefined') return initialValue;
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch (error) {
      console.error(`Error reading localStorage key "${key}":`, error);
      return initialValue;
    }
  });

  const setValue = (value: T | ((val: T) => T)) => {
    try {
      const valueToStore = value instanceof Function ? value(storedValue) : value;
      setStoredValue(valueToStore);
      if (typeof window !== 'undefined') {
        window.localStorage.setItem(key, JSON.stringify(valueToStore));
      }
    } catch (error) {
      console.error(`Error setting localStorage key "${key}":`, error);
    }
  };

  return [storedValue, setValue];
};

// Hook for debounced values
export const useDebounce = <T>(value: T, delay: number): T => {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => {
      clearTimeout(handler);
    };
  }, [value, delay]);

  return debouncedValue;
};

// Hook for throttled callbacks
export const useThrottle = <T extends (...args: any[]) => any>(
  callback: T,
  delay: number
): T => {
  const throttledCallback = useRef(throttle(callback, delay));

  useEffect(() => {
    throttledCallback.current = throttle(callback, delay);
  }, [callback, delay]);

  return throttledCallback.current as T;
};

// Hook for managing WebSocket connections and subscriptions
export const useWebSocketConnection = () => {
  const { connect, disconnect, isConnected, getStatus } = useWebSocket();
  const [connectionStatus, setConnectionStatus] = useState(getStatus());

  useEffect(() => {
    const updateStatus = () => setConnectionStatus(getStatus());
    const interval = setInterval(updateStatus, 1000);
    return () => clearInterval(interval);
  }, [getStatus]);

  const connectWithRetry = useCallback(async (maxRetries = 3) => {
    let retries = 0;
    while (retries < maxRetries && !isConnected()) {
      try {
        await connect();
        break;
      } catch (error) {
        retries++;
        if (retries < maxRetries) {
          await new Promise(resolve => setTimeout(resolve, 1000 * retries));
        } else {
          throw error;
        }
      }
    }
  }, [connect, isConnected]);

  return {
    connect: connectWithRetry,
    disconnect,
    isConnected: connectionStatus.connected,
    status: connectionStatus
  };
};

// Hook for real-time asset data
export const useAssetData = (assetId: string) => {
  const { subscribeToAsset, unsubscribeFromAsset, on, off } = useWebSocket();
  const updateAsset = useStore(state => state.updateAsset);
  const assets = useStore(state => state.assets);
  const asset = assets.find(a => a.id === assetId);

  useEffect(() => {
    if (assetId) {
      subscribeToAsset(assetId);

      const handleAssetUpdate = (updatedAsset: Asset) => {
        if (updatedAsset.id === assetId) {
          updateAsset(updatedAsset);
        }
      };

      on('asset_update', handleAssetUpdate);

      return () => {
        off('asset_update', handleAssetUpdate);
        unsubscribeFromAsset(assetId);
      };
    }
  }, [assetId, subscribeToAsset, unsubscribeFromAsset, on, off, updateAsset]);

  return asset;
};

// Hook for managing multiple assets subscriptions
export const useAssetsData = () => {
  const { subscribeToAsset, unsubscribeFromAsset, on, off } = useWebSocket();
  const updateAsset = useStore(state => state.updateAsset);
  const subscribedAssets = useRef<Set<string>>(new Set());

  const subscribeToAssets = useCallback((assetIds: string[]) => {
    // Unsubscribe from assets no longer needed
    subscribedAssets.current.forEach(assetId => {
      if (!assetIds.includes(assetId)) {
        unsubscribeFromAsset(assetId);
        subscribedAssets.current.delete(assetId);
      }
    });

    // Subscribe to new assets
    assetIds.forEach(assetId => {
      if (!subscribedAssets.current.has(assetId)) {
        subscribeToAsset(assetId);
        subscribedAssets.current.add(assetId);
      }
    });
  }, [subscribeToAsset, unsubscribeFromAsset]);

  const unsubscribeFromAssets = useCallback((assetIds: string[]) => {
    assetIds.forEach(assetId => {
      if (subscribedAssets.current.has(assetId)) {
        unsubscribeFromAsset(assetId);
        subscribedAssets.current.delete(assetId);
      }
    });
  }, [unsubscribeFromAsset]);

  useEffect(() => {
    const handleAssetUpdate = (updatedAsset: Asset) => {
      updateAsset(updatedAsset);
    };

    on('asset_update', handleAssetUpdate);

    return () => {
      off('asset_update', handleAssetUpdate);
      // Cleanup all subscriptions
      subscribedAssets.current.forEach(assetId => {
        unsubscribeFromAsset(assetId);
      });
      subscribedAssets.current.clear();
    };
  }, [on, off, updateAsset, unsubscribeFromAsset]);

  return {
    subscribeToAssets,
    unsubscribeFromAssets
  };
};

// Hook for real-time news updates
export const useNewsUpdates = (category?: string) => {
  const { subscribeToNews, on, off } = useWebSocket();
  const updateNews = useStore(state => state.updateNews);
  const news = useStore(state => state.news);

  useEffect(() => {
    subscribeToNews(category);

    const handleNewsUpdate = (data: { type: string; data: NewsItem; timestamp: string }) => {
      updateNews([data.data, ...news.slice(0, 99)]);
    };

    const handleBreakingNews = (newsItem: NewsItem) => {
      // Handle breaking news with higher priority
      updateNews([newsItem, ...news]);
    };

    on('news_update', handleNewsUpdate);
    on('breaking_news', handleBreakingNews);

    return () => {
      off('news_update', handleNewsUpdate);
      off('breaking_news', handleBreakingNews);
    };
  }, [category, subscribeToNews, on, off, updateNews, news]);

  return news;
};

// Hook for model predictions
export const useModelPredictions = (modelId?: string, assetId?: string) => {
  const { subscribeToModel, on, off } = useWebSocket();
  const addModelPrediction = useStore(state => state.addModelPrediction);
  const predictions = useStore(state => state.modelPredictions);

  useEffect(() => {
    if (modelId) {
      subscribeToModel(modelId);

      const handlePrediction = (prediction: any) => {
        if (!modelId || prediction.modelId === modelId) {
          if (!assetId || prediction.assetId === assetId) {
            addModelPrediction(prediction);
          }
        }
      };

      on('model_prediction', handlePrediction);

      return () => {
        off('model_prediction', handlePrediction);
      };
    }
  }, [modelId, assetId, subscribeToModel, on, off, addModelPrediction]);

  const filteredPredictions = predictions.filter(p => {
    if (modelId && p.modelId !== modelId) return false;
    if (assetId && p.assetId !== assetId) return false;
    return true;
  });

  return filteredPredictions;
};

// Hook for infinite scrolling
export const useInfiniteScroll = <T>(
  fetchMore: (offset: number) => Promise<T[]>,
  options: {
    threshold?: number;
    limit?: number;
    enabled?: boolean;
  } = {}
) => {
  const { threshold = 100, limit = 20, enabled = true } = options;
  const [items, setItems] = useState<T[]>([]);
  const [loading, setLoading] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const observerRef = useRef<IntersectionObserver | null>(null);
  const loadingRef = useRef<HTMLDivElement | null>(null);

  const loadMore = useCallback(async () => {
    if (loading || !hasMore || !enabled) return;

    try {
      setLoading(true);
      setError(null);
      const newItems = await fetchMore(items.length);
      
      if (newItems.length < limit) {
        setHasMore(false);
      }
      
      setItems(prev => [...prev, ...newItems]);
    } catch (err) {
      setError(err as Error);
    } finally {
      setLoading(false);
    }
  }, [fetchMore, items.length, limit, loading, hasMore, enabled]);

  useEffect(() => {
    if (!enabled) return;

    observerRef.current = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting) {
          loadMore();
        }
      },
      { threshold: 0.1 }
    );

    const currentRef = loadingRef.current;
    if (currentRef) {
      observerRef.current.observe(currentRef);
    }

    return () => {
      if (observerRef.current && currentRef) {
        observerRef.current.unobserve(currentRef);
      }
    };
  }, [loadMore, enabled]);

  const reset = useCallback(() => {
    setItems([]);
    setHasMore(true);
    setError(null);
  }, []);

  return {
    items,
    loading,
    hasMore,
    error,
    loadMore,
    reset,
    loadingRef
  };
};

// Hook for managing form state
export const useForm = <T extends Record<string, any>>(
  initialValues: T,
  validationRules?: Partial<Record<keyof T, (value: any) => string | null>>
) => {
  const [values, setValues] = useState<T>(initialValues);
  const [errors, setErrors] = useState<Partial<Record<keyof T, string>>>({});
  const [touched, setTouched] = useState<Partial<Record<keyof T, boolean>>>({});

  const setValue = useCallback((name: keyof T, value: any) => {
    setValues(prev => ({ ...prev, [name]: value }));
    
    // Clear error when user starts typing
    if (errors[name]) {
      setErrors(prev => ({ ...prev, [name]: undefined }));
    }
  }, [errors]);

  const setFieldTouched = useCallback((name: keyof T) => {
    setTouched(prev => ({ ...prev, [name]: true }));
  }, []);

  const validate = useCallback(() => {
    if (!validationRules) return true;

    const newErrors: Partial<Record<keyof T, string>> = {};
    let isValid = true;

    Object.keys(validationRules).forEach(key => {
      const rule = validationRules[key as keyof T];
      if (rule) {
        const error = rule(values[key as keyof T]);
        if (error) {
          newErrors[key as keyof T] = error;
          isValid = false;
        }
      }
    });

    setErrors(newErrors);
    return isValid;
  }, [values, validationRules]);

  const reset = useCallback(() => {
    setValues(initialValues);
    setErrors({});
    setTouched({});
  }, [initialValues]);

  const handleSubmit = useCallback((onSubmit: (values: T) => void | Promise<void>) => {
    return async (e?: React.FormEvent) => {
      e?.preventDefault();
      
      // Mark all fields as touched
      const allTouched = Object.keys(values).reduce((acc, key) => {
        acc[key as keyof T] = true;
        return acc;
      }, {} as Partial<Record<keyof T, boolean>>);
      setTouched(allTouched);

      if (validate()) {
        await onSubmit(values);
      }
    };
  }, [values, validate]);

  return {
    values,
    errors,
    touched,
    setValue,
    setTouched: setFieldTouched,
    validate,
    reset,
    handleSubmit,
    isValid: Object.keys(errors).length === 0
  };
};

// Hook for managing modal state
export const useModal = (initialOpen = false) => {
  const [isOpen, setIsOpen] = useState(initialOpen);

  const open = useCallback(() => setIsOpen(true), []);
  const close = useCallback(() => setIsOpen(false), []);
  const toggle = useCallback(() => setIsOpen(prev => !prev), []);

  return { isOpen, open, close, toggle };
};

// Hook for managing clipboard operations
export const useClipboard = () => {
  const [copied, setCopied] = useState(false);

  const copy = useCallback(async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
      return true;
    } catch (error) {
      console.error('Failed to copy to clipboard:', error);
      return false;
    }
  }, []);

  return { copied, copy };
};

// Hook for managing window size
export const useWindowSize = () => {
  const [windowSize, setWindowSize] = useState({
    width: typeof window !== 'undefined' ? window.innerWidth : 0,
    height: typeof window !== 'undefined' ? window.innerHeight : 0,
  });

  useEffect(() => {
    const handleResize = () => {
      setWindowSize({
        width: window.innerWidth,
        height: window.innerHeight,
      });
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return windowSize;
};

// Hook for managing dark mode
export const useDarkMode = () => {
  const [isDark, setIsDark] = useLocalStorage('darkMode', false);

  useEffect(() => {
    const root = window.document.documentElement;
    if (isDark) {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
  }, [isDark]);

  const toggle = useCallback(() => setIsDark(prev => !prev), [setIsDark]);
  const enable = useCallback(() => setIsDark(true), [setIsDark]);
  const disable = useCallback(() => setIsDark(false), [setIsDark]);

  return { isDark, toggle, enable, disable };
};

// Hook for managing previous value
export const usePrevious = <T>(value: T): T | undefined => {
  const ref = useRef<T>();
  
  useEffect(() => {
    ref.current = value;
  });
  
  return ref.current;
};

// Hook for managing interval
export const useInterval = (callback: () => void, delay: number | null) => {
  const savedCallback = useRef(callback);

  useEffect(() => {
    savedCallback.current = callback;
  }, [callback]);

  useEffect(() => {
    if (delay !== null) {
      const id = setInterval(() => savedCallback.current(), delay);
      return () => clearInterval(id);
    }
  }, [delay]);
};

// Hook for managing timeout
export const useTimeout = (callback: () => void, delay: number | null) => {
  const savedCallback = useRef(callback);

  useEffect(() => {
    savedCallback.current = callback;
  }, [callback]);

  useEffect(() => {
    if (delay !== null) {
      const id = setTimeout(() => savedCallback.current(), delay);
      return () => clearTimeout(id);
    }
  }, [delay]);
};

// Export all hooks
export default {
  useApi,
  useLocalStorage,
  useDebounce,
  useThrottle,
  useWebSocketConnection,
  useAssetData,
  useAssetsData,
  useNewsUpdates,
  useModelPredictions,
  useInfiniteScroll,
  useForm,
  useModal,
  useClipboard,
  useWindowSize,
  useDarkMode,
  usePrevious,
  useInterval,
  useTimeout,
};