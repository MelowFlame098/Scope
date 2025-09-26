"use client";

import { useEffect, useRef, useState, useCallback } from 'react';
import { useStore } from '@/lib/store';

interface WebSocketMessage {
  type: string;
  data: any;
  timestamp: string;
}

interface WebSocketConnectionOptions {
  url?: string;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
  heartbeatInterval?: number;
}

interface WebSocketConnectionState {
  isConnected: boolean;
  isConnecting: boolean;
  error: string | null;
  reconnectAttempts: number;
}

export const useWebSocketConnection = (options: WebSocketConnectionOptions = {}) => {
  const {
    url = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws',
    reconnectInterval = 3000,
    maxReconnectAttempts = 5,
    heartbeatInterval = 30000
  } = options;

  const [connectionState, setConnectionState] = useState<WebSocketConnectionState>({
    isConnected: false,
    isConnecting: false,
    error: null,
    reconnectAttempts: 0
  });

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const heartbeatTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const isManuallyClosedRef = useRef(false);

  // Store actions
  const { 
    updateAsset, 
    updateNews, 
    addModelPrediction, 
    addNotification,
    setConnected 
  } = useStore();

  const clearTimeouts = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    if (heartbeatTimeoutRef.current) {
      clearTimeout(heartbeatTimeoutRef.current);
      heartbeatTimeoutRef.current = null;
    }
  }, []);

  const startHeartbeat = useCallback(() => {
    clearTimeouts();
    heartbeatTimeoutRef.current = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'ping' }));
      }
    }, heartbeatInterval);
  }, [heartbeatInterval, clearTimeouts]);

  const handleMessage = useCallback((event: MessageEvent) => {
    try {
      const message: WebSocketMessage = JSON.parse(event.data);
      
      switch (message.type) {
        case 'asset_update':
          updateAsset(message.data);
          break;
          
        case 'news_update':
          updateNews([message.data]);
          break;
          
        case 'model_prediction':
          addModelPrediction(message.data);
          break;
          
        case 'notification':
          addNotification({
            message: message.data.message,
            type: message.data.type || 'info',
            read: false
          });
          break;
          
        case 'pong':
          // Heartbeat response - connection is alive
          break;
          
        case 'error':
          console.error('WebSocket error message:', message.data);
          setConnectionState(prev => ({ ...prev, error: message.data.message }));
          break;
          
        default:
          console.log('Unknown WebSocket message type:', message.type);
      }
    } catch (error) {
      console.error('Failed to parse WebSocket message:', error);
    }
  }, [updateAsset, updateNews, addModelPrediction, addNotification]);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN || connectionState.isConnecting) {
      return;
    }

    setConnectionState(prev => ({ 
      ...prev, 
      isConnecting: true, 
      error: null 
    }));
    
    isManuallyClosedRef.current = false;

    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('WebSocket connected');
        setConnectionState({
          isConnected: true,
          isConnecting: false,
          error: null,
          reconnectAttempts: 0
        });
        setConnected(true);
        startHeartbeat();
        
        // Send authentication or initialization message if needed
        ws.send(JSON.stringify({ 
          type: 'subscribe', 
          data: { channels: ['market_data', 'news', 'notifications'] }
        }));
      };

      ws.onmessage = handleMessage;

      ws.onclose = (event) => {
        console.log('WebSocket disconnected:', event.code, event.reason);
        setConnectionState(prev => ({ 
          ...prev, 
          isConnected: false, 
          isConnecting: false 
        }));
        setConnected(false);
        clearTimeouts();
        
        // Attempt to reconnect if not manually closed
        if (!isManuallyClosedRef.current && 
            connectionState.reconnectAttempts < maxReconnectAttempts) {
          
          setConnectionState(prev => ({ 
            ...prev, 
            reconnectAttempts: prev.reconnectAttempts + 1 
          }));
          
          reconnectTimeoutRef.current = setTimeout(() => {
            console.log(`Attempting to reconnect... (${connectionState.reconnectAttempts + 1}/${maxReconnectAttempts})`);
            connect();
          }, reconnectInterval);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnectionState(prev => ({ 
          ...prev, 
          error: 'Connection error occurred',
          isConnecting: false 
        }));
      };

    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      setConnectionState(prev => ({ 
        ...prev, 
        error: 'Failed to create connection',
        isConnecting: false 
      }));
    }
  }, [url, connectionState.isConnecting, connectionState.reconnectAttempts, maxReconnectAttempts, reconnectInterval, handleMessage, startHeartbeat, setConnected, clearTimeouts]);

  const disconnect = useCallback(() => {
    isManuallyClosedRef.current = true;
    clearTimeouts();
    
    if (wsRef.current) {
      wsRef.current.close(1000, 'Manual disconnect');
      wsRef.current = null;
    }
    
    setConnectionState({
      isConnected: false,
      isConnecting: false,
      error: null,
      reconnectAttempts: 0
    });
    setConnected(false);
  }, [clearTimeouts, setConnected]);

  const sendMessage = useCallback((message: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
      return true;
    } else {
      console.warn('WebSocket is not connected. Message not sent:', message);
      return false;
    }
  }, []);

  const subscribe = useCallback((channels: string[]) => {
    return sendMessage({ 
      type: 'subscribe', 
      data: { channels } 
    });
  }, [sendMessage]);

  const unsubscribe = useCallback((channels: string[]) => {
    return sendMessage({ 
      type: 'unsubscribe', 
      data: { channels } 
    });
  }, [sendMessage]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnect();
    };
  }, [disconnect]);

  // Auto-connect on mount (optional)
  useEffect(() => {
    // Uncomment the line below if you want to auto-connect on mount
    // connect();
  }, []);

  return {
    // Connection state
    isConnected: connectionState.isConnected,
    isConnecting: connectionState.isConnecting,
    error: connectionState.error,
    reconnectAttempts: connectionState.reconnectAttempts,
    
    // Connection methods
    connect,
    disconnect,
    
    // Messaging methods
    sendMessage,
    subscribe,
    unsubscribe,
    
    // Connection info
    readyState: wsRef.current?.readyState || WebSocket.CLOSED,
    url
  };
};

export default useWebSocketConnection;