import React from 'react';

/**
 * Performance monitoring utilities for the trading platform
 */

// Performance metrics interface
export interface PerformanceMetrics {
  loadTime: number;
  renderTime: number;
  interactionTime: number;
  memoryUsage: number;
  bundleSize: number;
}

// Web Vitals tracking
export interface WebVitals {
  FCP: number; // First Contentful Paint
  LCP: number; // Largest Contentful Paint
  FID: number; // First Input Delay
  CLS: number; // Cumulative Layout Shift
  TTFB: number; // Time to First Byte
}

/**
 * Performance monitor class for tracking application performance
 */
export class PerformanceMonitor {
  private metrics: Partial<PerformanceMetrics> = {};
  private vitals: Partial<WebVitals> = {};
  private observers: PerformanceObserver[] = [];

  constructor() {
    this.initializeObservers();
  }

  /**
   * Initialize performance observers
   */
  private initializeObservers(): void {
    if (typeof window === 'undefined') return;

    // Navigation timing observer
    if ('PerformanceObserver' in window) {
      const navObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        entries.forEach((entry) => {
          if (entry.entryType === 'navigation') {
            const navEntry = entry as PerformanceNavigationTiming;
            this.metrics.loadTime = navEntry.loadEventEnd - navEntry.loadEventStart;
            this.vitals.TTFB = navEntry.responseStart - navEntry.requestStart;
          }
        });
      });

      try {
        navObserver.observe({ entryTypes: ['navigation'] });
        this.observers.push(navObserver);
      } catch (error) {
        console.warn('Navigation timing observer not supported:', error);
      }

      // Paint timing observer
      const paintObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        entries.forEach((entry) => {
          if (entry.name === 'first-contentful-paint') {
            this.vitals.FCP = entry.startTime;
          }
        });
      });

      try {
        paintObserver.observe({ entryTypes: ['paint'] });
        this.observers.push(paintObserver);
      } catch (error) {
        console.warn('Paint timing observer not supported:', error);
      }

      // Layout shift observer
      const layoutObserver = new PerformanceObserver((list) => {
        let clsValue = 0;
        const entries = list.getEntries();
        entries.forEach((entry: any) => {
          if (!entry.hadRecentInput) {
            clsValue += entry.value;
          }
        });
        this.vitals.CLS = clsValue;
      });

      try {
        layoutObserver.observe({ entryTypes: ['layout-shift'] });
        this.observers.push(layoutObserver);
      } catch (error) {
        console.warn('Layout shift observer not supported:', error);
      }
    }
  }

  /**
   * Measure component render time
   */
  measureRenderTime(componentName: string, renderFn: () => void): number {
    const startTime = performance.now();
    renderFn();
    const endTime = performance.now();
    const renderTime = endTime - startTime;
    
    console.log(`${componentName} render time: ${renderTime.toFixed(2)}ms`);
    return renderTime;
  }

  /**
   * Measure API call performance
   */
  async measureApiCall<T>(
    apiName: string,
    apiCall: () => Promise<T>
  ): Promise<{ result: T; duration: number }> {
    const startTime = performance.now();
    try {
      const result = await apiCall();
      const endTime = performance.now();
      const duration = endTime - startTime;
      
      console.log(`API call ${apiName} completed in ${duration.toFixed(2)}ms`);
      return { result, duration };
    } catch (error) {
      const endTime = performance.now();
      const duration = endTime - startTime;
      console.error(`API call ${apiName} failed after ${duration.toFixed(2)}ms:`, error);
      throw error;
    }
  }

  /**
   * Get memory usage information
   */
  getMemoryUsage(): any {
    if (typeof window !== 'undefined' && 'memory' in performance) {
      return {
        usedJSHeapSize: (performance as any).memory.usedJSHeapSize,
        totalJSHeapSize: (performance as any).memory.totalJSHeapSize,
        jsHeapSizeLimit: (performance as any).memory.jsHeapSizeLimit,
      };
    }
    return null;
  }

  /**
   * Track user interaction timing
   */
  trackInteraction(interactionName: string, callback: () => void): void {
    const startTime = performance.now();
    callback();
    const endTime = performance.now();
    const duration = endTime - startTime;
    
    console.log(`Interaction ${interactionName} took ${duration.toFixed(2)}ms`);
  }

  /**
   * Get current performance metrics
   */
  getMetrics(): { metrics: Partial<PerformanceMetrics>; vitals: Partial<WebVitals> } {
    return {
      metrics: { ...this.metrics },
      vitals: { ...this.vitals },
    };
  }

  /**
   * Send performance data to analytics
   */
  sendToAnalytics(endpoint: string): void {
    const data = this.getMetrics();
    
    if (navigator.sendBeacon) {
      navigator.sendBeacon(endpoint, JSON.stringify(data));
    } else {
      fetch(endpoint, {
        method: 'POST',
        body: JSON.stringify(data),
        headers: {
          'Content-Type': 'application/json',
        },
      }).catch((error) => {
        console.error('Failed to send performance data:', error);
      });
    }
  }

  /**
   * Clean up observers
   */
  cleanup(): void {
    this.observers.forEach((observer) => {
      observer.disconnect();
    });
    this.observers = [];
  }
}

/**
 * React hook for performance monitoring
 */
export function usePerformanceMonitor() {
  const monitor = new PerformanceMonitor();

  return {
    measureRender: monitor.measureRenderTime.bind(monitor),
    measureApi: monitor.measureApiCall.bind(monitor),
    trackInteraction: monitor.trackInteraction.bind(monitor),
    getMetrics: monitor.getMetrics.bind(monitor),
    getMemoryUsage: monitor.getMemoryUsage.bind(monitor),
    sendToAnalytics: monitor.sendToAnalytics.bind(monitor),
    cleanup: monitor.cleanup.bind(monitor),
  };
}

/**
 * Debounce function for performance optimization
 */
export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout;
  return (...args: Parameters<T>) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => func.apply(null, args), wait);
  };
}

/**
 * Throttle function for performance optimization
 */
export function throttle<T extends (...args: any[]) => any>(
  func: T,
  limit: number
): (...args: Parameters<T>) => void {
  let inThrottle: boolean;
  return (...args: Parameters<T>) => {
    if (!inThrottle) {
      func.apply(null, args);
      inThrottle = true;
      setTimeout(() => (inThrottle = false), limit);
    }
  };
}

/**
 * Lazy loading utility for components
 */
export function createLazyComponent<T extends React.ComponentType<any>>(
  importFn: () => Promise<{ default: T }>
): React.LazyExoticComponent<T> {
  return React.lazy(importFn);
}

// Global performance monitor instance
export const globalPerformanceMonitor = new PerformanceMonitor();