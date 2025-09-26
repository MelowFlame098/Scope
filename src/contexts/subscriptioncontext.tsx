"use client";

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import axios from 'axios';

interface PlanFeatures {
  name: string;
  price: number;
  features: string[];
  max_portfolios: number;
  max_watchlists: number;
  advanced_analytics: boolean;
  ai_insights: boolean;
  social_trading: boolean;
  institutional_tools: boolean;
  priority_support: boolean;
}

interface SubscriptionPlan {
  id: string;
  name: string;
  price: number;
  features: string[];
  limits: {
    portfolios: number;
    watchlists: number;
  };
  capabilities: {
    advanced_analytics: boolean;
    ai_insights: boolean;
    social_trading: boolean;
    institutional_tools: boolean;
    priority_support: boolean;
  };
}

interface Subscription {
  plan: string;
  status: string;
  trial_ends_at?: string;
  subscription_ends_at?: string;
  auto_renew: boolean;
  created_at: string;
  updated_at: string;
}

interface SubscriptionContextType {
  subscription: Subscription | null;
  plans: SubscriptionPlan[];
  features: PlanFeatures | null;
  isLoading: boolean;
  error: string | null;
  fetchPlans: () => Promise<void>;
  fetchCurrentSubscription: () => Promise<void>;
  subscribeToPlan: (planId: string, trial?: boolean) => Promise<boolean>;
  cancelSubscription: () => Promise<boolean>;
  checkFeatureAccess: (feature: string) => Promise<boolean>;
  hasFeatureAccess: (feature: string) => boolean;
}

const SubscriptionContext = createContext<SubscriptionContextType | undefined>(undefined);

const API_BASE_URL = 'http://localhost:8000';

export const SubscriptionProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [subscription, setSubscription] = useState<Subscription | null>(null);
  const [plans, setPlans] = useState<SubscriptionPlan[]>([]);
  const [features, setFeatures] = useState<PlanFeatures | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const getAuthHeaders = () => {
    const token = localStorage.getItem('token');
    return token ? { Authorization: `Bearer ${token}` } : {};
  };

  const fetchPlans = async (retryCount = 0) => {
    try {
      setIsLoading(true);
      setError(null);
      const response = await axios.get(`${API_BASE_URL}/subscription/plans`);
      setPlans(response.data.plans);
    } catch (err: any) {
      // If it's a connection error and we haven't retried too many times, retry
      if ((err.code === 'ERR_NETWORK' || err.code === 'ERR_CONNECTION_REFUSED') && retryCount < 3) {
        console.log(`Retrying fetchPlans (attempt ${retryCount + 1}/3)...`);
        setTimeout(() => fetchPlans(retryCount + 1), 2000); // Retry after 2 seconds
        return;
      }
      setError(err.response?.data?.detail || 'Failed to fetch plans');
      console.error('Error fetching plans:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const fetchCurrentSubscription = async (retryCount = 0) => {
    try {
      setIsLoading(true);
      setError(null);
      const response = await axios.get(`${API_BASE_URL}/subscription/current`, {
        headers: getAuthHeaders()
      });
      setSubscription(response.data.subscription);
      setFeatures(response.data.features);
    } catch (err: any) {
      // If it's a connection error and we haven't retried too many times, retry
      if ((err.code === 'ERR_NETWORK' || err.code === 'ERR_CONNECTION_REFUSED') && retryCount < 3) {
        console.log(`Retrying fetchCurrentSubscription (attempt ${retryCount + 1}/3)...`);
        setTimeout(() => fetchCurrentSubscription(retryCount + 1), 2000); // Retry after 2 seconds
        return;
      }
      if (err.response?.status !== 401) {
        setError(err.response?.data?.detail || 'Failed to fetch subscription');
      }
      console.error('Error fetching subscription:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const subscribeToPlan = async (planId: string, trial: boolean = false): Promise<boolean> => {
    try {
      setIsLoading(true);
      setError(null);
      const response = await axios.post(
        `${API_BASE_URL}/subscription/subscribe`,
        { plan: planId, trial },
        { headers: getAuthHeaders() }
      );
      setSubscription(response.data.subscription);
      setFeatures(response.data.features);
      return true;
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to subscribe');
      console.error('Error subscribing:', err);
      return false;
    } finally {
      setIsLoading(false);
    }
  };

  const cancelSubscription = async (): Promise<boolean> => {
    try {
      setIsLoading(true);
      setError(null);
      await axios.post(`${API_BASE_URL}/subscription/cancel`, {}, {
        headers: getAuthHeaders()
      });
      await fetchCurrentSubscription(); // Refresh subscription data
      return true;
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to cancel subscription');
      console.error('Error canceling subscription:', err);
      return false;
    } finally {
      setIsLoading(false);
    }
  };

  const checkFeatureAccess = async (feature: string): Promise<boolean> => {
    try {
      const response = await axios.get(
        `${API_BASE_URL}/subscription/check-access/${feature}`,
        { headers: getAuthHeaders() }
      );
      return response.data.has_access;
    } catch (err: any) {
      console.error('Error checking feature access:', err);
      return false;
    }
  };

  const hasFeatureAccess = (feature: string): boolean => {
    if (!features) return false;
    
    switch (feature) {
      case 'advanced_analytics':
        return features.advanced_analytics;
      case 'ai_insights':
        return features.ai_insights;
      case 'social_trading':
        return features.social_trading;
      case 'institutional_tools':
        return features.institutional_tools;
      case 'priority_support':
        return features.priority_support;
      default:
        return false;
    }
  };

  // Load plans and subscription on mount, but not on auth pages
  useEffect(() => {
    // Don't make API calls on auth/landing pages to prevent blocking
    if (typeof window !== 'undefined') {
      const pathname = window.location.pathname;
      const isAuthPage = pathname.startsWith('/auth/') || pathname.startsWith('/landing') || pathname === '/';
      
      if (!isAuthPage) {
        fetchPlans();
        const token = localStorage.getItem('token');
        if (token) {
          fetchCurrentSubscription();
        }
      }
    }
  }, []);

  const value: SubscriptionContextType = {
    subscription,
    plans,
    features,
    isLoading,
    error,
    fetchPlans,
    fetchCurrentSubscription,
    subscribeToPlan,
    cancelSubscription,
    checkFeatureAccess,
    hasFeatureAccess
  };

  return (
    <SubscriptionContext.Provider value={value}>
      {children}
    </SubscriptionContext.Provider>
  );
};

export const useSubscription = (): SubscriptionContextType => {
  const context = useContext(SubscriptionContext);
  if (context === undefined) {
    throw new Error('useSubscription must be used within a SubscriptionProvider');
  }
  return context;
};

export default SubscriptionContext;