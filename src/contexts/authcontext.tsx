"use client";

import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import { useRouter } from 'next/navigation';
import { sessionService, UserSession } from '@/services/SessionService';

interface User {
  id: string;
  email: string;
  firstName: string;
  lastName: string;
  role: string;
  isEmailVerified: boolean;
  createdAt: string;
  lastLoginAt?: string;
  subscriptionPlan: 'free' | 'basic' | 'premium';
  subscriptionStatus: 'active' | 'trial' | 'expired' | 'cancelled';
  trialEndsAt?: string;
  subscriptionEndsAt?: string;
}

interface AuthContextType {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  sessionId: string | null;
  login: (email: string, password: string, captcha?: string, rememberMe?: boolean) => Promise<{ success: boolean; error?: string; requiresCaptcha?: boolean }>;
  register: (userData: RegisterData) => Promise<{ success: boolean; error?: string }>;
  logout: (logoutAllSessions?: boolean) => void;
  refreshToken: () => Promise<boolean>;
  updateUser: (userData: Partial<User>) => void;
  resendVerificationEmail: () => Promise<{ success: boolean; error?: string }>;
  forgotPassword: (email: string, captcha?: string) => Promise<{ success: boolean; error?: string; requiresCaptcha?: boolean }>;
  resetPassword: (token: string, newPassword: string) => Promise<{ success: boolean; error?: string }>;
  verifyEmail: (token: string) => Promise<{ success: boolean; error?: string }>;
  hasFeatureAccess: (feature: string) => boolean;
  checkSubscriptionAccess: (requiredPlan: 'free' | 'basic' | 'premium') => boolean;
  getSubscriptionLimits: () => SubscriptionLimits;
  refreshSubscription: () => Promise<void>;
  getActiveSessions: () => Promise<string[]>;
  terminateSession: (sessionId: string) => Promise<boolean>;
  refreshSession: () => Promise<boolean>;
}

interface SubscriptionLimits {
  maxPortfolios: number;
  maxWatchlists: number;
  advancedAnalytics: boolean;
  aiInsights: boolean;
  socialTrading: boolean;
  institutionalTools: boolean;
  prioritySupport: boolean;
}

interface RegisterData {
  email: string;
  password: string;
  firstName: string;
  lastName: string;
  captcha?: string;
  acceptTerms: boolean;
  acceptPrivacy: boolean;
  marketingEmails: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const router = useRouter();

  const isAuthenticated = !!user;

  // Helper function to set cookies
  const setCookie = (name: string, value: string, days: number = 7) => {
    const expires = new Date();
    expires.setTime(expires.getTime() + (days * 24 * 60 * 60 * 1000));
    document.cookie = `${name}=${value};expires=${expires.toUTCString()};path=/;SameSite=Lax`;
  };

  // Helper function to remove cookies
  const removeCookie = (name: string) => {
    document.cookie = `${name}=;expires=Thu, 01 Jan 1970 00:00:00 UTC;path=/;`;
  };

  // Check authentication status on mount, but not on auth pages
  useEffect(() => {
    // Don't check auth status on auth/landing pages to prevent blocking
    if (typeof window !== 'undefined') {
      const pathname = window.location.pathname;
      const isAuthPage = pathname.startsWith('/auth/') || pathname.startsWith('/landing') || pathname === '/';
      
      if (!isAuthPage) {
        checkAuthStatus();
      } else {
        setIsLoading(false);
      }
    }
  }, []);

  const checkAuthStatus = async () => {
    try {
      // Check authentication via the /api/auth/me endpoint
      // This will use the HTTP-only cookie automatically
      const response = await fetch('/api/auth/me', {
        method: 'GET',
        credentials: 'include', // Include cookies in the request
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (response.ok) {
        const result = await response.json();
        if (result.success && result.data) {
          // Transform subscription plan to lowercase to match frontend expectations
          const userData = {
            ...result.data,
            subscriptionPlan: result.data.subscription_plan?.toLowerCase() || 'free'
          };
          setUser(userData);
        } else {
          setUser(null);
        }
      } else {
        // Not authenticated or token is invalid
        setUser(null);
        // Clear any localStorage tokens (legacy)
        localStorage.removeItem('access_token');
        localStorage.removeItem('refresh_token');
      }
    } catch (error) {
      console.error('Auth check failed:', error);
      setUser(null);
      // Clear any localStorage tokens (legacy)
      localStorage.removeItem('access_token');
      localStorage.removeItem('refresh_token');
    } finally {
      setIsLoading(false);
    }
  };

  const login = async (email: string, password: string, captcha?: string, rememberMe?: boolean) => {
    try {
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        credentials: 'include', // Include cookies in the request
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          email,
          password,
          captcha_token: captcha,
          remember_me: rememberMe,
        }),
      });

      const data = await response.json();

      if (response.ok) {
        // The token is now stored as an HTTP-only cookie by the API
        // Check authentication status to get user data
        await checkAuthStatus();
        
        // Create Redis session after successful login
        if (user) {
          const sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
          const userSession: UserSession = {
            userId: user.id,
            username: user.email, // Using email as username since User interface doesn't have username
            email: user.email,
            role: user.role || 'user',
            permissions: [], // User interface doesn't have permissions property
            loginTime: Date.now(),
            lastActivity: Date.now(),
            userAgent: navigator.userAgent,
            ipAddress: 'client-side' // Will be set by server
          };
          
          await sessionService.createSession(sessionId, userSession);
          setSessionId(sessionId);
        }
        
        return { success: true };
      } else {
        return {
          success: false,
          error: data.error || 'Login failed',
          requiresCaptcha: data.requires_captcha,
        };
      }
    } catch (error) {
      console.error('Login error:', error);
      return {
        success: false,
        error: 'Network error. Please try again.',
      };
    }
  };

  const register = async (userData: RegisterData) => {
    try {
      // Parse CAPTCHA token to get ID and answer
      const captchaParts = userData.captcha?.split(':') || [];
      const captcha_id = captchaParts[0] || '';
      const captcha_answer = captchaParts[1] || '';
      
      const response = await fetch('/api/auth/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          email: userData.email,
          password: userData.password,
          first_name: userData.firstName,
          last_name: userData.lastName,
          captcha_id: captcha_id,
          captcha_answer: captcha_answer,
          acceptTerms: userData.acceptTerms,
          acceptPrivacy: userData.acceptPrivacy,
          marketingEmails: userData.marketingEmails,
        }),
      });

      const data = await response.json();

      if (response.ok) {
        // After successful registration, automatically log in the user
        const loginResult = await login(userData.email, userData.password);
        if (loginResult.success) {
          return { success: true };
        } else {
          // Registration succeeded but login failed
          return {
            success: true,
            message: 'Registration successful. Please log in manually.',
          };
        }
      } else {
        return {
          success: false,
          error: data.detail || 'Registration failed',
        };
      }
    } catch (error) {
      console.error('Registration error:', error);
      return {
        success: false,
        error: 'Network error. Please try again.',
      };
    }
  };

  const logout = async (logoutAllSessions?: boolean) => {
    try {
      // Destroy Redis session(s)
      if (sessionId) {
        if (logoutAllSessions && user) {
          // Destroy all sessions for the user
          const sessions = await sessionService.getUserSessions(user.id);
          await Promise.all(sessions.map(sid => sessionService.destroySession(sid)));
        } else {
          // Destroy current session only
          await sessionService.destroySession(sessionId);
        }
        setSessionId(null);
      }
      
      // Call logout API to clear HTTP-only cookies
      await fetch('/api/auth/logout', {
        method: 'POST',
        credentials: 'include',
      });
    } catch (error) {
      console.error('Logout API error:', error);
    }
    
    // Clear any legacy localStorage tokens and user data
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    setUser(null);
    
    // Redirect to landing page
    router.push('/landing');
  };

  const refreshToken = async (): Promise<boolean> => {
    try {
      const response = await fetch('/api/auth/refresh', {
        method: 'POST',
        credentials: 'include', // Include cookies for refresh token
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (response.ok) {
        // Token refreshed successfully (new token set as HTTP-only cookie)
        // Check authentication status to update user data
        await checkAuthStatus();
        return true;
      } else {
        // Refresh failed, clear user state
        setUser(null);
        localStorage.removeItem('access_token');
        localStorage.removeItem('refresh_token');
        return false;
      }
    } catch (error) {
      console.error('Token refresh error:', error);
      setUser(null);
      return false;
    }
  };

  const updateUser = (userData: Partial<User>) => {
    if (user) {
      setUser({ ...user, ...userData });
    }
  };

  const resendVerificationEmail = async () => {
    try {
      const response = await fetch('/api/auth/resend-verification', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('access_token')}`,
        },
      });

      const data = await response.json();

      if (response.ok) {
        return { success: true };
      } else {
        return {
          success: false,
          error: data.detail || 'Failed to resend verification email',
        };
      }
    } catch (error) {
      console.error('Resend verification error:', error);
      return {
        success: false,
        error: 'Network error. Please try again.',
      };
    }
  };

  const forgotPassword = async (email: string, captcha?: string) => {
    try {
      const response = await fetch('/api/auth/password-reset', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          email,
          captcha_token: captcha,
        }),
      });

      const data = await response.json();

      if (response.ok) {
        return { success: true };
      } else {
        return {
          success: false,
          error: data.detail || 'Failed to send reset email',
          requiresCaptcha: data.requires_captcha,
        };
      }
    } catch (error) {
      console.error('Forgot password error:', error);
      return {
        success: false,
        error: 'Network error. Please try again.',
      };
    }
  };

  const resetPassword = async (token: string, newPassword: string) => {
    try {
      const response = await fetch('/api/auth/reset-password-confirm', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          token,
          new_password: newPassword,
        }),
      });

      const data = await response.json();

      if (response.ok) {
        return { success: true };
      } else {
        return {
          success: false,
          error: data.detail || 'Failed to reset password',
        };
      }
    } catch (error) {
      console.error('Reset password error:', error);
      return {
        success: false,
        error: 'Network error. Please try again.',
      };
    }
  };

  const verifyEmail = async (token: string) => {
    try {
      const response = await fetch('/api/auth/verify-email', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          token,
        }),
      });

      const data = await response.json();

      if (response.ok) {
        // Update user verification status if user is logged in
        if (user) {
          setUser({ ...user, isEmailVerified: true });
        }
        return { success: true };
      } else {
        return {
          success: false,
          error: data.detail || 'Email verification failed',
        };
      }
    } catch (error) {
      console.error('Email verification error:', error);
      return {
        success: false,
        error: 'Network error. Please try again.',
      };
    }
  };

  // Subscription access control functions
  const hasFeatureAccess = (feature: string): boolean => {
    if (!user) return false;
    
    const plan = user.subscriptionPlan;
    const status = user.subscriptionStatus;
    
    // Check if subscription is active or in trial
    if (status === 'expired' || status === 'cancelled') {
      return plan === 'free' && isFeatureAvailableForFree(feature);
    }
    
    switch (plan) {
      case 'free':
        return isFeatureAvailableForFree(feature);
      case 'basic':
        return isFeatureAvailableForFree(feature) || isFeatureAvailableForBasic(feature);
      case 'premium':
        return true; // Premium has access to all features
      default:
        return false;
    }
  };

  const isFeatureAvailableForFree = (feature: string): boolean => {
    const freeFeatures = ['basic_market_data', 'portfolio_tracking', 'basic_charts'];
    return freeFeatures.includes(feature);
  };

  const isFeatureAvailableForBasic = (feature: string): boolean => {
    const basicFeatures = ['real_time_data', 'advanced_charts', 'price_alerts', 'technical_indicators'];
    return basicFeatures.includes(feature);
  };

  const checkSubscriptionAccess = (requiredPlan: 'free' | 'basic' | 'premium'): boolean => {
    if (!user) return false;
    
    const planHierarchy = { free: 0, basic: 1, premium: 2 };
    const userPlanLevel = planHierarchy[user.subscriptionPlan];
    const requiredPlanLevel = planHierarchy[requiredPlan];
    
    // Check if subscription is active
    if (user.subscriptionStatus === 'expired' || user.subscriptionStatus === 'cancelled') {
      return user.subscriptionPlan === 'free' && requiredPlan === 'free';
    }
    
    return userPlanLevel >= requiredPlanLevel;
  };

  const getSubscriptionLimits = (): SubscriptionLimits => {
    if (!user) {
      return {
        maxPortfolios: 1,
        maxWatchlists: 1,
        advancedAnalytics: false,
        aiInsights: false,
        socialTrading: false,
        institutionalTools: false,
        prioritySupport: false,
      };
    }
    
    switch (user.subscriptionPlan) {
      case 'free':
        return {
          maxPortfolios: 1,
          maxWatchlists: 3,
          advancedAnalytics: false,
          aiInsights: false,
          socialTrading: false,
          institutionalTools: false,
          prioritySupport: false,
        };
      case 'basic':
        return {
          maxPortfolios: 5,
          maxWatchlists: 10,
          advancedAnalytics: true,
          aiInsights: false,
          socialTrading: true,
          institutionalTools: false,
          prioritySupport: false,
        };
      case 'premium':
        return {
          maxPortfolios: -1, // Unlimited
          maxWatchlists: -1, // Unlimited
          advancedAnalytics: true,
          aiInsights: true,
          socialTrading: true,
          institutionalTools: true,
          prioritySupport: true,
        };
      default:
        return getSubscriptionLimits(); // Default to free
    }
  };

  const refreshSubscription = async (): Promise<void> => {
    if (!user) return;
    
    try {
      const token = localStorage.getItem('access_token');
      const response = await fetch('/api/auth/subscription', {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
      });
      
      if (response.ok) {
        const subscriptionData = await response.json();
        setUser({
          ...user,
          subscriptionPlan: subscriptionData.plan,
          subscriptionStatus: subscriptionData.status,
          trialEndsAt: subscriptionData.trial_ends_at,
          subscriptionEndsAt: subscriptionData.subscription_ends_at,
        });
      }
    } catch (error) {
      console.error('Failed to refresh subscription:', error);
    }
  };

  // Session management methods
  const getActiveSessions = async (): Promise<string[]> => {
    if (!user) return [];
    try {
      return await sessionService.getUserSessions(user.id);
    } catch (error) {
      console.error('Failed to get active sessions:', error);
      return [];
    }
  };

  const terminateSession = async (sessionId: string): Promise<boolean> => {
    try {
      await sessionService.destroySession(sessionId);
      return true;
    } catch (error) {
      console.error('Failed to terminate session:', error);
      return false;
    }
  };

  const refreshSession = async (): Promise<boolean> => {
    if (!sessionId) return false;
    try {
      await sessionService.updateActivity(sessionId);
      return true;
    } catch (error) {
      console.error('Failed to refresh session:', error);
      return false;
    }
  };

  const value: AuthContextType = {
    user,
    isLoading,
    isAuthenticated,
    sessionId,
    login,
    register,
    logout,
    refreshToken,
    updateUser,
    resendVerificationEmail,
    forgotPassword,
    resetPassword,
    verifyEmail,
    hasFeatureAccess,
    checkSubscriptionAccess,
    getSubscriptionLimits,
    refreshSubscription,
    getActiveSessions,
    terminateSession,
    refreshSession,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};