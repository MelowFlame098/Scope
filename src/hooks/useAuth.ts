import { useState, useEffect, useCallback } from 'react';
import { sessionService, UserSession } from '@/services/SessionService';

export interface AuthUser {
  id: string;
  username: string;
  email: string;
  role: string;
  permissions: string[];
}

export interface AuthState {
  user: AuthUser | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  sessionId: string | null;
}

export interface LoginCredentials {
  email: string;
  password: string;
  rememberMe?: boolean;
}

export interface RegisterData {
  username: string;
  email: string;
  password: string;
  firstName?: string;
  lastName?: string;
}

export function useAuth() {
  const [authState, setAuthState] = useState<AuthState>({
    user: null,
    isAuthenticated: false,
    isLoading: true,
    sessionId: null
  });

  // Check for existing session on mount
  useEffect(() => {
    checkAuthStatus();
  }, []);

  const checkAuthStatus = useCallback(async () => {
    try {
      const token = getCookie('access_token');
      const sessionId = getCookie('session_id');

      if (!token || !sessionId) {
        setAuthState(prev => ({ ...prev, isLoading: false }));
        return;
      }

      // Validate session with Redis
      const sessionData = await sessionService.getSession(sessionId);
      
      if (sessionData) {
        // Update session activity
        await sessionService.updateActivity(sessionId);
        
        setAuthState({
          user: {
            id: sessionData.userId,
            username: sessionData.username,
            email: sessionData.email,
            role: sessionData.role,
            permissions: sessionData.permissions
          },
          isAuthenticated: true,
          isLoading: false,
          sessionId: sessionId
        });
      } else {
        // Session invalid, clear cookies
        clearAuthCookies();
        setAuthState({
          user: null,
          isAuthenticated: false,
          isLoading: false,
          sessionId: null
        });
      }
    } catch (error) {
      console.error('Auth check failed:', error);
      clearAuthCookies();
      setAuthState({
        user: null,
        isAuthenticated: false,
        isLoading: false,
        sessionId: null
      });
    }
  }, []);

  const login = useCallback(async (credentials: LoginCredentials): Promise<{ success: boolean; error?: string }> => {
    try {
      setAuthState(prev => ({ ...prev, isLoading: true }));

      // Call backend login API
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(credentials),
      });

      const data = await response.json();

      if (!response.ok) {
        return { success: false, error: data.message || 'Login failed' };
      }

      const { access_token, user, session_id } = data;

      // Set cookies
      setCookie('access_token', access_token, credentials.rememberMe ? 30 : 1);
      setCookie('session_id', session_id, credentials.rememberMe ? 30 : 1);

      // Create Redis session
      const userSession: UserSession = {
        userId: user.id,
        username: user.username,
        email: user.email,
        role: user.role,
        permissions: user.permissions || [],
        loginTime: Date.now(),
        lastActivity: Date.now(),
        ipAddress: await getClientIP(),
        userAgent: navigator.userAgent,
        deviceId: getDeviceId()
      };

      await sessionService.createSession(session_id, userSession, {
        maxAge: credentials.rememberMe ? 30 * 24 * 60 * 60 : 24 * 60 * 60, // 30 days or 1 day
        maxConcurrentSessions: 5,
        extendOnActivity: true,
        trackActivity: true
      });

      // Update auth state
      setAuthState({
        user: {
          id: user.id,
          username: user.username,
          email: user.email,
          role: user.role,
          permissions: user.permissions || []
        },
        isAuthenticated: true,
        isLoading: false,
        sessionId: session_id
      });

      return { success: true };
    } catch (error) {
      console.error('Login error:', error);
      setAuthState(prev => ({ ...prev, isLoading: false }));
      return { success: false, error: 'Network error. Please try again.' };
    }
  }, []);

  const register = useCallback(async (data: RegisterData): Promise<{ success: boolean; error?: string }> => {
    try {
      setAuthState(prev => ({ ...prev, isLoading: true }));

      const response = await fetch('/api/auth/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });

      const result = await response.json();

      if (!response.ok) {
        setAuthState(prev => ({ ...prev, isLoading: false }));
        return { success: false, error: result.message || 'Registration failed' };
      }

      setAuthState(prev => ({ ...prev, isLoading: false }));
      return { success: true };
    } catch (error) {
      console.error('Registration error:', error);
      setAuthState(prev => ({ ...prev, isLoading: false }));
      return { success: false, error: 'Network error. Please try again.' };
    }
  }, []);

  const logout = useCallback(async (logoutAllSessions = false): Promise<void> => {
    try {
      const { sessionId } = authState;

      if (sessionId) {
        if (logoutAllSessions && authState.user) {
          // Destroy all user sessions
          await sessionService.destroyUserSessions(authState.user.id);
        } else {
          // Destroy current session only
          await sessionService.destroySession(sessionId);
        }
      }

      // Call backend logout
      await fetch('/api/auth/logout', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${getCookie('access_token')}`,
        },
      });

      // Clear cookies and state
      clearAuthCookies();
      setAuthState({
        user: null,
        isAuthenticated: false,
        isLoading: false,
        sessionId: null
      });

      // Redirect to login
      window.location.href = '/auth/login';
    } catch (error) {
      console.error('Logout error:', error);
      // Still clear local state even if server call fails
      clearAuthCookies();
      setAuthState({
        user: null,
        isAuthenticated: false,
        isLoading: false,
        sessionId: null
      });
      window.location.href = '/auth/login';
    }
  }, [authState.sessionId, authState.user]);

  const refreshSession = useCallback(async (): Promise<boolean> => {
    try {
      if (!authState.sessionId) return false;

      const success = await sessionService.updateActivity(authState.sessionId);
      return success;
    } catch (error) {
      console.error('Session refresh failed:', error);
      return false;
    }
  }, [authState.sessionId]);

  const getActiveSessions = useCallback(async (): Promise<string[]> => {
    try {
      if (!authState.user) return [];
      return await sessionService.getUserSessions(authState.user.id);
    } catch (error) {
      console.error('Failed to get active sessions:', error);
      return [];
    }
  }, [authState.user]);

  const terminateSession = useCallback(async (sessionId: string): Promise<boolean> => {
    try {
      return await sessionService.destroySession(sessionId);
    } catch (error) {
      console.error('Failed to terminate session:', error);
      return false;
    }
  }, []);

  return {
    ...authState,
    login,
    register,
    logout,
    refreshSession,
    getActiveSessions,
    terminateSession,
    checkAuthStatus
  };
}

// Utility functions
function getCookie(name: string): string | null {
  if (typeof document === 'undefined') return null;
  
  const value = `; ${document.cookie}`;
  const parts = value.split(`; ${name}=`);
  if (parts.length === 2) return parts.pop()?.split(';').shift() || null;
  return null;
}

function setCookie(name: string, value: string, days: number): void {
  if (typeof document === 'undefined') return;
  
  const expires = new Date();
  expires.setTime(expires.getTime() + days * 24 * 60 * 60 * 1000);
  document.cookie = `${name}=${value};expires=${expires.toUTCString()};path=/;secure;samesite=strict`;
}

function clearAuthCookies(): void {
  if (typeof document === 'undefined') return;
  
  document.cookie = 'access_token=;expires=Thu, 01 Jan 1970 00:00:00 GMT;path=/';
  document.cookie = 'session_id=;expires=Thu, 01 Jan 1970 00:00:00 GMT;path=/';
}

async function getClientIP(): Promise<string> {
  try {
    const response = await fetch('/api/client-ip');
    const data = await response.json();
    return data.ip || 'unknown';
  } catch {
    return 'unknown';
  }
}

function getDeviceId(): string {
  if (typeof window === 'undefined') return 'server';
  
  let deviceId = localStorage.getItem('device_id');
  if (!deviceId) {
    deviceId = `device_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    localStorage.setItem('device_id', deviceId);
  }
  return deviceId;
}