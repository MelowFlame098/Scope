import { redisService } from './RedisService';

export interface UserSession {
  userId: string;
  username: string;
  email: string;
  role: string;
  permissions: string[];
  loginTime: number;
  lastActivity: number;
  ipAddress: string;
  userAgent: string;
  deviceId?: string;
}

export interface SessionOptions {
  maxAge?: number; // Session duration in seconds (default: 24 hours)
  maxConcurrentSessions?: number; // Max concurrent sessions per user
  extendOnActivity?: boolean; // Extend session on activity
  trackActivity?: boolean; // Track user activity
}

class SessionService {
  private readonly SESSION_PREFIX = 'session:';
  private readonly USER_SESSIONS_PREFIX = 'user_sessions:';
  private readonly ACTIVE_SESSIONS_PREFIX = 'active_sessions';
  private readonly SESSION_ACTIVITY_PREFIX = 'session_activity:';
  
  private readonly defaultOptions: Required<SessionOptions> = {
    maxAge: 24 * 60 * 60, // 24 hours
    maxConcurrentSessions: 5,
    extendOnActivity: true,
    trackActivity: true
  };

  /**
   * Create a new user session
   */
  async createSession(
    sessionId: string,
    userSession: UserSession,
    options: SessionOptions = {}
  ): Promise<void> {
    const opts = { ...this.defaultOptions, ...options };
    const sessionKey = `${this.SESSION_PREFIX}${sessionId}`;
    const userSessionsKey = `${this.USER_SESSIONS_PREFIX}${userSession.userId}`;

    try {
      // Store session data
      await redisService.setHash(sessionKey, {
        userId: userSession.userId,
        username: userSession.username,
        email: userSession.email,
        role: userSession.role,
        permissions: JSON.stringify(userSession.permissions),
        loginTime: userSession.loginTime.toString(),
        lastActivity: userSession.lastActivity.toString(),
        ipAddress: userSession.ipAddress,
        userAgent: userSession.userAgent,
        deviceId: userSession.deviceId || ''
      });

      // Set session expiration
      await redisService.expire(sessionKey, opts.maxAge);

      // Track user sessions
      await redisService.addToSortedSet(userSessionsKey, sessionId, Date.now());
      await redisService.expire(userSessionsKey, opts.maxAge);

      // Add to active sessions
      await redisService.addToSortedSet(this.ACTIVE_SESSIONS_PREFIX, sessionId, Date.now());

      // Enforce concurrent session limit
      if (opts.maxConcurrentSessions > 0) {
        await this.enforceConcurrentSessionLimit(userSession.userId, opts.maxConcurrentSessions);
      }

      // Track session creation activity
      if (opts.trackActivity) {
        await this.trackActivity(sessionId, 'session_created', {
          ipAddress: userSession.ipAddress,
          userAgent: userSession.userAgent
        });
      }
    } catch (error) {
      console.error('Error creating session:', error);
      throw new Error('Failed to create session');
    }
  }

  /**
   * Get session data
   */
  async getSession(sessionId: string): Promise<UserSession | null> {
    try {
      const sessionKey = `${this.SESSION_PREFIX}${sessionId}`;
      const sessionData = await redisService.getHash(sessionKey);

      if (!sessionData || Object.keys(sessionData).length === 0) {
        return null;
      }

      return {
        userId: sessionData.userId,
        username: sessionData.username,
        email: sessionData.email,
        role: sessionData.role,
        permissions: JSON.parse(sessionData.permissions || '[]'),
        loginTime: parseInt(sessionData.loginTime),
        lastActivity: parseInt(sessionData.lastActivity),
        ipAddress: sessionData.ipAddress,
        userAgent: sessionData.userAgent,
        deviceId: sessionData.deviceId || undefined
      };
    } catch (error) {
      console.error('Error getting session:', error);
      return null;
    }
  }

  /**
   * Update session activity
   */
  async updateActivity(sessionId: string, options: SessionOptions = {}): Promise<boolean> {
    try {
      const opts = { ...this.defaultOptions, ...options };
      const sessionKey = `${this.SESSION_PREFIX}${sessionId}`;
      
      // Check if session exists
      const exists = await redisService.exists(sessionKey);
      if (!exists) {
        return false;
      }

      const now = Date.now();

      // Update last activity
      await redisService.setHashField(sessionKey, 'lastActivity', now.toString());

      // Extend session if configured
      if (opts.extendOnActivity) {
        await redisService.expire(sessionKey, opts.maxAge);
      }

      // Track activity
      if (opts.trackActivity) {
        await this.trackActivity(sessionId, 'activity_update');
      }

      return true;
    } catch (error) {
      console.error('Error updating session activity:', error);
      return false;
    }
  }

  /**
   * Destroy a session
   */
  async destroySession(sessionId: string): Promise<boolean> {
    try {
      const sessionKey = `${this.SESSION_PREFIX}${sessionId}`;
      
      // Get session data before deletion
      const session = await this.getSession(sessionId);
      if (!session) {
        return false;
      }

      // Remove session data
      await redisService.delete(sessionKey);

      // Remove from user sessions
      const userSessionsKey = `${this.USER_SESSIONS_PREFIX}${session.userId}`;
      await redisService.removeFromSortedSet(userSessionsKey, sessionId);

      // Remove from active sessions
      await redisService.removeFromSortedSet(this.ACTIVE_SESSIONS_PREFIX, sessionId);

      // Track session destruction
      await this.trackActivity(sessionId, 'session_destroyed');

      return true;
    } catch (error) {
      console.error('Error destroying session:', error);
      return false;
    }
  }

  /**
   * Get all active sessions for a user
   */
  async getUserSessions(userId: string): Promise<string[]> {
    try {
      const userSessionsKey = `${this.USER_SESSIONS_PREFIX}${userId}`;
      const sessions = await redisService.getSortedSetRange(userSessionsKey, 0, -1);
      
      // Filter out expired sessions
      const activeSessions: string[] = [];
      for (const sessionId of sessions) {
        const sessionKey = `${this.SESSION_PREFIX}${sessionId}`;
        const exists = await redisService.exists(sessionKey);
        if (exists) {
          activeSessions.push(sessionId);
        } else {
          // Clean up expired session from user sessions
          await redisService.removeFromSortedSet(userSessionsKey, sessionId);
        }
      }

      return activeSessions;
    } catch (error) {
      console.error('Error getting user sessions:', error);
      return [];
    }
  }

  /**
   * Destroy all sessions for a user
   */
  async destroyUserSessions(userId: string, excludeSessionId?: string): Promise<number> {
    try {
      const sessions = await this.getUserSessions(userId);
      let destroyedCount = 0;

      for (const sessionId of sessions) {
        if (excludeSessionId && sessionId === excludeSessionId) {
          continue;
        }

        const success = await this.destroySession(sessionId);
        if (success) {
          destroyedCount++;
        }
      }

      return destroyedCount;
    } catch (error) {
      console.error('Error destroying user sessions:', error);
      return 0;
    }
  }

  /**
   * Get session statistics
   */
  async getSessionStats(): Promise<{
    totalActiveSessions: number;
    sessionsLast24h: number;
    sessionsLastHour: number;
  }> {
    try {
      const now = Date.now();
      const last24h = now - (24 * 60 * 60 * 1000);
      const lastHour = now - (60 * 60 * 1000);

      const totalActiveSessions = await redisService.getSortedSetCount(this.ACTIVE_SESSIONS_PREFIX);
      const sessionsLast24h = await redisService.getSortedSetCountByScore(
        this.ACTIVE_SESSIONS_PREFIX,
        last24h,
        now
      );
      const sessionsLastHour = await redisService.getSortedSetCountByScore(
        this.ACTIVE_SESSIONS_PREFIX,
        lastHour,
        now
      );

      return {
        totalActiveSessions,
        sessionsLast24h,
        sessionsLastHour
      };
    } catch (error) {
      console.error('Error getting session stats:', error);
      return {
        totalActiveSessions: 0,
        sessionsLast24h: 0,
        sessionsLastHour: 0
      };
    }
  }

  /**
   * Clean up expired sessions
   */
  async cleanupExpiredSessions(): Promise<number> {
    try {
      const now = Date.now();
      const expiredThreshold = now - (this.defaultOptions.maxAge * 1000);
      
      // Get potentially expired sessions
      const expiredSessions = await redisService.getSortedSetRangeByScore(
        this.ACTIVE_SESSIONS_PREFIX,
        0,
        expiredThreshold
      );

      let cleanedCount = 0;
      for (const sessionId of expiredSessions) {
        const sessionKey = `${this.SESSION_PREFIX}${sessionId}`;
        const exists = await redisService.exists(sessionKey);
        
        if (!exists) {
          // Remove from active sessions
          await redisService.removeFromSortedSet(this.ACTIVE_SESSIONS_PREFIX, sessionId);
          cleanedCount++;
        }
      }

      return cleanedCount;
    } catch (error) {
      console.error('Error cleaning up expired sessions:', error);
      return 0;
    }
  }

  /**
   * Enforce concurrent session limit for a user
   */
  private async enforceConcurrentSessionLimit(userId: string, maxSessions: number): Promise<void> {
    try {
      const sessions = await this.getUserSessions(userId);
      
      if (sessions.length > maxSessions) {
        // Sort sessions by last activity (oldest first)
        const sessionDetails = await Promise.all(
          sessions.map(async (sessionId) => {
            const session = await this.getSession(sessionId);
            return { sessionId, lastActivity: session?.lastActivity || 0 };
          })
        );

        sessionDetails.sort((a, b) => a.lastActivity - b.lastActivity);

        // Remove oldest sessions
        const sessionsToRemove = sessionDetails.slice(0, sessions.length - maxSessions);
        for (const { sessionId } of sessionsToRemove) {
          await this.destroySession(sessionId);
        }
      }
    } catch (error) {
      console.error('Error enforcing concurrent session limit:', error);
    }
  }

  /**
   * Track session activity
   */
  private async trackActivity(
    sessionId: string,
    activityType: string,
    metadata: Record<string, any> = {}
  ): Promise<void> {
    try {
      const activityKey = `${this.SESSION_ACTIVITY_PREFIX}${sessionId}`;
      const activity = {
        type: activityType,
        timestamp: Date.now(),
        ...metadata
      };

      await redisService.addToList(activityKey, JSON.stringify(activity));
      
      // Keep only last 100 activities per session
      await redisService.trimList(activityKey, 0, 99);
      
      // Set expiration for activity log
      await redisService.expire(activityKey, this.defaultOptions.maxAge);
    } catch (error) {
      console.error('Error tracking session activity:', error);
    }
  }

  /**
   * Get session activity log
   */
  async getSessionActivity(sessionId: string): Promise<any[]> {
    try {
      const activityKey = `${this.SESSION_ACTIVITY_PREFIX}${sessionId}`;
      const activities = await redisService.getListRange(activityKey, 0, -1);
      
      return activities.map(activity => {
        try {
          return JSON.parse(activity);
        } catch {
          return { type: 'unknown', timestamp: 0 };
        }
      });
    } catch (error) {
      console.error('Error getting session activity:', error);
      return [];
    }
  }
}

export const sessionService = new SessionService();