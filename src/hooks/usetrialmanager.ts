"use client";

import { useState, useEffect, useCallback } from 'react';
import { useSubscription } from '../contexts/SubscriptionContext';

interface TrialStatus {
  isOnTrial: boolean;
  trialEndsAt: Date | null;
  daysRemaining: number;
  hasExpired: boolean;
  canStartTrial: boolean;
  message: string;
}

interface TrialManagerHook {
  trialStatus: TrialStatus;
  startTrial: (planId: string) => Promise<boolean>;
  checkTrialExpiration: () => void;
  getTrialMessage: () => string;
  isTrialExpiringSoon: () => boolean;
}

export const useTrialManager = (): TrialManagerHook => {
  const { subscription, subscribeToPlan, fetchCurrentSubscription } = useSubscription();
  const [trialStatus, setTrialStatus] = useState<TrialStatus>({
    isOnTrial: false,
    trialEndsAt: null,
    daysRemaining: 0,
    hasExpired: false,
    canStartTrial: true,
    message: 'No trial information available'
  });

  const fetchTrialStatus = useCallback(async () => {
    try {
      const token = localStorage.getItem('token');
      if (!token) return;
      
      const response = await fetch('/api/subscription/trial-status', {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      });
      
      if (response.ok) {
        const data = await response.json();
        setTrialStatus({
           isOnTrial: data.is_trial,
           trialEndsAt: data.trial_ends_at ? new Date(data.trial_ends_at) : null,
           daysRemaining: data.days_remaining,
           hasExpired: data.trial_expired,
           canStartTrial: data.can_start_trial,
           message: data.message
         });
        
        // If trial expired, refresh subscription data
        if (data.trial_expired) {
          window.location.reload();
        }
      }
    } catch (error) {
      console.error('Failed to fetch trial status:', error);
    }
  }, []);

  const startTrial = async (planId: string): Promise<boolean> => {
    try {
      const success = await subscribeToPlan(planId, true);
      if (success) {
        localStorage.setItem('has_used_trial', 'true');
        await fetchCurrentSubscription();
      }
      return success;
    } catch (error) {
      console.error('Error starting trial:', error);
      return false;
    }
  };

  const checkTrialExpiration = useCallback(() => {
    if (trialStatus.isOnTrial && trialStatus.hasExpired) {
      // Auto-downgrade logic is handled by the backend
      console.log('Trial has expired, auto-downgrading to free plan');
      
      // Refresh the page to get updated subscription data
      window.location.reload();
    }
  }, [trialStatus.isOnTrial, trialStatus.hasExpired]);

  const getTrialMessage = (): string => {
    return trialStatus.message || 'No trial information available';
  };

  const isTrialExpiringSoon = (): boolean => {
    return trialStatus.isOnTrial && trialStatus.daysRemaining <= 3;
  };

  // Update trial status when subscription changes
  useEffect(() => {
    fetchTrialStatus();
  }, [fetchTrialStatus]);

  // Check for trial expiration every hour
  useEffect(() => {
    const interval = setInterval(() => {
      fetchTrialStatus();
      checkTrialExpiration();
    }, 60 * 60 * 1000); // Check every hour

    return () => clearInterval(interval);
  }, [fetchTrialStatus, checkTrialExpiration]);

  // Check trial expiration on component mount
  useEffect(() => {
    if (trialStatus.isOnTrial) {
      checkTrialExpiration();
    }
  }, [checkTrialExpiration, trialStatus.isOnTrial]);

  return {
    trialStatus,
    startTrial,
    checkTrialExpiration,
    getTrialMessage,
    isTrialExpiringSoon
  };
};

export default useTrialManager;