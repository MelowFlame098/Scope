"use client";

import React from 'react';
import { useTrialManager } from '../hooks/useTrialManager';
import { useSubscription } from '../contexts/SubscriptionContext';

const TrialBanner: React.FC = () => {
  const { trialStatus, getTrialMessage, isTrialExpiringSoon } = useTrialManager();
  const { subscribeToPlan } = useSubscription();

  if (!trialStatus.isOnTrial && !trialStatus.hasExpired) {
    return null;
  }

  const handleUpgrade = async (planId: string) => {
    await subscribeToPlan(planId, false);
  };

  const getBannerStyle = () => {
    if (trialStatus.hasExpired) {
      return 'bg-red-50 border-red-200 text-red-800';
    }
    if (isTrialExpiringSoon()) {
      return 'bg-yellow-50 border-yellow-200 text-yellow-800';
    }
    return 'bg-blue-50 border-blue-200 text-blue-800';
  };

  const getIconColor = () => {
    if (trialStatus.hasExpired) return 'text-red-500';
    if (isTrialExpiringSoon()) return 'text-yellow-500';
    return 'text-blue-500';
  };

  return (
    <div className={`border rounded-lg p-4 mb-6 ${getBannerStyle()}`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className={`flex-shrink-0 ${getIconColor()}`}>
            {trialStatus.hasExpired ? (
              <svg className="h-5 w-5" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
            ) : (
              <svg className="h-5 w-5" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clipRule="evenodd" />
              </svg>
            )}
          </div>
          <div>
            <p className="font-medium">
              {trialStatus.hasExpired ? 'Trial Expired' : 'Trial Active'}
            </p>
            <p className="text-sm">
              {getTrialMessage()}
            </p>
          </div>
        </div>
        
        <div className="flex space-x-2">
          <button
            onClick={() => handleUpgrade('basic')}
            className="px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
          >
            Upgrade to Basic ($4.99)
          </button>
          <button
            onClick={() => handleUpgrade('premium')}
            className="px-4 py-2 bg-purple-600 text-white text-sm font-medium rounded-md hover:bg-purple-700 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2"
          >
            Upgrade to Premium ($9.99)
          </button>
        </div>
      </div>
      
      {trialStatus.isOnTrial && !trialStatus.hasExpired && (
        <div className="mt-3">
          <div className="bg-white bg-opacity-50 rounded-full h-2">
            <div 
              className="bg-current h-2 rounded-full transition-all duration-300"
              style={{ 
                width: `${Math.max(0, (30 - trialStatus.daysRemaining) / 30 * 100)}%` 
              }}
            />
          </div>
          <p className="text-xs mt-1">
            Trial progress: {30 - trialStatus.daysRemaining} of 30 days used
          </p>
        </div>
      )}
    </div>
  );
};

export default TrialBanner;