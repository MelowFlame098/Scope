"use client";

import React, { useState, useEffect } from 'react';
import { useSubscription } from '../contexts/SubscriptionContext';
import { useTrialManager } from '../hooks/useTrialManager';
import { CheckIcon, XMarkIcon, SparklesIcon, StarIcon } from '@heroicons/react/24/outline';
import FeatureLimits from './FeatureLimits';
import TrialBanner from './TrialBanner';

const Subscription: React.FC = () => {
  const {
    subscription,
    plans,
    features,
    isLoading,
    error,
    subscribeToPlan,
    cancelSubscription,
    fetchCurrentSubscription
  } = useSubscription();
  
  const { trialStatus, startTrial } = useTrialManager();

  const [selectedPlan, setSelectedPlan] = useState<string | null>(null);
  const [showCancelConfirm, setShowCancelConfirm] = useState(false);
  const [actionLoading, setActionLoading] = useState(false);

  useEffect(() => {
    fetchCurrentSubscription();
  }, []);

  const handleSubscribe = async (planId: string, trial: boolean = false) => {
    setActionLoading(true);
    let success;
    if (trial) {
      success = await startTrial(planId);
    } else {
      success = await subscribeToPlan(planId, false);
    }
    if (success) {
      setSelectedPlan(null);
    }
    setActionLoading(false);
  };

  const handleCancel = async () => {
    setActionLoading(true);
    const success = await cancelSubscription();
    if (success) {
      setShowCancelConfirm(false);
    }
    setActionLoading(false);
  };

  const getPlanIcon = (planId: string) => {
    switch (planId) {
      case 'free':
        return <StarIcon className="w-8 h-8 text-gray-500" />;
      case 'basic':
        return <SparklesIcon className="w-8 h-8 text-blue-500" />;
      case 'premium':
        return <StarIcon className="w-8 h-8 text-purple-500" />;
      default:
        return <StarIcon className="w-8 h-8 text-gray-500" />;
    }
  };

  const getPlanColor = (planId: string) => {
    switch (planId) {
      case 'free':
        return 'border-gray-200 bg-gray-50';
      case 'basic':
        return 'border-blue-200 bg-blue-50';
      case 'premium':
        return 'border-purple-200 bg-purple-50';
      default:
        return 'border-gray-200 bg-gray-50';
    }
  };

  const getButtonColor = (planId: string) => {
    switch (planId) {
      case 'free':
        return 'bg-gray-600 hover:bg-gray-700';
      case 'basic':
        return 'bg-blue-600 hover:bg-blue-700';
      case 'premium':
        return 'bg-purple-600 hover:bg-purple-700';
      default:
        return 'bg-gray-600 hover:bg-gray-700';
    }
  };

  const isCurrentPlan = (planId: string) => {
    return subscription?.plan === planId;
  };

  const canStartTrial = (planId: string) => {
    return planId !== 'free' && trialStatus.canStartTrial;
  };

  const formatTrialEndDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    });
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto p-6">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">Choose Your Plan</h1>
        <p className="text-lg text-gray-600 max-w-2xl mx-auto">
          Unlock powerful features and take your trading to the next level with our subscription plans.
        </p>
      </div>

      {error && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-red-600">{error}</p>
        </div>
      )}

      <TrialBanner />

      {/* Current Subscription Status */}
      {subscription && (
        <div className="mb-8 p-6 bg-white border border-gray-200 rounded-lg shadow-sm">
          <h2 className="text-xl font-semibold mb-4">Current Subscription</h2>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-lg font-medium capitalize">{subscription.plan} Plan</p>
              <p className="text-gray-600 capitalize">Status: {subscription.status}</p>
              {subscription.trial_ends_at && subscription.status === 'trial' && (
                <p className="text-orange-600">
                  Trial ends: {formatTrialEndDate(subscription.trial_ends_at)}
                </p>
              )}
            </div>
            {subscription.plan !== 'free' && (
              <button
                onClick={() => setShowCancelConfirm(true)}
                className="px-4 py-2 text-red-600 border border-red-300 rounded-lg hover:bg-red-50"
              >
                Cancel Subscription
              </button>
            )}
          </div>
        </div>
      )}

      {/* Pricing Plans */}
      <div className="grid md:grid-cols-3 gap-8 mb-8">
        {plans.map((plan) => (
          <div
            key={plan.id}
            className={`relative p-6 border-2 rounded-xl transition-all duration-200 ${
              isCurrentPlan(plan.id)
                ? 'border-green-500 bg-green-50'
                : getPlanColor(plan.id)
            } hover:shadow-lg`}
          >
            {isCurrentPlan(plan.id) && (
              <div className="absolute -top-3 left-1/2 transform -translate-x-1/2">
                <span className="bg-green-500 text-white px-3 py-1 rounded-full text-sm font-medium">
                  Current Plan
                </span>
              </div>
            )}

            <div className="text-center mb-6">
              <div className="flex justify-center mb-4">
                {getPlanIcon(plan.id)}
              </div>
              <h3 className="text-2xl font-bold text-gray-900 mb-2">{plan.name}</h3>
              <div className="text-4xl font-bold text-gray-900 mb-2">
                ${plan.price}
                <span className="text-lg font-normal text-gray-600">/month</span>
              </div>
              {plan.id !== 'free' && canStartTrial(plan.id) && (
                <p className="text-sm text-green-600 font-medium">30-day free trial available</p>
              )}
            </div>

            <div className="space-y-4 mb-8">
              <div className="border-t pt-4">
                <h4 className="font-semibold text-gray-900 mb-3">Features:</h4>
                <ul className="space-y-2">
                  {plan.features.map((feature, index) => (
                    <li key={index} className="flex items-center text-sm text-gray-600">
                      <CheckIcon className="w-4 h-4 text-green-500 mr-2 flex-shrink-0" />
                      {feature}
                    </li>
                  ))}
                </ul>
              </div>

              <div className="border-t pt-4">
                <h4 className="font-semibold text-gray-900 mb-3">Limits:</h4>
                <ul className="space-y-2">
                  <li className="flex items-center text-sm text-gray-600">
                    <CheckIcon className="w-4 h-4 text-green-500 mr-2 flex-shrink-0" />
                    {plan.limits.portfolios === -1 ? 'Unlimited' : plan.limits.portfolios} Portfolios
                  </li>
                  <li className="flex items-center text-sm text-gray-600">
                    <CheckIcon className="w-4 h-4 text-green-500 mr-2 flex-shrink-0" />
                    {plan.limits.watchlists === -1 ? 'Unlimited' : plan.limits.watchlists} Watchlists
                  </li>
                </ul>
              </div>

              <div className="border-t pt-4">
                <h4 className="font-semibold text-gray-900 mb-3">Advanced Features:</h4>
                <ul className="space-y-2">
                  {Object.entries(plan.capabilities).map(([key, value]) => (
                    <li key={key} className="flex items-center text-sm text-gray-600">
                      {value ? (
                        <CheckIcon className="w-4 h-4 text-green-500 mr-2 flex-shrink-0" />
                      ) : (
                        <XMarkIcon className="w-4 h-4 text-gray-400 mr-2 flex-shrink-0" />
                      )}
                      {key.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                    </li>
                  ))}
                </ul>
              </div>
            </div>

            <div className="space-y-3">
              {!isCurrentPlan(plan.id) && (
                <>
                  {plan.id !== 'free' && canStartTrial(plan.id) && (
                    <button
                      onClick={() => handleSubscribe(plan.id, true)}
                      disabled={actionLoading}
                      className={`w-full py-3 px-4 rounded-lg font-medium text-white transition-colors ${
                        getButtonColor(plan.id)
                      } disabled:opacity-50 disabled:cursor-not-allowed border-2 border-dashed border-white border-opacity-30`}
                    >
                      {actionLoading ? 'Processing...' : 'Start 30-Day Free Trial'}
                    </button>
                  )}
                  <button
                    onClick={() => handleSubscribe(plan.id, false)}
                    disabled={actionLoading}
                    className={`w-full py-3 px-4 rounded-lg font-medium transition-colors ${
                      plan.id === 'free'
                        ? 'bg-gray-600 hover:bg-gray-700 text-white'
                        : 'border-2 border-gray-300 text-gray-700 hover:bg-gray-50'
                    } disabled:opacity-50 disabled:cursor-not-allowed`}
                  >
                    {actionLoading ? 'Processing...' : plan.id === 'free' ? 'Downgrade to Free' : 'Subscribe Now'}
                  </button>
                </>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Feature Usage & Limits */}
      {subscription && (
        <div className="mt-8">
          <FeatureLimits 
            usage={{
              portfolios: 2, // This would come from actual user data
              watchlistItems: 15, // This would come from actual user data
              alerts: 8, // This would come from actual user data
              apiCalls: 450, // This would come from actual user data
            }}
          />
        </div>
      )}

      {/* Cancel Confirmation Modal */}
      {showCancelConfirm && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white p-6 rounded-lg max-w-md w-full mx-4">
            <h3 className="text-lg font-semibold mb-4">Cancel Subscription</h3>
            <p className="text-gray-600 mb-6">
              Are you sure you want to cancel your subscription? You'll lose access to premium features.
            </p>
            <div className="flex space-x-4">
              <button
                onClick={() => setShowCancelConfirm(false)}
                className="flex-1 py-2 px-4 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50"
              >
                Keep Subscription
              </button>
              <button
                onClick={handleCancel}
                disabled={actionLoading}
                className="flex-1 py-2 px-4 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50"
              >
                {actionLoading ? 'Canceling...' : 'Cancel Subscription'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Subscription;