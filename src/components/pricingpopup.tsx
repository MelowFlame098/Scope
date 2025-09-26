"use client";

import React, { useState } from 'react';
import { X, Check, Star, Zap, Shield } from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext';

interface PricingPopupProps {
  isOpen: boolean;
  onClose: () => void;
  feature?: string;
  requiredPlan?: 'basic' | 'premium';
  title?: string;
  description?: string;
}

interface PricingPlan {
  id: string;
  name: string;
  price: number;
  period: string;
  description: string;
  features: string[];
  limits: {
    portfolios: number | string;
    watchlists: number | string;
  };
  popular?: boolean;
  icon: React.ReactNode;
  buttonText: string;
  buttonVariant: 'outline' | 'primary' | 'premium';
}

const pricingPlans: PricingPlan[] = [
  {
    id: 'basic',
    name: 'Basic',
    price: 4.99,
    period: 'month',
    description: 'Perfect for individual traders',
    features: [
      'Real-time market data',
      'Advanced charting tools',
      'Price alerts & notifications',
      'Technical indicators',
      'Social trading features',
      'Email support'
    ],
    limits: {
      portfolios: 5,
      watchlists: 10
    },
    icon: <Zap className="w-6 h-6" />,
    buttonText: 'Start Basic Plan',
    buttonVariant: 'primary'
  },
  {
    id: 'premium',
    name: 'Premium',
    price: 9.99,
    period: 'month',
    description: 'For serious traders and investors',
    features: [
      'Everything in Basic',
      'AI-powered insights',
      'Unlimited portfolios',
      'Unlimited watchlists',
      'Institutional-grade tools',
      'Priority support',
      'Advanced analytics',
      'Custom indicators'
    ],
    limits: {
      portfolios: 'Unlimited',
      watchlists: 'Unlimited'
    },
    popular: true,
    icon: <Star className="w-6 h-6" />,
    buttonText: 'Start Premium Plan',
    buttonVariant: 'premium'
  }
];

export const PricingPopup: React.FC<PricingPopupProps> = ({
  isOpen,
  onClose,
  feature,
  requiredPlan,
  title,
  description
}) => {
  const { user } = useAuth();
  const [isLoading, setIsLoading] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleSubscribe = async (planId: string) => {
    setIsLoading(planId);
    
    try {
      // Here you would integrate with your payment processor
      // For now, we'll simulate the subscription process
      const response = await fetch('/api/subscription/subscribe', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('access_token')}`,
        },
        body: JSON.stringify({
          plan: planId,
          feature: feature
        }),
      });

      if (response.ok) {
        // Redirect to payment or success page
        window.location.href = `/subscription/checkout?plan=${planId}`;
      } else {
        console.error('Subscription failed');
      }
    } catch (error) {
      console.error('Subscription error:', error);
    } finally {
      setIsLoading(null);
    }
  };

  const getFeatureTitle = () => {
    if (title) return title;
    
    switch (feature) {
      case '/dashboard/analytics':
        return 'Advanced Analytics';
      case '/dashboard/ai-insights':
        return 'AI-Powered Insights';
      case '/dashboard/social-trading':
        return 'Social Trading';
      case '/dashboard/institutional':
        return 'Institutional Tools';
      default:
        return 'Premium Feature';
    }
  };

  const getFeatureDescription = () => {
    if (description) return description;
    
    switch (feature) {
      case '/dashboard/analytics':
        return 'Access advanced analytics and detailed market insights to make better trading decisions.';
      case '/dashboard/ai-insights':
        return 'Get AI-powered market analysis and personalized trading recommendations.';
      case '/dashboard/social-trading':
        return 'Follow top traders, copy their strategies, and share your own insights.';
      case '/dashboard/institutional':
        return 'Access professional-grade tools used by institutional traders and fund managers.';
      default:
        return 'This feature requires a subscription to access.';
    }
  };

  const filteredPlans = requiredPlan 
    ? pricingPlans.filter(plan => 
        requiredPlan === 'basic' ? true : plan.id === 'premium'
      )
    : pricingPlans;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl max-w-4xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="sticky top-0 bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between rounded-t-2xl">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">{getFeatureTitle()}</h2>
            <p className="text-gray-600 mt-1">{getFeatureDescription()}</p>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 rounded-full transition-colors"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* Current Plan Info */}
        {user && (
          <div className="px-6 py-4 bg-blue-50 border-b border-gray-200">
            <div className="flex items-center gap-3">
              <Shield className="w-5 h-5 text-blue-600" />
              <div>
                <p className="text-sm font-medium text-blue-900">
                  Current Plan: {user.subscriptionPlan?.charAt(0).toUpperCase() + user.subscriptionPlan?.slice(1) || 'Free'}
                </p>
                <p className="text-xs text-blue-700">
                  Upgrade to access this feature and unlock more capabilities
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Pricing Plans */}
        <div className="px-6 py-8">
          <div className="grid md:grid-cols-2 gap-6">
            {filteredPlans.map((plan) => (
              <div
                key={plan.id}
                className={`relative rounded-xl border-2 p-6 ${
                  plan.popular
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-200 bg-white'
                }`}
              >
                {plan.popular && (
                  <div className="absolute -top-3 left-1/2 transform -translate-x-1/2">
                    <span className="bg-blue-500 text-white px-3 py-1 rounded-full text-sm font-medium">
                      Most Popular
                    </span>
                  </div>
                )}

                <div className="text-center mb-6">
                  <div className="flex items-center justify-center mb-3">
                    <div className={`p-3 rounded-full ${
                      plan.popular ? 'bg-blue-100 text-blue-600' : 'bg-gray-100 text-gray-600'
                    }`}>
                      {plan.icon}
                    </div>
                  </div>
                  <h3 className="text-xl font-bold text-gray-900">{plan.name}</h3>
                  <p className="text-gray-600 text-sm mt-1">{plan.description}</p>
                  <div className="mt-4">
                    <span className="text-3xl font-bold text-gray-900">${plan.price}</span>
                    <span className="text-gray-600">/{plan.period}</span>
                  </div>
                </div>

                <div className="space-y-3 mb-6">
                  <div className="text-sm">
                    <div className="flex justify-between items-center py-1">
                      <span className="text-gray-600">Portfolios:</span>
                      <span className="font-medium">{plan.limits.portfolios}</span>
                    </div>
                    <div className="flex justify-between items-center py-1">
                      <span className="text-gray-600">Watchlists:</span>
                      <span className="font-medium">{plan.limits.watchlists}</span>
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    {plan.features.map((feature, index) => (
                      <div key={index} className="flex items-center gap-2">
                        <Check className="w-4 h-4 text-green-500 flex-shrink-0" />
                        <span className="text-sm text-gray-700">{feature}</span>
                      </div>
                    ))}
                  </div>
                </div>

                <button
                  onClick={() => handleSubscribe(plan.id)}
                  disabled={isLoading === plan.id}
                  className={`w-full py-3 px-4 rounded-lg font-medium transition-colors ${
                    plan.buttonVariant === 'premium'
                      ? 'bg-gradient-to-r from-purple-600 to-blue-600 text-white hover:from-purple-700 hover:to-blue-700'
                      : plan.buttonVariant === 'primary'
                      ? 'bg-blue-600 text-white hover:bg-blue-700'
                      : 'border border-gray-300 text-gray-700 hover:bg-gray-50'
                  } disabled:opacity-50 disabled:cursor-not-allowed`}
                >
                  {isLoading === plan.id ? 'Processing...' : plan.buttonText}
                </button>
              </div>
            ))}
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-4 bg-gray-50 border-t border-gray-200 rounded-b-2xl">
          <p className="text-center text-sm text-gray-600">
            All plans include a 7-day free trial. Cancel anytime.
          </p>
        </div>
      </div>
    </div>
  );
};

export default PricingPopup;