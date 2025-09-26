"use client";

import { useState } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import { ProtectedRoute } from '@/components/ProtectedRoute';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { 
  CreditCardIcon,
  CheckIcon,
  XMarkIcon,
  StarIcon,
  BoltIcon,
  ShieldCheckIcon
} from '@heroicons/react/24/outline';

const SubscriptionPage = () => {
  const { user } = useAuth();
  const [selectedPlan, setSelectedPlan] = useState(user?.subscriptionPlan || 'free');

  const plans = [
    {
      id: 'free',
      name: 'Free',
      price: 0,
      period: 'month',
      description: 'Perfect for getting started with basic trading features',
      features: [
        'Basic portfolio tracking',
        'Real-time market data',
        'Basic charting tools',
        'Email support',
        'Mobile app access'
      ],
      limitations: [
        'Limited to 5 watchlist items',
        'Basic indicators only',
        'No AI insights',
        'No advanced analytics'
      ],
      popular: false,
      color: 'gray'
    },
    {
      id: 'basic',
      name: 'Basic',
      price: 29,
      period: 'month',
      description: 'Enhanced features for serious traders',
      features: [
        'Everything in Free',
        'Advanced charting tools',
        'Technical indicators',
        'Social trading features',
        'Priority email support',
        'Extended market data',
        'Portfolio analytics'
      ],
      limitations: [
        'Limited AI insights',
        'No institutional tools',
        'Standard data refresh rates'
      ],
      popular: true,
      color: 'blue'
    },
    {
      id: 'premium',
      name: 'Premium',
      price: 99,
      period: 'month',
      description: 'Professional-grade tools and AI-powered insights',
      features: [
        'Everything in Basic',
        'AI-powered insights',
        'Advanced ML models',
        'Institutional tools',
        'Real-time data feeds',
        'Custom indicators',
        'API access',
        'Phone support',
        'Advanced risk management'
      ],
      limitations: [],
      popular: false,
      color: 'purple'
    }
  ];

  const billingHistory = [
    {
      id: '1',
      date: '2024-01-15',
      amount: 0,
      plan: 'Free',
      status: 'active'
    },
    {
      id: '2',
      date: '2023-12-15',
      amount: 29,
      plan: 'Basic',
      status: 'paid'
    },
    {
      id: '3',
      date: '2023-11-15',
      amount: 29,
      plan: 'Basic',
      status: 'paid'
    }
  ];

  const handlePlanSelect = (planId: string) => {
    setSelectedPlan(planId as 'free' | 'basic' | 'premium');
  };

  const handleUpgrade = (planId: string) => {
    console.log('Upgrading to:', planId);
    // Handle plan upgrade
  };

  const getPlanColor = (color: string) => {
    switch (color) {
      case 'blue':
        return 'border-blue-500 bg-blue-500/10';
      case 'purple':
        return 'border-purple-500 bg-purple-500/10';
      default:
        return 'border-gray-600 bg-gray-800';
    }
  };

  const getButtonColor = (color: string) => {
    switch (color) {
      case 'blue':
        return 'bg-blue-600 hover:bg-blue-700';
      case 'purple':
        return 'bg-purple-600 hover:bg-purple-700';
      default:
        return 'bg-gray-600 hover:bg-gray-700';
    }
  };

  return (
    <ProtectedRoute requiredPlan="free">
      <div className="min-h-screen bg-gray-900 p-6">
        <div className="max-w-7xl mx-auto">
          {/* Header */}
          <div className="mb-8 text-center">
            <h1 className="text-4xl font-bold text-white mb-4 flex items-center justify-center">
              <CreditCardIcon className="w-10 h-10 mr-3" />
              Subscription Plans
            </h1>
            <p className="text-xl text-gray-400 max-w-2xl mx-auto">
              Choose the perfect plan for your trading needs. Upgrade or downgrade anytime.
            </p>
          </div>

          {/* Current Plan Status */}
          <Card className="bg-gray-800 border-gray-700 mb-8">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-xl font-semibold text-white mb-2">Current Plan</h2>
                  <div className="flex items-center space-x-3">
                    <Badge className="bg-blue-600 text-white text-lg px-3 py-1">
                      {user?.subscriptionPlan || 'Free'}
                    </Badge>
                    <span className="text-gray-400">•</span>
                    <span className="text-gray-300">Active since January 15, 2024</span>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-2xl font-bold text-white">
                    ${plans.find(p => p.id === (user?.subscriptionPlan || 'free'))?.price || 0}
                  </div>
                  <div className="text-gray-400">per month</div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Pricing Plans */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-12">
            {plans.map((plan) => (
              <Card
                key={plan.id}
                className={`relative ${getPlanColor(plan.color)} ${
                  plan.popular ? 'ring-2 ring-blue-500' : ''
                }`}
              >
                {plan.popular && (
                  <div className="absolute -top-4 left-1/2 transform -translate-x-1/2">
                    <Badge className="bg-blue-600 text-white px-4 py-1 flex items-center">
                      <StarIcon className="w-4 h-4 mr-1" />
                      Most Popular
                    </Badge>
                  </div>
                )}

                <CardHeader className="text-center pb-4">
                  <CardTitle className="text-2xl font-bold text-white mb-2">
                    {plan.name}
                  </CardTitle>
                  <div className="mb-4">
                    <span className="text-4xl font-bold text-white">${plan.price}</span>
                    <span className="text-gray-400">/{plan.period}</span>
                  </div>
                  <p className="text-gray-300">{plan.description}</p>
                </CardHeader>

                <CardContent className="space-y-6">
                  {/* Features */}
                  <div>
                    <h4 className="font-semibold text-white mb-3 flex items-center">
                      <CheckIcon className="w-4 h-4 mr-2 text-green-400" />
                      Included Features
                    </h4>
                    <ul className="space-y-2">
                      {plan.features.map((feature, index) => (
                        <li key={index} className="flex items-center text-gray-300">
                          <CheckIcon className="w-4 h-4 mr-2 text-green-400 flex-shrink-0" />
                          {feature}
                        </li>
                      ))}
                    </ul>
                  </div>

                  {/* Limitations */}
                  {plan.limitations.length > 0 && (
                    <div>
                      <h4 className="font-semibold text-white mb-3 flex items-center">
                        <XMarkIcon className="w-4 h-4 mr-2 text-red-400" />
                        Limitations
                      </h4>
                      <ul className="space-y-2">
                        {plan.limitations.map((limitation, index) => (
                          <li key={index} className="flex items-center text-gray-400">
                            <XMarkIcon className="w-4 h-4 mr-2 text-red-400 flex-shrink-0" />
                            {limitation}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Action Button */}
                  <div className="pt-4">
                    {plan.id === (user?.subscriptionPlan || 'free') ? (
                      <Button disabled className="w-full bg-gray-600 text-gray-300">
                        Current Plan
                      </Button>
                    ) : (
                      <Button
                        onClick={() => handleUpgrade(plan.id)}
                        className={`w-full ${getButtonColor(plan.color)}`}
                      >
                        {plan.price === 0 ? 'Downgrade' : 'Upgrade'} to {plan.name}
                      </Button>
                    )}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {/* Features Comparison */}
          <Card className="bg-gray-800 border-gray-700 mb-8">
            <CardHeader>
              <CardTitle className="text-white text-xl">Feature Comparison</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-gray-700">
                      <th className="text-left py-3 px-4 text-gray-300 font-medium">Feature</th>
                      <th className="text-center py-3 px-4 text-gray-300 font-medium">Free</th>
                      <th className="text-center py-3 px-4 text-gray-300 font-medium">Basic</th>
                      <th className="text-center py-3 px-4 text-gray-300 font-medium">Premium</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      { feature: 'Portfolio Tracking', free: true, basic: true, premium: true },
                      { feature: 'Real-time Data', free: true, basic: true, premium: true },
                      { feature: 'Advanced Charts', free: false, basic: true, premium: true },
                      { feature: 'AI Insights', free: false, basic: 'Limited', premium: true },
                      { feature: 'Social Trading', free: false, basic: true, premium: true },
                      { feature: 'API Access', free: false, basic: false, premium: true },
                      { feature: 'Phone Support', free: false, basic: false, premium: true }
                    ].map((row, index) => (
                      <tr key={index} className="border-b border-gray-700/50">
                        <td className="py-3 px-4 text-white">{row.feature}</td>
                        <td className="py-3 px-4 text-center">
                          {row.free === true ? (
                            <CheckIcon className="w-5 h-5 text-green-400 mx-auto" />
                          ) : row.free === false ? (
                            <XMarkIcon className="w-5 h-5 text-red-400 mx-auto" />
                          ) : (
                            <span className="text-yellow-400">{row.free}</span>
                          )}
                        </td>
                        <td className="py-3 px-4 text-center">
                          {row.basic === true ? (
                            <CheckIcon className="w-5 h-5 text-green-400 mx-auto" />
                          ) : row.basic === false ? (
                            <XMarkIcon className="w-5 h-5 text-red-400 mx-auto" />
                          ) : (
                            <span className="text-yellow-400">{row.basic}</span>
                          )}
                        </td>
                        <td className="py-3 px-4 text-center">
                          {row.premium === true ? (
                            <CheckIcon className="w-5 h-5 text-green-400 mx-auto" />
                          ) : row.premium === false ? (
                            <XMarkIcon className="w-5 h-5 text-red-400 mx-auto" />
                          ) : (
                            <span className="text-yellow-400">{row.premium}</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>

          {/* Billing History */}
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader>
              <CardTitle className="text-white">Billing History</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {billingHistory.map((bill) => (
                  <div key={bill.id} className="flex items-center justify-between p-3 bg-gray-700/50 rounded-lg">
                    <div>
                      <div className="font-medium text-white">{bill.plan} Plan</div>
                      <div className="text-sm text-gray-400">{bill.date}</div>
                    </div>
                    <div className="flex items-center space-x-3">
                      <div className="text-white font-medium">
                        ${bill.amount.toFixed(2)}
                      </div>
                      <Badge className={bill.status === 'paid' ? 'bg-green-600' : 'bg-blue-600'}>
                        {bill.status}
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </ProtectedRoute>
  );
};

export default SubscriptionPage;