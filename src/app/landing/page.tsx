"use client";

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { 
  ChartBarIcon, 
  BoltIcon, 
  ShieldCheckIcon, 
  GlobeAltIcon,
  ArrowRightIcon,
  CheckIcon
} from '@heroicons/react/24/outline';

export default function LandingPage() {
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(false);

  const handleGetStarted = () => {
    setIsLoading(true);
    router.push('/auth/register');
  };

  const handleSignIn = () => {
    setIsLoading(true);
    router.push('/auth/login');
  };

  const features = [
    {
      icon: ChartBarIcon,
      title: "Real-Time Market Data",
      description: "Live market feeds with advanced charting and technical indicators"
    },
    {
      icon: BoltIcon,
      title: "AI-Powered Insights",
      description: "Machine learning algorithms for market analysis and predictions"
    },
    {
      icon: ShieldCheckIcon,
      title: "Secure Trading",
      description: "Bank-level security with encrypted transactions and data protection"
    },
    {
      icon: GlobeAltIcon,
      title: "Global Markets",
      description: "Access to stocks, crypto, forex, and commodities worldwide"
    }
  ];

  const pricingPlans = [
    {
      name: "Basic",
      price: "Free",
      features: ["Real-time quotes", "Basic charts", "Portfolio tracking", "Community access"]
    },
    {
      name: "Pro",
      price: "$29/mo",
      features: ["Advanced analytics", "AI insights", "Paper trading", "Priority support"],
      popular: true
    },
    {
      name: "Enterprise",
      price: "$99/mo",
      features: ["Institutional tools", "API access", "Custom indicators", "Dedicated support"]
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900">
      {/* Header */}
      <header className="relative z-10 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <ChartBarIcon className="h-8 w-8 text-blue-400" />
            <span className="text-2xl font-bold text-white">FinScope</span>
          </div>
          <div className="flex items-center space-x-4">
            <Button 
              variant="ghost" 
              onClick={() => router.push('/support')}
              disabled={isLoading}
              className="text-white hover:text-blue-400"
            >
              Support
            </Button>
            <Button 
              variant="ghost" 
              onClick={handleSignIn}
              disabled={isLoading}
              className="text-white hover:text-blue-400"
            >
              Sign In
            </Button>
            <Button 
              onClick={handleGetStarted}
              disabled={isLoading}
              className="bg-blue-600 hover:bg-blue-700 text-white"
            >
              Get Started
              <ArrowRightIcon className="ml-2 h-4 w-4" />
            </Button>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="relative px-6 py-20">
        <div className="max-w-7xl mx-auto text-center">
          <Badge className="mb-6 bg-blue-600/20 text-blue-400 border-blue-600/30">
            Advanced Trading Platform
          </Badge>
          <h1 className="text-5xl md:text-7xl font-bold text-white mb-6">
            Trade Smarter with
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-400">
              {" "}AI-Powered{" "}
            </span>
            Analytics
          </h1>
          <p className="text-xl text-gray-300 mb-8 max-w-3xl mx-auto">
            Professional-grade financial analytics, real-time market data, and AI-driven insights 
            to help you make informed trading decisions.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button 
              size="lg" 
              onClick={handleGetStarted}
              disabled={isLoading}
              className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-3"
            >
              Start Trading Now
              <ArrowRightIcon className="ml-2 h-5 w-5" />
            </Button>
            <Button 
              size="lg" 
              variant="outline" 
              onClick={handleSignIn}
              disabled={isLoading}
              className="border-gray-600 text-white hover:bg-gray-800 px-8 py-3"
            >
              View Demo
            </Button>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="px-6 py-20 bg-black/20">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-white mb-4">
              Everything You Need to Trade
            </h2>
            <p className="text-xl text-gray-300">
              Comprehensive tools and features for modern traders
            </p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, index) => (
              <Card key={index} className="bg-gray-800/50 border-gray-700 hover:bg-gray-800/70 transition-colors">
                <CardHeader>
                  <feature.icon className="h-12 w-12 text-blue-400 mb-4" />
                  <CardTitle className="text-white">{feature.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-gray-300">{feature.description}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Pricing Section */}
      <section className="px-6 py-20">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-white mb-4">
              Choose Your Plan
            </h2>
            <p className="text-xl text-gray-300">
              Start free, upgrade when you need more
            </p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {pricingPlans.map((plan, index) => (
              <Card 
                key={index} 
                className={`relative ${
                  plan.popular 
                    ? 'bg-gradient-to-b from-blue-900/50 to-purple-900/50 border-blue-500' 
                    : 'bg-gray-800/50 border-gray-700'
                }`}
              >
                {plan.popular && (
                  <Badge className="absolute -top-3 left-1/2 transform -translate-x-1/2 bg-blue-600 text-white">
                    Most Popular
                  </Badge>
                )}
                <CardHeader className="text-center">
                  <CardTitle className="text-white text-2xl">{plan.name}</CardTitle>
                  <div className="text-3xl font-bold text-white mt-4">
                    {plan.price}
                  </div>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-3">
                    {plan.features.map((feature, featureIndex) => (
                      <li key={featureIndex} className="flex items-center text-gray-300">
                        <CheckIcon className="h-5 w-5 text-green-400 mr-3" />
                        {feature}
                      </li>
                    ))}
                  </ul>
                  <Button 
                    className={`w-full mt-6 ${
                      plan.popular 
                        ? 'bg-blue-600 hover:bg-blue-700' 
                        : 'bg-gray-700 hover:bg-gray-600'
                    } text-white`}
                    onClick={handleGetStarted}
                    disabled={isLoading}
                  >
                    Get Started
                  </Button>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="px-6 py-12 bg-black/40">
        <div className="max-w-7xl mx-auto text-center">
          <div className="flex items-center justify-center space-x-2 mb-4">
            <ChartBarIcon className="h-6 w-6 text-blue-400" />
            <span className="text-xl font-bold text-white">FinScope</span>
          </div>
          <p className="text-gray-400">
            © 2024 FinScope. All rights reserved. Professional trading platform.
          </p>
        </div>
      </footer>
    </div>
  );
}