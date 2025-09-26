'use client';

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  ChartBarIcon,
  PresentationChartLineIcon,
  UserGroupIcon,
  BuildingOfficeIcon,
  RocketLaunchIcon,
  BriefcaseIcon,
  CurrencyDollarIcon,
  BellIcon,
  CogIcon,
  CreditCardIcon,
} from '@heroicons/react/24/outline';
import { useAuth } from '@/contexts/AuthContext';
import { FeatureAccess } from '@/components/ProtectedRoute';

interface SidebarProps {
  user: any;
  onLogout: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ user, onLogout }) => {
  const pathname = usePathname();
  const { hasFeatureAccess } = useAuth();

  const menuItems = [
    {
      id: 'dashboard',
      label: 'Dashboard',
      icon: ChartBarIcon,
      href: '/dashboard',
      description: 'Main overview',
      requiredPlan: 'free' as const
    },
    {
      id: 'portfolio',
      label: 'Portfolio',
      icon: BriefcaseIcon,
      href: '/dashboard/portfolio',
      description: 'Manage portfolios',
      requiredPlan: 'free' as const
    },
    {
      id: 'trading',
      label: 'Trading',
      icon: CurrencyDollarIcon,
      href: '/dashboard/trading',
      description: 'Execute trades',
      requiredPlan: 'free' as const
    },
    {
      id: 'analytics',
      label: 'Analytics',
      icon: PresentationChartLineIcon,
      href: '/dashboard/analytics',
      description: 'Advanced analysis',
      requiredPlan: 'basic' as const,
      feature: 'advancedAnalytics'
    },
    {
      id: 'ai-insights',
      label: 'AI Insights',
      icon: RocketLaunchIcon,
      href: '/dashboard/ai-insights',
      description: 'AI-powered analysis',
      requiredPlan: 'premium' as const,
      feature: 'aiInsights'
    },
    {
      id: 'social-trading',
      label: 'Social Trading',
      icon: UserGroupIcon,
      href: '/dashboard/social-trading',
      description: 'Community trading',
      requiredPlan: 'basic' as const,
      feature: 'socialTrading'
    },
    {
      id: 'institutional',
      label: 'Institutional',
      icon: BuildingOfficeIcon,
      href: '/dashboard/institutional',
      description: 'Professional tools',
      requiredPlan: 'premium' as const,
      feature: 'institutionalTools'
    },
    {
      id: 'notifications',
      label: 'Notifications',
      icon: BellIcon,
      href: '/dashboard/notifications',
      description: 'Alerts & updates',
      requiredPlan: 'free' as const
    },
    {
      id: 'settings',
      label: 'Settings',
      icon: CogIcon,
      href: '/dashboard/settings',
      description: 'Account settings',
      requiredPlan: 'free' as const
    },
    {
      id: 'subscription',
      label: 'Subscription',
      icon: CreditCardIcon,
      href: '/dashboard/subscription',
      description: 'Manage plan',
      requiredPlan: 'free' as const
    }
  ];

  const isActive = (href: string) => {
    if (href === '/dashboard') {
      return pathname === '/dashboard';
    }
    return pathname.startsWith(href);
  };

  const canAccessFeature = (item: any) => {
    if (!item.feature) return true;
    return hasFeatureAccess(item.feature);
  };

  return (
    <div className="w-64 bg-gray-800 border-r border-gray-700 flex flex-col h-full">
      {/* Logo */}
      <div className="p-6 border-b border-gray-700">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg flex items-center justify-center">
            <span className="text-white font-bold text-sm">FS</span>
          </div>
          <div>
            <h1 className="text-xl font-bold text-white">FinScope</h1>
            <p className="text-xs text-gray-400">AI Trading Platform</p>
          </div>
        </div>
      </div>

      {/* User Info */}
      {user && (
        <div className="p-4 border-b border-gray-700">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gray-600 rounded-full flex items-center justify-center">
              <span className="text-white font-medium text-sm">
                {user.firstName?.[0]}{user.lastName?.[0]}
              </span>
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-white truncate">
                {user.firstName} {user.lastName}
              </p>
              <p className="text-xs text-gray-400 truncate">
                {user.subscriptionPlan || 'Free'} Plan
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Navigation */}
      <nav className="flex-1 p-4 space-y-2 overflow-y-auto">
        <div className="mb-6">
          <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
            Navigation
          </h2>
          {menuItems.map((item) => {
            const Icon = item.icon;
            const active = isActive(item.href);
            const hasAccess = canAccessFeature(item);
            
            const linkContent = (
              <div className={`w-full flex items-center space-x-3 px-3 py-2.5 rounded-lg text-left transition-all duration-200 group ${
                active
                  ? 'bg-blue-600 text-white'
                  : hasAccess
                  ? 'text-gray-300 hover:bg-gray-700 hover:text-white'
                  : 'text-gray-500 cursor-not-allowed'
              }`}>
                <Icon className={`w-5 h-5 ${
                  active ? 'text-white' : hasAccess ? 'text-gray-400 group-hover:text-gray-300' : 'text-gray-600'
                }`} />
                <div className="flex-1">
                  <div className={`text-sm font-medium ${
                    active ? 'text-white' : hasAccess ? 'text-gray-300' : 'text-gray-500'
                  }`}>
                    {item.label}
                    {!hasAccess && (
                      <span className="ml-2 text-xs bg-yellow-600 text-yellow-100 px-2 py-0.5 rounded">
                        {item.requiredPlan}
                      </span>
                    )}
                  </div>
                  <div className="text-xs text-gray-400">
                    {item.description}
                  </div>
                </div>
              </div>
            );

            if (!hasAccess) {
              return (
                <FeatureAccess key={item.id} feature={item.feature || 'basic'} requiredPlan={item.requiredPlan}>
                  {linkContent}
                </FeatureAccess>
              );
            }

            return (
              <Link key={item.id} href={item.href}>
                {linkContent}
              </Link>
            );
          })}
        </div>
      </nav>

      {/* Logout Button */}
      <div className="p-4 border-t border-gray-700">
        <button
          onClick={onLogout}
          className="w-full flex items-center justify-center px-4 py-2 text-sm font-medium text-gray-300 bg-gray-700 rounded-lg hover:bg-gray-600 hover:text-white transition-colors"
        >
          Sign Out
        </button>
      </div>
    </div>
  );
};