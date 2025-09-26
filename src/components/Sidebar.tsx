'use client';

import React from 'react';
import {
  ChartBarIcon,
  NewspaperIcon,
  MagnifyingGlassIcon,
  ChatBubbleLeftRightIcon,
  CogIcon,
  UserIcon,
  BellIcon,
  HeartIcon,
  BookmarkIcon,
  ArrowTrendingUpIcon,
  BriefcaseIcon,
  ShieldCheckIcon,
  CurrencyDollarIcon,
  PresentationChartLineIcon,
  CreditCardIcon,
} from '@heroicons/react/24/outline';
import { useStore } from '../store/useStore';

const Sidebar: React.FC = () => {
  const { activeView, setActiveView, watchlist, notifications, user } = useStore();
  
  const menuItems = [
    {
      id: 'dashboard',
      label: 'Dashboard',
      icon: ChartBarIcon,
      description: 'Trading overview'
    },
    {
      id: 'portfolio',
      label: 'Portfolio',
      icon: BriefcaseIcon,
      description: 'Manage portfolios'
    },
    {
      id: 'trading',
      label: 'Trading',
      icon: CurrencyDollarIcon,
      description: 'Execute trades'
    },
    {
      id: 'charting',
      label: 'Advanced Charts',
      icon: PresentationChartLineIcon,
      description: 'Technical analysis'
    },
    {
      id: 'risk',
      label: 'Risk Management',
      icon: ShieldCheckIcon,
      description: 'Risk assessment'
    },
    {
      id: 'news',
      label: 'News & Analysis',
      icon: NewspaperIcon,
      description: 'Market insights'
    },
    {
      id: 'research',
      label: 'Asset Research',
      icon: MagnifyingGlassIcon,
      description: 'Deep analysis'
    },
    {
      id: 'forum',
      label: 'Community',
      icon: ChatBubbleLeftRightIcon,
      description: 'Discussion forum'
    },
    {
      id: 'notifications',
      label: 'Notifications',
      icon: BellIcon,
      description: 'Alerts & updates'
    },
    {
      id: 'subscription',
      label: 'Subscription',
      icon: CreditCardIcon,
      description: 'Manage subscription'
    }
  ];

  const quickActions = [
    { icon: HeartIcon, label: 'Watchlist', count: watchlist?.length || 0 },
    { icon: BookmarkIcon, label: 'Saved', count: 8 },
    { icon: ArrowTrendingUpIcon, label: 'Trending', count: 24 },
    { icon: BellIcon, label: 'Alerts', count: notifications?.filter(n => !n.read)?.length || 0 }
  ];

  return (
    <div className="w-64 bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 flex flex-col">
      {/* Logo */}
      <div className="p-6 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 bg-gradient-to-br from-primary-500 to-primary-600 rounded-lg flex items-center justify-center">
            <span className="text-white font-bold text-sm">FS</span>
          </div>
          <div>
            <h1 className="text-xl font-bold text-gray-900 dark:text-white">FinScope</h1>
            <p className="text-xs text-gray-500 dark:text-gray-400">AI Trading Platform</p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 space-y-2">
        <div className="mb-6">
          <h2 className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-3">
            Navigation
          </h2>
          {menuItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeView === item.id;
            
            return (
              <button
                key={item.id}
                onClick={() => setActiveView(item.id as any)}
                className={`w-full flex items-center space-x-3 px-3 py-2.5 rounded-lg text-left transition-all duration-200 group ${
                  isActive
                    ? 'bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300 border border-primary-200 dark:border-primary-800'
                    : 'text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700/50'
                }`}
              >
                <Icon className={`w-5 h-5 ${isActive ? 'text-primary-600 dark:text-primary-400' : 'text-gray-500 dark:text-gray-400 group-hover:text-gray-700 dark:group-hover:text-gray-300'}`} />
                <div className="flex-1">
                  <div className={`text-sm font-medium ${isActive ? 'text-primary-700 dark:text-primary-300' : 'text-gray-900 dark:text-white'}`}>
                    {item.label}
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    {item.description}
                  </div>
                </div>
              </button>
            );
          })}
        </div>

        {/* Quick Actions */}
        <div className="mb-6">
          <h2 className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-3">
            Quick Access
          </h2>
          <div className="space-y-1">
            {quickActions.map((action, index) => {
              const Icon = action.icon;
              return (
                <button
                  key={index}
                  className="w-full flex items-center justify-between px-3 py-2 rounded-lg text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors"
                >
                  <div className="flex items-center space-x-3">
                    <Icon className="w-4 h-4 text-gray-500 dark:text-gray-400" />
                    <span className="text-sm">{action.label}</span>
                  </div>
                  {action.count > 0 && (
                    <span className="text-xs bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 px-2 py-1 rounded-full">
                      {action.count}
                    </span>
                  )}
                </button>
              );
            })}
          </div>
        </div>
      </nav>

      {/* User Profile */}
      <div className="p-4 border-t border-gray-200 dark:border-gray-700">
        <div className="flex items-center space-x-3 p-3 rounded-lg bg-gray-50 dark:bg-gray-700/50">
          <div className="w-8 h-8 bg-gradient-to-br from-gray-400 to-gray-500 rounded-full flex items-center justify-center">
            {user?.avatar ? (
              <img src={user.avatar} alt={user.username} className="w-8 h-8 rounded-full" />
            ) : (
              <UserIcon className="w-4 h-4 text-white" />
            )}
          </div>
          <div className="flex-1">
            <div className="text-sm font-medium text-gray-900 dark:text-white">
              {user?.username || 'Guest User'}
            </div>
            <div className="text-xs text-gray-500 dark:text-gray-400">
              {user?.subscription || 'Free Plan'}
            </div>
          </div>
          <button className="p-1 rounded-md hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors">
            <CogIcon className="w-4 h-4 text-gray-500 dark:text-gray-400" />
          </button>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;