'use client';

import React from 'react';
import {
  MagnifyingGlassIcon,
  BellIcon,
  SunIcon,
  MoonIcon,
  Cog6ToothIcon,
} from '@heroicons/react/24/outline';
import { useStore } from '../store/useStore';
import { useDarkMode } from '../hooks';

const Header: React.FC = () => {
  const {
    activeView,
    selectedAssets,
    selectedModels,
    searchQuery,
    setSearchQuery,
    notifications
  } = useStore();
  
  const { isDark, toggle } = useDarkMode();

  const getViewTitle = () => {
    switch (activeView) {
      case 'dashboard':
        return 'Trading Dashboard';
      case 'portfolio':
        return 'Portfolio Manager';
      case 'trading':
        return 'Trading Interface';
      case 'charting':
        return 'Advanced Charts';
      case 'risk':
        return 'Risk Management';
      case 'analytics':
        return 'Analytics Dashboard';
      case 'news':
        return 'News & Analysis';
      case 'research':
        return 'Asset Research';
      case 'forum':
        return 'Community Forum';
      case 'notifications':
        return 'Notification Center';
      default:
        return 'FinScope';
    }
  };

  const marketStatus = {
    isOpen: true,
    nextClose: '4:00 PM EST',
    timezone: 'Eastern Time'
  };

  const unreadNotifications = notifications?.filter(n => !n.read)?.length || 0;

  return (
    <header className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-6 py-4">
      <div className="flex items-center justify-between">
        {/* Left Section */}
        <div className="flex items-center space-x-6">
          {/* View Title */}
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
              {getViewTitle()}
            </h1>
            <div className="flex items-center space-x-4 mt-1">
              {/* Selected Assets Indicator */}
              {selectedAssets.length > 0 && (
                <div className="flex items-center space-x-2">
                  <span className="text-sm text-gray-500 dark:text-gray-400">Assets:</span>
                  <div className="flex space-x-1">
                    {selectedAssets.slice(0, 3).map((asset, index) => (
                      <span
                        key={index}
                        className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-300"
                      >
                        {asset.symbol}
                      </span>
                    ))}
                    {selectedAssets.length > 3 && (
                      <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300">
                        +{selectedAssets.length - 3}
                      </span>
                    )}
                  </div>
                </div>
              )}
              
              {/* Selected Models Indicator */}
              {selectedModels.length > 0 && (
                <div className="flex items-center space-x-2">
                  <span className="text-sm text-gray-500 dark:text-gray-400">Models:</span>
                  <div className="flex space-x-1">
                    {selectedModels.slice(0, 2).map((model, index) => (
                      <span
                        key={index}
                        className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-300"
                      >
                        {model.name}
                      </span>
                    ))}
                    {selectedModels.length > 2 && (
                      <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300">
                        +{selectedModels.length - 2}
                      </span>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Center Section - Search */}
        <div className="flex-1 max-w-lg mx-8">
          <div className="relative">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <MagnifyingGlassIcon className="h-5 w-5 text-gray-400" />
            </div>
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="block w-full pl-10 pr-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg leading-5 bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              placeholder="Search assets, news, or community posts..."
            />
          </div>
        </div>

        {/* Right Section */}
        <div className="flex items-center space-x-4">
          {/* Market Status */}
          <div className="hidden md:flex items-center space-x-2">
            <div className={`w-2 h-2 rounded-full ${
              marketStatus.isOpen ? 'bg-green-400' : 'bg-red-400'
            }`}></div>
            <span className="text-sm text-gray-600 dark:text-gray-300">
              Market {marketStatus.isOpen ? 'Open' : 'Closed'}
            </span>
            <span className="text-xs text-gray-500 dark:text-gray-400">
              {marketStatus.nextClose}
            </span>
          </div>

          {/* Notifications */}
          <button className="relative p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors">
            <BellIcon className="h-6 w-6" />
            {unreadNotifications > 0 && (
              <span className="absolute top-0 right-0 block h-2 w-2 rounded-full bg-red-400 ring-2 ring-white dark:ring-gray-800"></span>
            )}
          </button>

          {/* Theme Toggle */}
          <button
            onClick={toggle}
            className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
          >
            {isDark ? (
              <SunIcon className="h-6 w-6" />
            ) : (
              <MoonIcon className="h-6 w-6" />
            )}
          </button>

          {/* Settings */}
          <button className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors">
            <Cog6ToothIcon className="h-6 w-6" />
          </button>
        </div>
      </div>
    </header>
  );
};

export default Header;