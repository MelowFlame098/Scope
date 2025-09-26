'use client';

import React from 'react';
import { usePathname } from 'next/navigation';
import {
  BellIcon,
  MagnifyingGlassIcon,
  Cog6ToothIcon,
  UserCircleIcon,
} from '@heroicons/react/24/outline';
import { useAuth } from '@/contexts/AuthContext';

interface HeaderProps {
  user: any;
  onLogout: () => void;
}

export const Header: React.FC<HeaderProps> = ({ user, onLogout }) => {
  const pathname = usePathname();
  const { user: authUser } = useAuth();

  const getPageTitle = () => {
    const path = pathname.split('/').pop();
    switch (path) {
      case 'dashboard':
        return 'Dashboard';
      case 'analytics':
        return 'Advanced Analytics';
      case 'ai-insights':
        return 'AI Insights';
      case 'social-trading':
        return 'Social Trading';
      case 'institutional':
        return 'Institutional Tools';
      case 'portfolio':
        return 'Portfolio Management';
      case 'trading':
        return 'Trading Interface';
      case 'notifications':
        return 'Notifications';
      case 'settings':
        return 'Settings';
      case 'subscription':
        return 'Subscription';
      default:
        return 'Dashboard';
    }
  };

  const getSubscriptionBadge = () => {
    const plan = authUser?.subscriptionPlan || 'free';
    const colors = {
      free: 'bg-gray-100 text-gray-800',
      basic: 'bg-blue-100 text-blue-800',
      premium: 'bg-purple-100 text-purple-800'
    };
    
    return (
      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
        colors[plan as keyof typeof colors] || colors.free
      }`}>
        {plan.charAt(0).toUpperCase() + plan.slice(1)}
      </span>
    );
  };

  return (
    <header className="bg-gray-800 border-b border-gray-700 px-6 py-4">
      <div className="flex items-center justify-between">
        {/* Left side - Page title */}
        <div className="flex items-center space-x-4">
          <h1 className="text-2xl font-bold text-white">
            {getPageTitle()}
          </h1>
          {getSubscriptionBadge()}
        </div>

        {/* Right side - Search and user actions */}
        <div className="flex items-center space-x-4">
          {/* Search */}
          <div className="relative">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <MagnifyingGlassIcon className="h-5 w-5 text-gray-400" />
            </div>
            <input
              type="text"
              placeholder="Search..."
              className="block w-64 pl-10 pr-3 py-2 border border-gray-600 rounded-lg bg-gray-700 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>

          {/* Notifications */}
          <button className="relative p-2 text-gray-400 hover:text-white transition-colors">
            <BellIcon className="h-6 w-6" />
            <span className="absolute top-0 right-0 block h-2 w-2 rounded-full bg-red-400 ring-2 ring-gray-800"></span>
          </button>

          {/* Settings */}
          <button className="p-2 text-gray-400 hover:text-white transition-colors">
            <Cog6ToothIcon className="h-6 w-6" />
          </button>

          {/* User menu */}
          <div className="relative">
            <button className="flex items-center space-x-3 text-gray-300 hover:text-white transition-colors">
              <div className="w-8 h-8 bg-gray-600 rounded-full flex items-center justify-center">
                {user?.firstName?.[0]}{user?.lastName?.[0]}
              </div>
              <div className="hidden md:block text-left">
                <div className="text-sm font-medium">
                  {user?.firstName} {user?.lastName}
                </div>
                <div className="text-xs text-gray-400">
                  {user?.email}
                </div>
              </div>
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};