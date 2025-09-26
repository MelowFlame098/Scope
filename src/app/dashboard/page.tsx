"use client";

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { Sidebar } from '@/components/layout/Sidebar';
import { Header } from '@/components/layout/Header';
import { FiveAreasLayout } from '@/components/dashboard/FiveAreasLayout';
import { useStore } from '@/lib/store';
import { useWebSocketConnection } from '@/hooks/useWebSocketConnection';
import { useAuth } from '@/contexts/AuthContext';
import { ProtectedRoute } from '@/components/ProtectedRoute';
import { PricingPopup } from '@/components/PricingPopup';
import { usePricingPopup } from '@/hooks/usePricingPopup';

const DashboardPage = () => {
  const router = useRouter();
  const { user, isLoading, isAuthenticated, logout } = useAuth();
  const [error, setError] = useState('');
  const fetchAssets = useStore((state) => state.fetchAssets);
  const fetchModels = useStore((state) => state.fetchModels);
  const fetchNews = useStore((state) => state.fetchNews);
  const { connect } = useWebSocketConnection();
  const { showPricingPopup, hidePricingPopup, isPricingPopupVisible } = usePricingPopup();

  useEffect(() => {
    // Redirect to login if not authenticated
    if (!isLoading && !isAuthenticated) {
      router.push('/auth/login');
      return;
    }

    // Initialize app data when authenticated
    if (isAuthenticated && user) {
      initializeAppData();
    }
  }, [isAuthenticated, isLoading, user, router]);

  const initializeAppData = async () => {
    try {
      // Initialize data on component mount
      await Promise.all([
        fetchAssets(),
        fetchModels(),
        fetchNews()
      ]);
      
      // Connect to WebSocket
      connect();
    } catch (error) {
      console.error('Failed to initialize app data:', error);
      setError('Failed to load application data. Please refresh the page.');
    }
  };

  const handleLogout = async () => {
    try {
      await logout();
      router.push('/landing');
    } catch (error) {
      console.error('Logout failed:', error);
      // Force redirect even if logout fails
      router.push('/landing');
    }
  };

  // Show loading state while checking authentication
  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-blue-500/30 border-t-blue-500 rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-white text-lg">Loading Dashboard...</p>
        </div>
      </div>
    );
  }

  // Show error state
  if (error) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-center max-w-md">
          <div className="w-16 h-16 bg-red-500/20 rounded-full flex items-center justify-center mx-auto mb-4">
            <svg className="w-8 h-8 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
            </svg>
          </div>
          <h1 className="text-2xl font-bold text-white mb-4">Error Loading Dashboard</h1>
          <p className="text-gray-300 mb-6">{error}</p>
          <button
            onClick={() => window.location.reload()}
            className="bg-blue-500 hover:bg-blue-600 text-white px-6 py-2 rounded-lg transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  // Show email verification warning if needed
  const EmailVerificationBanner = () => {
    if (!user || user.isEmailVerified) return null;

    return (
      <div className="bg-yellow-500/20 border-l-4 border-yellow-500 p-4 mb-4">
        <div className="flex items-center">
          <div className="flex-shrink-0">
            <svg className="h-5 w-5 text-yellow-400" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
          </div>
          <div className="ml-3">
            <p className="text-sm text-yellow-300">
              Please verify your email address to access all features.{' '}
              <button className="font-medium underline hover:text-yellow-200">
                Resend verification email
              </button>
            </p>
          </div>
        </div>
      </div>
    );
  };

  return (
    <ProtectedRoute requiredPlan="free">
      <div className="flex h-screen bg-gray-900">
        <Sidebar user={user} onLogout={handleLogout} />
        <div className="flex-1 flex flex-col overflow-hidden">
          <Header user={user} onLogout={handleLogout} />
          <main className="flex-1 flex overflow-hidden">
            <FiveAreasLayout user={user} />
          </main>
        </div>
      </div>
      
      {/* Pricing Popup */}
      {isPricingPopupVisible && (
        <PricingPopup
          isOpen={isPricingPopupVisible}
          onClose={hidePricingPopup}
          feature="Dashboard Access"
          description="Access to the main dashboard with basic portfolio tracking and market data."
        />
      )}
    </ProtectedRoute>
  );
};

export default DashboardPage;