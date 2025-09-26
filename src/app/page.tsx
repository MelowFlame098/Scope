"use client";

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/contexts/AuthContext';

export default function Home() {
  const router = useRouter();
  const { isAuthenticated, isLoading } = useAuth();

  useEffect(() => {
    // Wait for auth check to complete
    if (isLoading) return;
    
    if (isAuthenticated) {
      // User is authenticated, redirect to dashboard
      router.push('/dashboard');
    } else {
      // User is not authenticated, redirect to landing page
      router.push('/landing');
    }
  }, [isAuthenticated, isLoading, router]);

  // Show loading state while redirecting
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900 flex items-center justify-center">
      <div className="text-center">
        <div className="w-16 h-16 border-4 border-blue-500/30 border-t-blue-500 rounded-full animate-spin mx-auto mb-4"></div>
        <p className="text-white text-lg">Loading FinScope...</p>
      </div>
    </div>
  );
}