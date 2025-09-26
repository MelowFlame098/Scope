"use client";

import React, { useEffect, useState } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import { usePricingPopup } from '@/hooks/usePricingPopup';
import PricingPopup from './PricingPopup';
import { useRouter } from 'next/navigation';
import { Loader2, Lock, Shield } from 'lucide-react';

interface ProtectedRouteProps {
  children: React.ReactNode;
  requiredPlan?: 'free' | 'basic' | 'premium';
  feature?: string;
  title?: string;
  description?: string;
  fallback?: React.ReactNode;
  redirectOnFail?: boolean;
}

export const ProtectedRoute: React.FC<ProtectedRouteProps> = ({
  children,
  requiredPlan = 'free',
  feature,
  title,
  description,
  fallback,
  redirectOnFail = false
}) => {
  const { user, isLoading, checkSubscriptionAccess } = useAuth();
  const { popupState, showPricingPopup, hidePricingPopup } = usePricingPopup();
  const router = useRouter();
  const [hasAccess, setHasAccess] = useState<boolean | null>(null);

  useEffect(() => {
    if (!isLoading && user) {
      const access = checkSubscriptionAccess(requiredPlan);
      setHasAccess(access);
      
      if (!access) {
        if (redirectOnFail) {
          router.push('/pricing?upgrade=true&feature=' + encodeURIComponent(feature || 'premium'));
        } else if (requiredPlan !== 'free') {
          showPricingPopup(
            feature || 'premium-feature',
            requiredPlan as 'basic' | 'premium',
            title,
            description
          );
        }
      }
    }
  }, [user, isLoading, requiredPlan, feature, title, description, checkSubscriptionAccess, showPricingPopup, redirectOnFail, router]);

  // Show loading state
  if (isLoading || hasAccess === null) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <Loader2 className="w-8 h-8 animate-spin mx-auto mb-4 text-blue-600" />
          <p className="text-gray-600">Loading...</p>
        </div>
      </div>
    );
  }

  // Show access denied state
  if (!hasAccess) {
    if (fallback) {
      return (
        <>
          {fallback}
          <PricingPopup
            isOpen={popupState.isVisible}
            onClose={hidePricingPopup}
            feature={popupState.feature}
            requiredPlan={popupState.plan as 'basic' | 'premium'}
            title={popupState.title}
            description={popupState.description}
          />
        </>
      );
    }

    return (
      <>
        <div className="flex items-center justify-center min-h-[400px]">
          <div className="text-center max-w-md mx-auto p-6">
            <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <Lock className="w-8 h-8 text-gray-400" />
            </div>
            <h3 className="text-xl font-semibold text-gray-900 mb-2">
              {title || 'Premium Feature'}
            </h3>
            <p className="text-gray-600 mb-6">
              {description || `This feature requires a ${requiredPlan} subscription to access.`}
            </p>
            <button
              onClick={() => showPricingPopup(
                feature || 'premium-feature',
                requiredPlan as 'basic' | 'premium',
                title,
                description
              )}
              className="bg-blue-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-blue-700 transition-colors"
            >
              View Pricing Plans
            </button>
          </div>
        </div>
        <PricingPopup
          isOpen={popupState.isVisible}
          onClose={hidePricingPopup}
          feature={popupState.feature}
          requiredPlan={popupState.plan as 'basic' | 'premium'}
          title={popupState.title}
          description={popupState.description}
        />
      </>
    );
  }

  // User has access, render children
  return (
    <>
      {children}
      <PricingPopup
        isOpen={popupState.isVisible}
        onClose={hidePricingPopup}
        feature={popupState.feature}
        requiredPlan={popupState.plan as 'basic' | 'premium'}
        title={popupState.title}
        description={popupState.description}
      />
    </>
  );
};

// Higher-order component version
export const withSubscriptionProtection = <P extends object>(
  Component: React.ComponentType<P>,
  options: {
    requiredPlan?: 'free' | 'basic' | 'premium';
    feature?: string;
    title?: string;
    description?: string;
    fallback?: React.ReactNode;
    redirectOnFail?: boolean;
  } = {}
) => {
  const ProtectedComponent: React.FC<P> = (props) => {
    return (
      <ProtectedRoute {...options}>
        <Component {...props} />
      </ProtectedRoute>
    );
  };

  ProtectedComponent.displayName = `withSubscriptionProtection(${Component.displayName || Component.name})`;
  
  return ProtectedComponent;
};

// Feature access component for inline protection
interface FeatureAccessProps {
  requiredPlan: 'free' | 'basic' | 'premium';
  feature?: string;
  title?: string;
  description?: string;
  children: React.ReactNode;
  fallback?: React.ReactNode;
  showUpgradeButton?: boolean;
}

export const FeatureAccess: React.FC<FeatureAccessProps> = ({
  requiredPlan,
  feature,
  title,
  description,
  children,
  fallback,
  showUpgradeButton = true
}) => {
  const { checkSubscriptionAccess } = useAuth();
  const { popupState, showPricingPopup, hidePricingPopup } = usePricingPopup();
  const hasAccess = checkSubscriptionAccess(requiredPlan);

  if (!hasAccess) {
    if (fallback) {
      return (
        <>
          {fallback}
          <PricingPopup
            isOpen={popupState.isVisible}
            onClose={hidePricingPopup}
            feature={popupState.feature}
            requiredPlan={popupState.plan as 'basic' | 'premium'}
            title={popupState.title}
            description={popupState.description}
          />
        </>
      );
    }

    return (
      <>
        <div className="relative">
          <div className="absolute inset-0 bg-gray-50 bg-opacity-90 flex items-center justify-center z-10 rounded-lg">
            <div className="text-center p-4">
              <Shield className="w-8 h-8 text-gray-400 mx-auto mb-2" />
              <p className="text-sm text-gray-600 mb-3">
                {title || `${requiredPlan.charAt(0).toUpperCase() + requiredPlan.slice(1)} plan required`}
              </p>
              {showUpgradeButton && (
                <button
                  onClick={() => showPricingPopup(
                    feature || 'premium-feature',
                    requiredPlan as 'basic' | 'premium',
                    title,
                    description
                  )}
                  className="bg-blue-600 text-white px-4 py-2 rounded text-sm font-medium hover:bg-blue-700 transition-colors"
                >
                  Upgrade
                </button>
              )}
            </div>
          </div>
          <div className="filter blur-sm pointer-events-none">
            {children}
          </div>
        </div>
        <PricingPopup
          isOpen={popupState.isVisible}
          onClose={hidePricingPopup}
          feature={popupState.feature}
          requiredPlan={popupState.plan as 'basic' | 'premium'}
          title={popupState.title}
          description={popupState.description}
        />
      </>
    );
  }

  return <>{children}</>;
};



export default ProtectedRoute;