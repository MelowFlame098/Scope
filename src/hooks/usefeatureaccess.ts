import { useSubscription } from '../contexts/SubscriptionContext';

export interface FeatureAccess {
  // Trading Features
  basicTrading: boolean;
  advancedTrading: boolean;
  algorithmicTrading: boolean;
  
  // Portfolio Features
  basicPortfolio: boolean;
  advancedPortfolio: boolean;
  multiplePortfolios: boolean;
  
  // Analytics Features
  basicAnalytics: boolean;
  advancedAnalytics: boolean;
  aiInsights: boolean;
  
  // Data Features
  realTimeData: boolean;
  historicalData: boolean;
  premiumData: boolean;
  
  // Social Features
  communityAccess: boolean;
  socialTrading: boolean;
  copyTrading: boolean;
  
  // Professional Features
  institutionalTools: boolean;
  complianceTools: boolean;
  apiAccess: boolean;
  
  // Support Features
  basicSupport: boolean;
  prioritySupport: boolean;
  dedicatedSupport: boolean;
}

export interface FeatureLimits {
  maxPortfolios: number;
  maxWatchlistItems: number;
  maxAlerts: number;
  maxApiCalls: number;
  dataRefreshRate: number; // in seconds
}

export const useFeatureAccess = () => {
  const { subscription, hasFeatureAccess } = useSubscription();
  
  const getFeatureAccess = (): FeatureAccess => {
    // Map backend plan format to frontend format
    const backendPlan = subscription?.plan?.toLowerCase() || 'free';
    const planType = backendPlan === 'basic' ? 'basic' : backendPlan === 'premium' ? 'premium' : 'free';
    
    switch (planType) {
      case 'free':
        return {
          // Trading Features
          basicTrading: true,
          advancedTrading: false,
          algorithmicTrading: false,
          
          // Portfolio Features
          basicPortfolio: true,
          advancedPortfolio: false,
          multiplePortfolios: false,
          
          // Analytics Features
          basicAnalytics: true,
          advancedAnalytics: false,
          aiInsights: false,
          
          // Data Features
          realTimeData: false,
          historicalData: true,
          premiumData: false,
          
          // Social Features
          communityAccess: true,
          socialTrading: false,
          copyTrading: false,
          
          // Professional Features
          institutionalTools: false,
          complianceTools: false,
          apiAccess: false,
          
          // Support Features
          basicSupport: true,
          prioritySupport: false,
          dedicatedSupport: false,
        };
        
      case 'basic':
        return {
          // Trading Features
          basicTrading: true,
          advancedTrading: true,
          algorithmicTrading: false,
          
          // Portfolio Features
          basicPortfolio: true,
          advancedPortfolio: true,
          multiplePortfolios: true,
          
          // Analytics Features
          basicAnalytics: true,
          advancedAnalytics: true,
          aiInsights: true,
          
          // Data Features
          realTimeData: true,
          historicalData: true,
          premiumData: false,
          
          // Social Features
          communityAccess: true,
          socialTrading: true,
          copyTrading: false,
          
          // Professional Features
          institutionalTools: false,
          complianceTools: false,
          apiAccess: false,
          
          // Support Features
          basicSupport: true,
          prioritySupport: true,
          dedicatedSupport: false,
        };
        
      case 'premium':
        return {
          // Trading Features
          basicTrading: true,
          advancedTrading: true,
          algorithmicTrading: true,
          
          // Portfolio Features
          basicPortfolio: true,
          advancedPortfolio: true,
          multiplePortfolios: true,
          
          // Analytics Features
          basicAnalytics: true,
          advancedAnalytics: true,
          aiInsights: true,
          
          // Data Features
          realTimeData: true,
          historicalData: true,
          premiumData: true,
          
          // Social Features
          communityAccess: true,
          socialTrading: true,
          copyTrading: true,
          
          // Professional Features
          institutionalTools: true,
          complianceTools: true,
          apiAccess: true,
          
          // Support Features
          basicSupport: true,
          prioritySupport: true,
          dedicatedSupport: true,
        };
        
      default:
        // Default to free tier
        return getFeatureAccess();
    }
  };
  
  const getFeatureLimits = (): FeatureLimits => {
    // Map backend plan format to frontend format
    const backendPlan = subscription?.plan?.toLowerCase() || 'free';
    const planType = backendPlan === 'basic' ? 'basic' : backendPlan === 'premium' ? 'premium' : 'free';
    
    switch (planType) {
      case 'free':
        return {
          maxPortfolios: 1,
          maxWatchlistItems: 10,
          maxAlerts: 5,
          maxApiCalls: 100,
          dataRefreshRate: 60, // 1 minute
        };
        
      case 'basic':
        return {
          maxPortfolios: 5,
          maxWatchlistItems: 50,
          maxAlerts: 25,
          maxApiCalls: 1000,
          dataRefreshRate: 30, // 30 seconds
        };
        
      case 'premium':
        return {
          maxPortfolios: -1, // unlimited
          maxWatchlistItems: -1, // unlimited
          maxAlerts: -1, // unlimited
          maxApiCalls: 10000,
          dataRefreshRate: 5, // 5 seconds
        };
        
      default:
        return getFeatureLimits();
    }
  };
  
  const checkFeatureAccess = (feature: keyof FeatureAccess): boolean => {
    return getFeatureAccess()[feature];
  };
  
  const checkLimit = (limitType: keyof FeatureLimits, currentValue: number): boolean => {
    const limits = getFeatureLimits();
    const limit = limits[limitType];
    
    // -1 means unlimited
    if (limit === -1) return true;
    
    return currentValue < limit;
  };
  
  const getUpgradeMessage = (feature: keyof FeatureAccess): string => {
    // Map backend plan format to frontend format
    const backendPlan = subscription?.plan?.toLowerCase() || 'free';
    const planType = backendPlan === 'basic' ? 'basic' : backendPlan === 'premium' ? 'premium' : 'free';
    
    if (planType === 'free') {
      return 'Upgrade to Basic ($4.99/month) or Premium ($9.99/month) to access this feature.';
    } else if (planType === 'basic') {
      return 'Upgrade to Premium ($9.99/month) to access this feature.';
    }
    
    return 'This feature is available in your current plan.';
  };
  
  return {
    featureAccess: getFeatureAccess(),
    featureLimits: getFeatureLimits(),
    checkFeatureAccess,
    checkLimit,
    getUpgradeMessage,
    subscription,
    hasFeatureAccess,
  };
};

export default useFeatureAccess;