'use client';

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

interface TutorialStep {
  id: string;
  title: string;
  description: string;
  component: string;
  selector?: string;
  position?: 'top' | 'bottom' | 'left' | 'right';
  subscriptionRequired?: 'free' | 'basic' | 'premium';
  category: 'dashboard' | 'portfolio' | 'trading' | 'analytics' | 'ai' | 'social' | 'research';
  duration: number; // in seconds
  interactive?: boolean;
  prerequisites?: string[];
}

interface TutorialModule {
  id: string;
  title: string;
  description: string;
  category: string;
  subscriptionRequired: 'free' | 'basic' | 'premium';
  estimatedTime: number; // in minutes
  steps: TutorialStep[];
  completed: boolean;
  progress: number;
}

interface TutorialContextType {
  // State
  currentModule: TutorialModule | null;
  currentStep: TutorialStep | null;
  isActive: boolean;
  modules: TutorialModule[];
  userProgress: Record<string, number>;
  completedModules: string[];
  
  // Actions
  startTutorial: (moduleId: string) => void;
  nextStep: () => void;
  previousStep: () => void;
  skipStep: () => void;
  completeTutorial: () => void;
  pauseTutorial: () => void;
  resumeTutorial: () => void;
  resetTutorial: (moduleId: string) => void;
  markStepComplete: (stepId: string) => void;
  
  // Utilities
  getTutorialForPage: (page: string) => TutorialModule | null;
  getAvailableModules: (subscription: string) => TutorialModule[];
  getRecommendedNext: () => TutorialModule | null;
}

const TutorialContext = createContext<TutorialContextType | undefined>(undefined);

// Tutorial modules data
const TUTORIAL_MODULES: TutorialModule[] = [
  {
    id: 'getting-started',
    title: 'Getting Started with FinScope',
    description: 'Learn the basics of navigating FinScope and understanding your dashboard',
    category: 'dashboard',
    subscriptionRequired: 'free',
    estimatedTime: 10,
    completed: false,
    progress: 0,
    steps: [
      {
        id: 'welcome',
        title: 'Welcome to FinScope',
        description: 'Get an overview of what FinScope can do for your financial journey',
        component: 'dashboard',
        category: 'dashboard',
        duration: 60,
        interactive: false
      },
      {
        id: 'navigation',
        title: 'Navigation Basics',
        description: 'Learn how to navigate between different sections',
        component: 'sidebar',
        selector: '[data-testid="sidebar"]',
        position: 'right',
        category: 'dashboard',
        duration: 90,
        interactive: true
      },
      {
        id: 'dashboard-overview',
        title: 'Dashboard Overview',
        description: 'Understand your main dashboard and key metrics',
        component: 'dashboard',
        selector: '[data-testid="dashboard-main"]',
        position: 'top',
        category: 'dashboard',
        duration: 120,
        interactive: true
      }
    ]
  },
  {
    id: 'portfolio-management',
    title: 'Portfolio Management',
    description: 'Learn how to create, manage, and optimize your investment portfolio',
    category: 'portfolio',
    subscriptionRequired: 'free',
    estimatedTime: 15,
    completed: false,
    progress: 0,
    steps: [
      {
        id: 'portfolio-basics',
        title: 'Portfolio Basics',
        description: 'Understanding portfolio allocation and diversification',
        component: 'portfolio',
        category: 'portfolio',
        duration: 180,
        interactive: false
      },
      {
        id: 'adding-assets',
        title: 'Adding Assets',
        description: 'Learn how to add stocks, crypto, and other assets to your portfolio',
        component: 'portfolio',
        selector: '[data-testid="add-asset-button"]',
        position: 'bottom',
        category: 'portfolio',
        duration: 150,
        interactive: true
      },
      {
        id: 'portfolio-analytics',
        title: 'Portfolio Analytics',
        description: 'Understand performance metrics and risk analysis',
        component: 'portfolio',
        selector: '[data-testid="portfolio-analytics"]',
        position: 'left',
        category: 'portfolio',
        duration: 200,
        interactive: true,
        subscriptionRequired: 'basic'
      }
    ]
  },
  {
    id: 'trading-interface',
    title: 'Trading Interface',
    description: 'Master the trading interface and execution strategies',
    category: 'trading',
    subscriptionRequired: 'basic',
    estimatedTime: 20,
    completed: false,
    progress: 0,
    steps: [
      {
        id: 'trading-basics',
        title: 'Trading Fundamentals',
        description: 'Understanding order types and market mechanics',
        component: 'trading',
        category: 'trading',
        duration: 240,
        interactive: false,
        subscriptionRequired: 'basic'
      },
      {
        id: 'placing-orders',
        title: 'Placing Orders',
        description: 'Learn how to place different types of orders',
        component: 'trading',
        selector: '[data-testid="order-form"]',
        position: 'right',
        category: 'trading',
        duration: 180,
        interactive: true,
        subscriptionRequired: 'basic'
      },
      {
        id: 'risk-management',
        title: 'Risk Management',
        description: 'Setting stop losses and position sizing',
        component: 'trading',
        selector: '[data-testid="risk-controls"]',
        position: 'top',
        category: 'trading',
        duration: 200,
        interactive: true,
        subscriptionRequired: 'basic'
      }
    ]
  },
  {
    id: 'advanced-analytics',
    title: 'Advanced Analytics',
    description: 'Deep dive into technical analysis and market insights',
    category: 'analytics',
    subscriptionRequired: 'premium',
    estimatedTime: 25,
    completed: false,
    progress: 0,
    steps: [
      {
        id: 'technical-analysis',
        title: 'Technical Analysis',
        description: 'Understanding charts, indicators, and patterns',
        component: 'analytics',
        category: 'analytics',
        duration: 300,
        interactive: false,
        subscriptionRequired: 'premium'
      },
      {
        id: 'custom-indicators',
        title: 'Custom Indicators',
        description: 'Creating and using custom technical indicators',
        component: 'analytics',
        selector: '[data-testid="custom-indicators"]',
        position: 'bottom',
        category: 'analytics',
        duration: 250,
        interactive: true,
        subscriptionRequired: 'premium'
      },
      {
        id: 'backtesting',
        title: 'Strategy Backtesting',
        description: 'Test your strategies against historical data',
        component: 'analytics',
        selector: '[data-testid="backtesting-panel"]',
        position: 'left',
        category: 'analytics',
        duration: 280,
        interactive: true,
        subscriptionRequired: 'premium'
      }
    ]
  },
  {
    id: 'ai-assistant',
    title: 'AI-Powered Insights',
    description: 'Leverage AI for market analysis and decision making',
    category: 'ai',
    subscriptionRequired: 'premium',
    estimatedTime: 18,
    completed: false,
    progress: 0,
    steps: [
      {
        id: 'ai-overview',
        title: 'AI Assistant Overview',
        description: 'Understanding AI capabilities and limitations',
        component: 'ai-chat',
        category: 'ai',
        duration: 180,
        interactive: false,
        subscriptionRequired: 'premium'
      },
      {
        id: 'market-analysis',
        title: 'AI Market Analysis',
        description: 'Getting AI-powered market insights and predictions',
        component: 'ai-chat',
        selector: '[data-testid="ai-analysis"]',
        position: 'right',
        category: 'ai',
        duration: 220,
        interactive: true,
        subscriptionRequired: 'premium'
      },
      {
        id: 'personalized-recommendations',
        title: 'Personalized Recommendations',
        description: 'Receiving tailored investment recommendations',
        component: 'ai-chat',
        selector: '[data-testid="ai-recommendations"]',
        position: 'top',
        category: 'ai',
        duration: 200,
        interactive: true,
        subscriptionRequired: 'premium'
      }
    ]
  },
  {
    id: 'social-trading',
    title: 'Social Trading & Community',
    description: 'Connect with other traders and share strategies',
    category: 'social',
    subscriptionRequired: 'basic',
    estimatedTime: 12,
    completed: false,
    progress: 0,
    steps: [
      {
        id: 'community-overview',
        title: 'Community Features',
        description: 'Exploring the FinScope trading community',
        component: 'community',
        category: 'social',
        duration: 150,
        interactive: false,
        subscriptionRequired: 'basic'
      },
      {
        id: 'following-traders',
        title: 'Following Top Traders',
        description: 'Learn how to follow and copy successful traders',
        component: 'community',
        selector: '[data-testid="trader-list"]',
        position: 'bottom',
        category: 'social',
        duration: 180,
        interactive: true,
        subscriptionRequired: 'basic'
      },
      {
        id: 'sharing-strategies',
        title: 'Sharing Your Strategies',
        description: 'Share your own trading strategies with the community',
        component: 'community',
        selector: '[data-testid="share-strategy"]',
        position: 'right',
        category: 'social',
        duration: 160,
        interactive: true,
        subscriptionRequired: 'basic'
      }
    ]
  },
  {
    id: 'research-tools',
    title: 'Research & Analysis Tools',
    description: 'Master the research tools for informed decision making',
    category: 'research',
    subscriptionRequired: 'basic',
    estimatedTime: 22,
    completed: false,
    progress: 0,
    steps: [
      {
        id: 'market-screener',
        title: 'Market Screener',
        description: 'Using filters to find investment opportunities',
        component: 'research',
        category: 'research',
        duration: 200,
        interactive: false,
        subscriptionRequired: 'basic'
      },
      {
        id: 'fundamental-analysis',
        title: 'Fundamental Analysis',
        description: 'Analyzing company financials and metrics',
        component: 'research',
        selector: '[data-testid="fundamental-data"]',
        position: 'left',
        category: 'research',
        duration: 250,
        interactive: true,
        subscriptionRequired: 'basic'
      },
      {
        id: 'news-sentiment',
        title: 'News & Sentiment Analysis',
        description: 'Understanding market sentiment and news impact',
        component: 'news',
        selector: '[data-testid="sentiment-analysis"]',
        position: 'top',
        category: 'research',
        duration: 180,
        interactive: true,
        subscriptionRequired: 'basic'
      }
    ]
  }
];

interface TutorialProviderProps {
  children: ReactNode;
}

export const TutorialProvider: React.FC<TutorialProviderProps> = ({ children }) => {
  const [currentModule, setCurrentModule] = useState<TutorialModule | null>(null);
  const [currentStep, setCurrentStep] = useState<TutorialStep | null>(null);
  const [isActive, setIsActive] = useState(false);
  const [modules, setModules] = useState<TutorialModule[]>(TUTORIAL_MODULES);
  const [userProgress, setUserProgress] = useState<Record<string, number>>({});
  const [completedModules, setCompletedModules] = useState<string[]>([]);

  // Load progress from localStorage on mount
  useEffect(() => {
    const savedProgress = localStorage.getItem('tutorial-progress');
    const savedCompleted = localStorage.getItem('tutorial-completed');
    
    if (savedProgress) {
      setUserProgress(JSON.parse(savedProgress));
    }
    
    if (savedCompleted) {
      setCompletedModules(JSON.parse(savedCompleted));
    }
  }, []);

  // Save progress to localStorage
  useEffect(() => {
    localStorage.setItem('tutorial-progress', JSON.stringify(userProgress));
  }, [userProgress]);

  useEffect(() => {
    localStorage.setItem('tutorial-completed', JSON.stringify(completedModules));
  }, [completedModules]);

  const startTutorial = (moduleId: string) => {
    const module = modules.find(m => m.id === moduleId);
    if (module) {
      setCurrentModule(module);
      setCurrentStep(module.steps[0] || null);
      setIsActive(true);
    }
  };

  const nextStep = () => {
    if (!currentModule || !currentStep) return;
    
    const currentIndex = currentModule.steps.findIndex(s => s.id === currentStep.id);
    const nextIndex = currentIndex + 1;
    
    if (nextIndex < currentModule.steps.length) {
      setCurrentStep(currentModule.steps[nextIndex]);
      
      // Update progress
      const progress = ((nextIndex + 1) / currentModule.steps.length) * 100;
      setUserProgress(prev => ({
        ...prev,
        [currentModule.id]: progress
      }));
    } else {
      completeTutorial();
    }
  };

  const previousStep = () => {
    if (!currentModule || !currentStep) return;
    
    const currentIndex = currentModule.steps.findIndex(s => s.id === currentStep.id);
    const prevIndex = currentIndex - 1;
    
    if (prevIndex >= 0) {
      setCurrentStep(currentModule.steps[prevIndex]);
      
      // Update progress
      const progress = ((prevIndex + 1) / currentModule.steps.length) * 100;
      setUserProgress(prev => ({
        ...prev,
        [currentModule.id]: progress
      }));
    }
  };

  const skipStep = () => {
    nextStep();
  };

  const completeTutorial = () => {
    if (currentModule) {
      setCompletedModules(prev => [...prev, currentModule.id]);
      setUserProgress(prev => ({
        ...prev,
        [currentModule.id]: 100
      }));
    }
    
    setCurrentModule(null);
    setCurrentStep(null);
    setIsActive(false);
  };

  const pauseTutorial = () => {
    setIsActive(false);
  };

  const resumeTutorial = () => {
    if (currentModule && currentStep) {
      setIsActive(true);
    }
  };

  const resetTutorial = (moduleId: string) => {
    setUserProgress(prev => ({
      ...prev,
      [moduleId]: 0
    }));
    
    setCompletedModules(prev => prev.filter(id => id !== moduleId));
    
    if (currentModule?.id === moduleId) {
      setCurrentModule(null);
      setCurrentStep(null);
      setIsActive(false);
    }
  };

  const markStepComplete = (stepId: string) => {
    if (!currentModule) return;
    
    const stepIndex = currentModule.steps.findIndex(s => s.id === stepId);
    if (stepIndex !== -1) {
      const progress = ((stepIndex + 1) / currentModule.steps.length) * 100;
      setUserProgress(prev => ({
        ...prev,
        [currentModule.id]: Math.max(prev[currentModule.id] || 0, progress)
      }));
    }
  };

  const getTutorialForPage = (page: string): TutorialModule | null => {
    return modules.find(module => 
      module.steps.some(step => step.component === page)
    ) || null;
  };

  const getAvailableModules = (subscription: string): TutorialModule[] => {
    const subscriptionLevels = { free: 0, basic: 1, premium: 2 };
    const userLevel = subscriptionLevels[subscription as keyof typeof subscriptionLevels] || 0;
    
    return modules.filter(module => {
      const requiredLevel = subscriptionLevels[module.subscriptionRequired];
      return userLevel >= requiredLevel;
    });
  };

  const getRecommendedNext = (): TutorialModule | null => {
    const availableModules = modules.filter(module => 
      !completedModules.includes(module.id)
    );
    
    // Recommend based on completion order and prerequisites
    const basicModules = availableModules.filter(m => 
      ['getting-started', 'portfolio-management'].includes(m.id)
    );
    
    if (basicModules.length > 0) {
      return basicModules[0];
    }
    
    return availableModules[0] || null;
  };

  const value: TutorialContextType = {
    // State
    currentModule,
    currentStep,
    isActive,
    modules,
    userProgress,
    completedModules,
    
    // Actions
    startTutorial,
    nextStep,
    previousStep,
    skipStep,
    completeTutorial,
    pauseTutorial,
    resumeTutorial,
    resetTutorial,
    markStepComplete,
    
    // Utilities
    getTutorialForPage,
    getAvailableModules,
    getRecommendedNext
  };

  return (
    <TutorialContext.Provider value={value}>
      {children}
    </TutorialContext.Provider>
  );
};

export const useTutorial = (): TutorialContextType => {
  const context = useContext(TutorialContext);
  if (!context) {
    throw new Error('useTutorial must be used within a TutorialProvider');
  }
  return context;
};

export type { TutorialStep, TutorialModule, TutorialContextType };