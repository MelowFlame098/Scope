// CAPTCHA utility functions for client-side integration

interface CaptchaChallenge {
  challenge_id: string;
  challenge_type: 'math' | 'image' | 'recaptcha' | 'hcaptcha';
  challenge_data?: {
    question?: string;
    image_url?: string;
    site_key?: string;
  };
  expires_at: string;
}

interface CaptchaResponse {
  success: boolean;
  challenge?: CaptchaChallenge;
  error?: string;
}

class CaptchaManager {
  private currentChallenge: CaptchaChallenge | null = null;
  private callbacks: Map<string, (challenge: CaptchaChallenge | null) => void> = new Map();

  async getCaptchaChallenge(type: 'math' | 'image' | 'recaptcha' | 'hcaptcha' = 'math'): Promise<CaptchaChallenge | null> {
    try {
      const response = await fetch('/api/auth/captcha', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ type }),
      });

      const data: CaptchaResponse = await response.json();

      if (data.success && data.challenge) {
        this.currentChallenge = data.challenge;
        this.notifyCallbacks(data.challenge);
        return data.challenge;
      } else {
        console.error('Failed to get CAPTCHA challenge:', data.error);
        return null;
      }
    } catch (error) {
      console.error('CAPTCHA request failed:', error);
      return null;
    }
  }

  getCurrentChallenge(): CaptchaChallenge | null {
    return this.currentChallenge;
  }

  clearChallenge(): void {
    this.currentChallenge = null;
    this.notifyCallbacks(null);
  }

  isExpired(challenge: CaptchaChallenge): boolean {
    return new Date(challenge.expires_at) <= new Date();
  }

  onChallengeUpdate(id: string, callback: (challenge: CaptchaChallenge | null) => void): void {
    this.callbacks.set(id, callback);
  }

  offChallengeUpdate(id: string): void {
    this.callbacks.delete(id);
  }

  private notifyCallbacks(challenge: CaptchaChallenge | null): void {
    this.callbacks.forEach(callback => callback(challenge));
  }

  // Generate a simple math CAPTCHA for fallback
  generateMathChallenge(): { question: string; answer: number } {
    const operations = ['+', '-', '*'];
    const operation = operations[Math.floor(Math.random() * operations.length)];
    
    let num1: number, num2: number, answer: number;
    
    switch (operation) {
      case '+':
        num1 = Math.floor(Math.random() * 50) + 1;
        num2 = Math.floor(Math.random() * 50) + 1;
        answer = num1 + num2;
        break;
      case '-':
        num1 = Math.floor(Math.random() * 50) + 25;
        num2 = Math.floor(Math.random() * 25) + 1;
        answer = num1 - num2;
        break;
      case '*':
        num1 = Math.floor(Math.random() * 12) + 1;
        num2 = Math.floor(Math.random() * 12) + 1;
        answer = num1 * num2;
        break;
      default:
        num1 = 5;
        num2 = 3;
        answer = 8;
    }
    
    return {
      question: `${num1} ${operation} ${num2} = ?`,
      answer
    };
  }
}

// Create a singleton instance
export const captchaManager = new CaptchaManager();

// React hook for CAPTCHA integration
import { useState, useEffect, useCallback } from 'react';

export interface UseCaptchaReturn {
  challenge: CaptchaChallenge | null;
  isLoading: boolean;
  error: string | null;
  refreshChallenge: (type?: 'math' | 'image' | 'recaptcha' | 'hcaptcha') => Promise<void>;
  clearChallenge: () => void;
  isExpired: boolean;
}

export const useCaptcha = (autoLoad: boolean = false): UseCaptchaReturn => {
  const [challenge, setChallenge] = useState<CaptchaChallenge | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refreshChallenge = useCallback(async (type: 'math' | 'image' | 'recaptcha' | 'hcaptcha' = 'math') => {
    setIsLoading(true);
    setError(null);
    
    try {
      const newChallenge = await captchaManager.getCaptchaChallenge(type);
      setChallenge(newChallenge);
      
      if (!newChallenge) {
        setError('Failed to load CAPTCHA challenge');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load CAPTCHA');
    } finally {
      setIsLoading(false);
    }
  }, []);

  const clearChallenge = useCallback(() => {
    setChallenge(null);
    setError(null);
    captchaManager.clearChallenge();
  }, []);

  const isExpired = challenge ? captchaManager.isExpired(challenge) : false;

  useEffect(() => {
    if (autoLoad) {
      refreshChallenge();
    }
  }, [autoLoad, refreshChallenge]);

  useEffect(() => {
    const handleChallengeUpdate = (newChallenge: CaptchaChallenge | null) => {
      setChallenge(newChallenge);
    };

    const id = Math.random().toString(36).substr(2, 9);
    captchaManager.onChallengeUpdate(id, handleChallengeUpdate);

    return () => {
      captchaManager.offChallengeUpdate(id);
    };
  }, []);

  return {
    challenge,
    isLoading,
    error,
    refreshChallenge,
    clearChallenge,
    isExpired,
  };
};

// Utility functions for different CAPTCHA types
export const captchaUtils = {
  // Validate math CAPTCHA answer
  validateMathAnswer: (question: string, userAnswer: string): boolean => {
    try {
      // Extract the math expression from the question
      const expression = question.replace(' = ?', '').trim();
      
      // Safely evaluate the expression (only allow basic math operations)
      const allowedChars = /^[0-9+\-*\s()]+$/;
      if (!allowedChars.test(expression)) {
        return false;
      }
      
      // Use Function constructor for safe evaluation
      const result = new Function('return ' + expression)();
      return parseInt(userAnswer) === result;
    } catch {
      return false;
    }
  },

  // Format CAPTCHA token for API requests
  formatCaptchaToken: (challenge: CaptchaChallenge, userResponse: string): string => {
    return `${challenge.challenge_id}:${userResponse}`;
  },

  // Check if CAPTCHA is required based on error response
  isCaptchaRequired: (errorResponse: any): boolean => {
    return errorResponse?.requires_captcha === true || 
           errorResponse?.detail?.includes('CAPTCHA') ||
           errorResponse?.error?.includes('CAPTCHA');
  },

  // Get CAPTCHA error message
  getCaptchaErrorMessage: (challenge: CaptchaChallenge | null): string | null => {
    if (!challenge) {
      return 'CAPTCHA challenge not available';
    }
    
    if (captchaManager.isExpired(challenge)) {
      return 'CAPTCHA challenge has expired. Please refresh.';
    }
    
    return null;
  },
};

export default captchaManager;