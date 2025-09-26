"use client";

import { useState, useCallback } from 'react';

export interface PricingPopupState {
  isVisible: boolean;
  feature?: string;
  plan?: string;
  title?: string;
  description?: string;
}

export const usePricingPopup = () => {
  const [popupState, setPopupState] = useState<PricingPopupState>({
    isVisible: false,
  });

  const showPricingPopup = useCallback((feature?: string, plan?: string, title?: string, description?: string) => {
    setPopupState({
      isVisible: true,
      feature,
      plan,
      title,
      description,
    });
  }, []);

  const hidePricingPopup = useCallback(() => {
    setPopupState({
      isVisible: false,
    });
  }, []);

  return {
    popupState,
    showPricingPopup,
    hidePricingPopup,
    isPricingPopupVisible: popupState.isVisible,
  };
};

export default usePricingPopup;