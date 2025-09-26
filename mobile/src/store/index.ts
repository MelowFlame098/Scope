import { configureStore } from '@reduxjs/toolkit';
import { persistStore, persistReducer } from 'redux-persist';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { combineReducers } from '@reduxjs/toolkit';

// Reducers
import authReducer from './slices/authSlice';
import portfolioReducer from './slices/portfolioSlice';
import tradingReducer from './slices/tradingSlice';
import marketDataReducer from './slices/marketDataSlice';
import newsReducer from './slices/newsSlice';
import notificationReducer from './slices/notificationSlice';
import settingsReducer from './slices/settingsSlice';
import socialReducer from './slices/socialSlice';
import aiReducer from './slices/aiSlice';

// API slices
import { apiSlice } from './api/apiSlice';

const persistConfig = {
  key: 'root',
  storage: AsyncStorage,
  whitelist: ['auth', 'settings', 'portfolio'], // Only persist these reducers
  blacklist: ['api'], // Don't persist API cache
};

const rootReducer = combineReducers({
  auth: authReducer,
  portfolio: portfolioReducer,
  trading: tradingReducer,
  marketData: marketDataReducer,
  news: newsReducer,
  notifications: notificationReducer,
  settings: settingsReducer,
  social: socialReducer,
  ai: aiReducer,
  api: apiSlice.reducer,
});

const persistedReducer = persistReducer(persistConfig, rootReducer);

export const store = configureStore({
  reducer: persistedReducer,
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: ['persist/PERSIST', 'persist/REHYDRATE'],
      },
    }).concat(apiSlice.middleware),
  devTools: __DEV__,
});

export const persistor = persistStore(store);

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;

// Typed hooks
import { useDispatch, useSelector, TypedUseSelectorHook } from 'react-redux';
export const useAppDispatch = () => useDispatch<AppDispatch>();
export const useAppSelector: TypedUseSelectorHook<RootState> = useSelector;