import React, { useEffect } from 'react';
import { StatusBar, Platform } from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { Provider } from 'react-redux';
import { PersistGate } from 'redux-persist/integration/react';
import { store, persistor } from './src/store';
import { AppNavigator } from './src/navigation/AppNavigator';
import { AuthProvider } from './src/context/AuthContext';
import { WebSocketProvider } from './src/context/WebSocketContext';
import { NotificationProvider } from './src/context/NotificationContext';
import { ThemeProvider } from './src/context/ThemeContext';
import { LoadingScreen } from './src/components/common/LoadingScreen';
import { ErrorBoundary } from './src/components/common/ErrorBoundary';
import { initializeApp } from './src/services/AppInitializer';

const App: React.FC = () => {
  useEffect(() => {
    // Initialize app services
    initializeApp();
  }, []);

  return (
    <ErrorBoundary>
      <Provider store={store}>
        <PersistGate loading={<LoadingScreen />} persistor={persistor}>
          <ThemeProvider>
            <AuthProvider>
              <WebSocketProvider>
                <NotificationProvider>
                  <NavigationContainer>
                    <StatusBar
                      barStyle={Platform.OS === 'ios' ? 'dark-content' : 'light-content'}
                      backgroundColor="#1a1a1a"
                    />
                    <AppNavigator />
                  </NavigationContainer>
                </NotificationProvider>
              </WebSocketProvider>
            </AuthProvider>
          </ThemeProvider>
        </PersistGate>
      </Provider>
    </ErrorBoundary>
  );
};

export default App;