import React from 'react';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createStackNavigator } from '@react-navigation/stack';
import Icon from 'react-native-vector-icons/MaterialIcons';
import { useAuth } from '../context/AuthContext';
import { useTheme } from '../context/ThemeContext';

// Auth Screens
import { LoginScreen } from '../screens/Auth/LoginScreen';
import { RegisterScreen } from '../screens/Auth/RegisterScreen';
import { BiometricSetupScreen } from '../screens/Auth/BiometricSetupScreen';

// Main Screens
import { DashboardScreen } from '../screens/Dashboard/DashboardScreen';
import { TradingScreen } from '../screens/Trading/TradingScreen';
import { PortfolioScreen } from '../screens/Portfolio/PortfolioScreen';
import { NewsScreen } from '../screens/News/NewsScreen';
import { ProfileScreen } from '../screens/Profile/ProfileScreen';

// Detail Screens
import { AssetDetailScreen } from '../screens/Trading/AssetDetailScreen';
import { OrderDetailScreen } from '../screens/Trading/OrderDetailScreen';
import { PortfolioDetailScreen } from '../screens/Portfolio/PortfolioDetailScreen';
import { NewsDetailScreen } from '../screens/News/NewsDetailScreen';
import { SettingsScreen } from '../screens/Profile/SettingsScreen';
import { NotificationScreen } from '../screens/Profile/NotificationScreen';

export type RootStackParamList = {
  Auth: undefined;
  Main: undefined;
  AssetDetail: { symbol: string; name: string };
  OrderDetail: { orderId: string };
  PortfolioDetail: { portfolioId: string };
  NewsDetail: { articleId: string };
  Settings: undefined;
  Notifications: undefined;
  BiometricSetup: undefined;
};

export type MainTabParamList = {
  Dashboard: undefined;
  Trading: undefined;
  Portfolio: undefined;
  News: undefined;
  Profile: undefined;
};

const Stack = createStackNavigator<RootStackParamList>();
const Tab = createBottomTabNavigator<MainTabParamList>();

const MainTabNavigator: React.FC = () => {
  const { theme } = useTheme();

  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        tabBarIcon: ({ focused, color, size }) => {
          let iconName: string;

          switch (route.name) {
            case 'Dashboard':
              iconName = 'dashboard';
              break;
            case 'Trading':
              iconName = 'trending-up';
              break;
            case 'Portfolio':
              iconName = 'account-balance-wallet';
              break;
            case 'News':
              iconName = 'article';
              break;
            case 'Profile':
              iconName = 'person';
              break;
            default:
              iconName = 'help';
          }

          return <Icon name={iconName} size={size} color={color} />;
        },
        tabBarActiveTintColor: theme.colors.primary,
        tabBarInactiveTintColor: theme.colors.textSecondary,
        tabBarStyle: {
          backgroundColor: theme.colors.surface,
          borderTopColor: theme.colors.border,
          paddingBottom: 5,
          height: 60,
        },
        headerStyle: {
          backgroundColor: theme.colors.surface,
          elevation: 0,
          shadowOpacity: 0,
        },
        headerTintColor: theme.colors.text,
        headerTitleStyle: {
          fontWeight: '600',
          fontSize: 18,
        },
      })}
    >
      <Tab.Screen 
        name="Dashboard" 
        component={DashboardScreen}
        options={{ title: 'Dashboard' }}
      />
      <Tab.Screen 
        name="Trading" 
        component={TradingScreen}
        options={{ title: 'Trading' }}
      />
      <Tab.Screen 
        name="Portfolio" 
        component={PortfolioScreen}
        options={{ title: 'Portfolio' }}
      />
      <Tab.Screen 
        name="News" 
        component={NewsScreen}
        options={{ title: 'News' }}
      />
      <Tab.Screen 
        name="Profile" 
        component={ProfileScreen}
        options={{ title: 'Profile' }}
      />
    </Tab.Navigator>
  );
};

const AuthStackNavigator: React.FC = () => {
  const { theme } = useTheme();

  return (
    <Stack.Navigator
      screenOptions={{
        headerStyle: {
          backgroundColor: theme.colors.surface,
          elevation: 0,
          shadowOpacity: 0,
        },
        headerTintColor: theme.colors.text,
        headerTitleStyle: {
          fontWeight: '600',
        },
        cardStyle: {
          backgroundColor: theme.colors.background,
        },
      }}
    >
      <Stack.Screen 
        name="Login" 
        component={LoginScreen}
        options={{ headerShown: false }}
      />
      <Stack.Screen 
        name="Register" 
        component={RegisterScreen}
        options={{ title: 'Create Account' }}
      />
      <Stack.Screen 
        name="BiometricSetup" 
        component={BiometricSetupScreen}
        options={{ title: 'Security Setup' }}
      />
    </Stack.Navigator>
  );
};

export const AppNavigator: React.FC = () => {
  const { isAuthenticated } = useAuth();
  const { theme } = useTheme();

  return (
    <Stack.Navigator
      screenOptions={{
        headerStyle: {
          backgroundColor: theme.colors.surface,
          elevation: 0,
          shadowOpacity: 0,
        },
        headerTintColor: theme.colors.text,
        headerTitleStyle: {
          fontWeight: '600',
        },
        cardStyle: {
          backgroundColor: theme.colors.background,
        },
      }}
    >
      {isAuthenticated ? (
        <>
          <Stack.Screen 
            name="Main" 
            component={MainTabNavigator}
            options={{ headerShown: false }}
          />
          <Stack.Screen 
            name="AssetDetail" 
            component={AssetDetailScreen}
            options={({ route }) => ({ 
              title: route.params.name,
              headerBackTitleVisible: false,
            })}
          />
          <Stack.Screen 
            name="OrderDetail" 
            component={OrderDetailScreen}
            options={{ 
              title: 'Order Details',
              headerBackTitleVisible: false,
            }}
          />
          <Stack.Screen 
            name="PortfolioDetail" 
            component={PortfolioDetailScreen}
            options={{ 
              title: 'Portfolio Details',
              headerBackTitleVisible: false,
            }}
          />
          <Stack.Screen 
            name="NewsDetail" 
            component={NewsDetailScreen}
            options={{ 
              title: 'Article',
              headerBackTitleVisible: false,
            }}
          />
          <Stack.Screen 
            name="Settings" 
            component={SettingsScreen}
            options={{ 
              title: 'Settings',
              headerBackTitleVisible: false,
            }}
          />
          <Stack.Screen 
            name="Notifications" 
            component={NotificationScreen}
            options={{ 
              title: 'Notifications',
              headerBackTitleVisible: false,
            }}
          />
        </>
      ) : (
        <Stack.Screen 
          name="Auth" 
          component={AuthStackNavigator}
          options={{ headerShown: false }}
        />
      )}
    </Stack.Navigator>
  );
};