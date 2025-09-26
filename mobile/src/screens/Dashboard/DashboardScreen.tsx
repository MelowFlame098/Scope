import React, { useEffect, useState, useCallback } from 'react';
import {
  View,
  Text,
  ScrollView,
  RefreshControl,
  TouchableOpacity,
  Dimensions,
  StyleSheet,
} from 'react-native';
import { LineChart, PieChart } from 'react-native-chart-kit';
import Icon from 'react-native-vector-icons/MaterialIcons';
import { useAppSelector, useAppDispatch } from '../../store';
import { useTheme } from '../../context/ThemeContext';
import { useWebSocket } from '../../context/WebSocketContext';
import { fetchPortfolioSummary } from '../../store/slices/portfolioSlice';
import { fetchMarketOverview } from '../../store/slices/marketDataSlice';
import { Card } from '../../components/common/Card';
import { LoadingSpinner } from '../../components/common/LoadingSpinner';
import { ErrorMessage } from '../../components/common/ErrorMessage';
import { PriceChangeIndicator } from '../../components/common/PriceChangeIndicator';
import { QuickActionButton } from '../../components/common/QuickActionButton';
import { WatchlistWidget } from '../../components/widgets/WatchlistWidget';
import { NewsWidget } from '../../components/widgets/NewsWidget';
import { AIInsightsWidget } from '../../components/widgets/AIInsightsWidget';

const { width: screenWidth } = Dimensions.get('window');

export const DashboardScreen: React.FC = () => {
  const dispatch = useAppDispatch();
  const { theme } = useTheme();
  const { isConnected } = useWebSocket();
  const [refreshing, setRefreshing] = useState(false);

  const {
    user,
    portfolio,
    marketData,
    isLoading,
    error,
  } = useAppSelector((state) => ({
    user: state.auth.user,
    portfolio: state.portfolio.summary,
    marketData: state.marketData.overview,
    isLoading: state.portfolio.isLoading || state.marketData.isLoading,
    error: state.portfolio.error || state.marketData.error,
  }));

  const loadData = useCallback(async () => {
    try {
      await Promise.all([
        dispatch(fetchPortfolioSummary()).unwrap(),
        dispatch(fetchMarketOverview()).unwrap(),
      ]);
    } catch (error) {
      console.error('Failed to load dashboard data:', error);
    }
  }, [dispatch]);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await loadData();
    setRefreshing(false);
  }, [loadData]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const portfolioChartData = {
    labels: portfolio?.performanceHistory?.labels || [],
    datasets: [
      {
        data: portfolio?.performanceHistory?.values || [],
        color: (opacity = 1) => `rgba(34, 197, 94, ${opacity})`,
        strokeWidth: 2,
      },
    ],
  };

  const allocationData = portfolio?.allocation?.map((item, index) => ({
    name: item.symbol,
    population: item.percentage,
    color: theme.colors.chartColors[index % theme.colors.chartColors.length],
    legendFontColor: theme.colors.text,
    legendFontSize: 12,
  })) || [];

  const quickActions = [
    {
      icon: 'trending-up',
      label: 'Trade',
      onPress: () => {/* Navigate to trading */},
      color: theme.colors.success,
    },
    {
      icon: 'account-balance-wallet',
      label: 'Portfolio',
      onPress: () => {/* Navigate to portfolio */},
      color: theme.colors.primary,
    },
    {
      icon: 'analytics',
      label: 'Analytics',
      onPress: () => {/* Navigate to analytics */},
      color: theme.colors.warning,
    },
    {
      icon: 'notifications',
      label: 'Alerts',
      onPress: () => {/* Navigate to alerts */},
      color: theme.colors.info,
    },
  ];

  if (isLoading && !portfolio) {
    return (
      <View style={[styles.container, { backgroundColor: theme.colors.background }]}>
        <LoadingSpinner size="large" />
      </View>
    );
  }

  if (error) {
    return (
      <View style={[styles.container, { backgroundColor: theme.colors.background }]}>
        <ErrorMessage message={error} onRetry={loadData} />
      </View>
    );
  }

  return (
    <ScrollView
      style={[styles.container, { backgroundColor: theme.colors.background }]}
      refreshControl={
        <RefreshControl
          refreshing={refreshing}
          onRefresh={onRefresh}
          tintColor={theme.colors.primary}
          colors={[theme.colors.primary]}
        />
      }
      showsVerticalScrollIndicator={false}
    >
      {/* Header */}
      <View style={styles.header}>
        <View>
          <Text style={[styles.greeting, { color: theme.colors.textSecondary }]}>
            Good {getTimeOfDay()}, {user?.firstName}
          </Text>
          <View style={styles.connectionStatus}>
            <View
              style={[
                styles.connectionDot,
                { backgroundColor: isConnected ? theme.colors.success : theme.colors.error },
              ]}
            />
            <Text style={[styles.connectionText, { color: theme.colors.textSecondary }]}>
              {isConnected ? 'Live Data' : 'Offline'}
            </Text>
          </View>
        </View>
        <TouchableOpacity style={styles.notificationButton}>
          <Icon name="notifications" size={24} color={theme.colors.text} />
          {/* Notification badge */}
          <View style={[styles.notificationBadge, { backgroundColor: theme.colors.error }]}>
            <Text style={styles.notificationBadgeText}>3</Text>
          </View>
        </TouchableOpacity>
      </View>

      {/* Portfolio Summary */}
      <Card style={styles.portfolioCard}>
        <View style={styles.portfolioHeader}>
          <Text style={[styles.portfolioTitle, { color: theme.colors.text }]}>
            Portfolio Value
          </Text>
          <TouchableOpacity>
            <Icon name="more-vert" size={20} color={theme.colors.textSecondary} />
          </TouchableOpacity>
        </View>
        <Text style={[styles.portfolioValue, { color: theme.colors.text }]}>
          ${portfolio?.totalValue?.toLocaleString() || '0'}
        </Text>
        <View style={styles.portfolioChange}>
          <PriceChangeIndicator
            value={portfolio?.dayChange || 0}
            percentage={portfolio?.dayChangePercent || 0}
            showIcon
          />
          <Text style={[styles.portfolioChangeText, { color: theme.colors.textSecondary }]}>
            Today
          </Text>
        </View>
      </Card>

      {/* Quick Actions */}
      <View style={styles.quickActionsContainer}>
        <Text style={[styles.sectionTitle, { color: theme.colors.text }]}>Quick Actions</Text>
        <View style={styles.quickActions}>
          {quickActions.map((action, index) => (
            <QuickActionButton
              key={index}
              icon={action.icon}
              label={action.label}
              onPress={action.onPress}
              color={action.color}
            />
          ))}
        </View>
      </View>

      {/* Portfolio Performance Chart */}
      {portfolio?.performanceHistory && (
        <Card style={styles.chartCard}>
          <View style={styles.chartHeader}>
            <Text style={[styles.chartTitle, { color: theme.colors.text }]}>
              Portfolio Performance
            </Text>
            <View style={styles.chartPeriodSelector}>
              {['1D', '1W', '1M', '3M', '1Y'].map((period) => (
                <TouchableOpacity
                  key={period}
                  style={[
                    styles.periodButton,
                    period === '1M' && { backgroundColor: theme.colors.primary },
                  ]}
                >
                  <Text
                    style={[
                      styles.periodButtonText,
                      {
                        color: period === '1M' ? theme.colors.surface : theme.colors.textSecondary,
                      },
                    ]}
                  >
                    {period}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>
          <LineChart
            data={portfolioChartData}
            width={screenWidth - 60}
            height={200}
            chartConfig={{
              backgroundColor: theme.colors.surface,
              backgroundGradientFrom: theme.colors.surface,
              backgroundGradientTo: theme.colors.surface,
              decimalPlaces: 2,
              color: (opacity = 1) => `rgba(34, 197, 94, ${opacity})`,
              labelColor: (opacity = 1) => `rgba(${theme.colors.textSecondary}, ${opacity})`,
              style: {
                borderRadius: 16,
              },
              propsForDots: {
                r: '0',
              },
            }}
            bezier
            style={styles.chart}
          />
        </Card>
      )}

      {/* Asset Allocation */}
      {allocationData.length > 0 && (
        <Card style={styles.chartCard}>
          <Text style={[styles.chartTitle, { color: theme.colors.text }]}>
            Asset Allocation
          </Text>
          <PieChart
            data={allocationData}
            width={screenWidth - 60}
            height={200}
            chartConfig={{
              color: (opacity = 1) => `rgba(255, 255, 255, ${opacity})`,
            }}
            accessor="population"
            backgroundColor="transparent"
            paddingLeft="15"
            absolute
          />
        </Card>
      )}

      {/* Market Overview */}
      <Card style={styles.marketCard}>
        <Text style={[styles.sectionTitle, { color: theme.colors.text }]}>Market Overview</Text>
        <View style={styles.marketGrid}>
          {marketData?.indices?.map((index, i) => (
            <View key={i} style={styles.marketItem}>
              <Text style={[styles.marketSymbol, { color: theme.colors.text }]}>
                {index.symbol}
              </Text>
              <Text style={[styles.marketPrice, { color: theme.colors.text }]}>
                {index.price}
              </Text>
              <PriceChangeIndicator
                value={index.change}
                percentage={index.changePercent}
                size="small"
              />
            </View>
          )) || []}
        </View>
      </Card>

      {/* Widgets */}
      <WatchlistWidget />
      <AIInsightsWidget />
      <NewsWidget />

      {/* Bottom spacing */}
      <View style={styles.bottomSpacing} />
    </ScrollView>
  );
};

const getTimeOfDay = (): string => {
  const hour = new Date().getHours();
  if (hour < 12) return 'morning';
  if (hour < 18) return 'afternoon';
  return 'evening';
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingVertical: 16,
  },
  greeting: {
    fontSize: 16,
    fontWeight: '500',
  },
  connectionStatus: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 4,
  },
  connectionDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 6,
  },
  connectionText: {
    fontSize: 12,
  },
  notificationButton: {
    position: 'relative',
  },
  notificationBadge: {
    position: 'absolute',
    top: -4,
    right: -4,
    width: 16,
    height: 16,
    borderRadius: 8,
    justifyContent: 'center',
    alignItems: 'center',
  },
  notificationBadgeText: {
    color: 'white',
    fontSize: 10,
    fontWeight: 'bold',
  },
  portfolioCard: {
    marginHorizontal: 20,
    marginBottom: 20,
    padding: 20,
  },
  portfolioHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  portfolioTitle: {
    fontSize: 16,
    fontWeight: '500',
  },
  portfolioValue: {
    fontSize: 32,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  portfolioChange: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  portfolioChangeText: {
    marginLeft: 8,
    fontSize: 14,
  },
  quickActionsContainer: {
    paddingHorizontal: 20,
    marginBottom: 20,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 16,
  },
  quickActions: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  chartCard: {
    marginHorizontal: 20,
    marginBottom: 20,
    padding: 20,
  },
  chartHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  chartTitle: {
    fontSize: 16,
    fontWeight: '600',
  },
  chartPeriodSelector: {
    flexDirection: 'row',
  },
  periodButton: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
    marginLeft: 8,
  },
  periodButtonText: {
    fontSize: 12,
    fontWeight: '500',
  },
  chart: {
    marginVertical: 8,
    borderRadius: 16,
  },
  marketCard: {
    marginHorizontal: 20,
    marginBottom: 20,
    padding: 20,
  },
  marketGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  marketItem: {
    width: '48%',
    padding: 12,
    borderRadius: 8,
    marginBottom: 12,
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
  },
  marketSymbol: {
    fontSize: 14,
    fontWeight: '600',
    marginBottom: 4,
  },
  marketPrice: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  bottomSpacing: {
    height: 20,
  },
});