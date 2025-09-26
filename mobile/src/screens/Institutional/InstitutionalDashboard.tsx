import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  RefreshControl,
  FlatList,
  Alert,
  StyleSheet,
  Dimensions,
} from 'react-native';
import { useSelector, useDispatch } from 'react-redux';
import Icon from 'react-native-vector-icons/MaterialIcons';
import { LineChart, PieChart, BarChart } from 'react-native-chart-kit';
import { RootState } from '../../store';
import { institutionalService, RiskMetrics, ComplianceViolation, PerformanceAttribution } from '../../services/InstitutionalService';

const { width: screenWidth } = Dimensions.get('window');

interface InstitutionalDashboardProps {
  navigation: any;
}

const InstitutionalDashboard: React.FC<InstitutionalDashboardProps> = ({ navigation }) => {
  const dispatch = useDispatch();
  const { user } = useSelector((state: RootState) => state.auth);
  const { theme } = useSelector((state: RootState) => state.settings);

  const [activeTab, setActiveTab] = useState<'overview' | 'risk' | 'compliance' | 'performance'>('overview');
  const [refreshing, setRefreshing] = useState(false);
  const [loading, setLoading] = useState(false);

  // Data states
  const [portfolios, setPortfolios] = useState<any[]>([]);
  const [riskMetrics, setRiskMetrics] = useState<RiskMetrics | null>(null);
  const [complianceViolations, setComplianceViolations] = useState<ComplianceViolation[]>([]);
  const [performanceData, setPerformanceData] = useState<PerformanceAttribution | null>(null);
  const [recentTrades, setRecentTrades] = useState<any[]>([]);
  const [alerts, setAlerts] = useState<any[]>([]);

  const isDark = theme === 'dark';
  const colors = {
    background: isDark ? '#1a1a1a' : '#ffffff',
    surface: isDark ? '#2d2d2d' : '#f8f9fa',
    text: isDark ? '#ffffff' : '#000000',
    textSecondary: isDark ? '#b0b0b0' : '#666666',
    primary: '#007AFF',
    success: '#34C759',
    danger: '#FF3B30',
    warning: '#FF9500',
    border: isDark ? '#404040' : '#e0e0e0',
  };

  useEffect(() => {
    loadInitialData();
    setupRealTimeUpdates();

    return () => {
      institutionalService.disconnect();
    };
  }, []);

  const loadInitialData = async () => {
    setLoading(true);
    try {
      await Promise.all([
        loadPortfolios(),
        loadRiskMetrics(),
        loadComplianceViolations(),
        loadPerformanceData(),
        loadRecentTrades(),
        loadAlerts(),
      ]);
    } catch (error) {
      console.error('Failed to load initial data:', error);
    } finally {
      setLoading(false);
    }
  };

  const setupRealTimeUpdates = () => {
    institutionalService.subscribeToComplianceAlerts((violation: ComplianceViolation) => {
      setComplianceViolations(prev => [violation, ...prev]);
      setAlerts(prev => [{
        id: violation.id,
        type: 'compliance',
        severity: violation.severity,
        message: violation.description,
        timestamp: violation.detectedAt,
      }, ...prev]);
    });

    institutionalService.subscribeToRiskAlerts((alert: any) => {
      setAlerts(prev => [{
        id: alert.id,
        type: 'risk',
        severity: alert.severity,
        message: alert.message,
        timestamp: new Date(),
      }, ...prev]);
    });
  };

  const loadPortfolios = async () => {
    try {
      // Mock data for demonstration
      setPortfolios([
        {
          id: '1',
          name: 'Global Equity Fund',
          aum: 250000000,
          return: 12.5,
          risk: 'Medium',
          allocation: {
            equities: 65,
            bonds: 25,
            alternatives: 10,
          },
        },
        {
          id: '2',
          name: 'Fixed Income Portfolio',
          aum: 180000000,
          return: 6.8,
          risk: 'Low',
          allocation: {
            equities: 15,
            bonds: 75,
            alternatives: 10,
          },
        },
      ]);
    } catch (error) {
      console.error('Failed to load portfolios:', error);
    }
  };

  const loadRiskMetrics = async () => {
    try {
      // Mock data for demonstration
      setRiskMetrics({
        portfolioId: '1',
        asOfDate: new Date(),
        var: {
          oneDay: { confidence95: -2.1, confidence99: -3.2 },
          tenDay: { confidence95: -6.8, confidence99: -10.1 },
          twentyDay: { confidence95: -9.5, confidence99: -14.2 },
        },
        expectedShortfall: {
          oneDay: { confidence95: -2.8, confidence99: -4.1 },
          tenDay: { confidence95: -8.9, confidence99: -13.2 },
        },
        volatility: {
          realized: 18.5,
          implied: 19.2,
          forecast: 18.8,
        },
        beta: {
          market: 0.95,
          sector: 1.12,
          style: 0.88,
        },
        tracking: {
          error: 2.3,
          correlation: 0.92,
          informationRatio: 1.45,
        },
        concentration: {
          herfindahl: 0.08,
          topTenWeight: 42.5,
          effectiveNumber: 125,
        },
        liquidity: {
          averageDaysToLiquidate: 3.2,
          liquidityScore: 85,
          illiquidPercentage: 8.5,
        },
        leverage: {
          gross: 1.15,
          net: 0.98,
          adjustedGross: 1.08,
        },
        stress: {
          scenarios: [
            { name: '2008 Crisis', pnl: -18.5, probability: 0.05 },
            { name: 'COVID-19', pnl: -12.3, probability: 0.10 },
            { name: 'Rate Shock', pnl: -8.7, probability: 0.15 },
          ],
          worstCase: -25.2,
          bestCase: 15.8,
        },
      });
    } catch (error) {
      console.error('Failed to load risk metrics:', error);
    }
  };

  const loadComplianceViolations = async () => {
    try {
      // Mock data for demonstration
      setComplianceViolations([
        {
          id: '1',
          ruleId: 'rule1',
          rule: {
            id: 'rule1',
            name: 'Position Limit',
            description: 'Maximum position size exceeded',
            type: 'position_limit',
            scope: 'portfolio',
            parameters: { threshold: 5, operator: 'greater_than', value: 5 },
            severity: 'warning',
            action: 'alert',
            isActive: true,
            createdBy: 'system',
            createdAt: new Date(),
            lastModified: new Date(),
          },
          portfolioId: '1',
          clientId: 'client1',
          severity: 'warning',
          status: 'open',
          description: 'AAPL position exceeds 5% limit',
          currentValue: 6.2,
          thresholdValue: 5.0,
          deviation: 1.2,
          detectedAt: new Date(),
          impact: {
            financialImpact: 50000,
            riskImpact: 'Medium',
            clientImpact: 'Low',
          },
        },
      ]);
    } catch (error) {
      console.error('Failed to load compliance violations:', error);
    }
  };

  const loadPerformanceData = async () => {
    try {
      // Mock data for demonstration
      setPerformanceData({
        portfolioId: '1',
        benchmarkId: 'SPY',
        period: '1Y',
        totalReturn: {
          portfolio: 12.5,
          benchmark: 10.2,
          activeReturn: 2.3,
        },
        attribution: {
          allocation: {
            sector: [
              { name: 'Technology', contribution: 1.8 },
              { name: 'Healthcare', contribution: 0.5 },
              { name: 'Financials', contribution: -0.2 },
            ],
            country: [
              { name: 'US', contribution: 1.2 },
              { name: 'Europe', contribution: 0.3 },
              { name: 'Asia', contribution: 0.1 },
            ],
            assetClass: [
              { name: 'Equities', contribution: 2.1 },
              { name: 'Bonds', contribution: 0.2 },
              { name: 'Alternatives', contribution: 0.0 },
            ],
          },
          selection: {
            sector: [
              { name: 'Technology', contribution: 0.8 },
              { name: 'Healthcare', contribution: 0.3 },
              { name: 'Financials', contribution: -0.1 },
            ],
            country: [
              { name: 'US', contribution: 0.5 },
              { name: 'Europe', contribution: 0.2 },
              { name: 'Asia', contribution: 0.1 },
            ],
            security: [
              { name: 'AAPL', contribution: 0.4 },
              { name: 'MSFT', contribution: 0.3 },
              { name: 'GOOGL', contribution: 0.2 },
            ],
          },
          interaction: {
            total: 0.1,
            breakdown: [
              { factor: 'Sector-Security', contribution: 0.05 },
              { factor: 'Country-Security', contribution: 0.03 },
              { factor: 'Other', contribution: 0.02 },
            ],
          },
          currency: {
            hedged: 0.0,
            unhedged: -0.1,
            hedgingCost: -0.05,
          },
        },
        riskAdjusted: {
          sharpeRatio: { portfolio: 1.45, benchmark: 1.12 },
          informationRatio: 1.25,
          treynorRatio: { portfolio: 0.125, benchmark: 0.098 },
          jensenAlpha: 0.023,
        },
        breakdown: {
          daily: [],
          monthly: [],
        },
      });
    } catch (error) {
      console.error('Failed to load performance data:', error);
    }
  };

  const loadRecentTrades = async () => {
    try {
      // Mock data for demonstration
      setRecentTrades([
        {
          id: '1',
          symbol: 'AAPL',
          side: 'buy',
          quantity: 1000,
          price: 175.50,
          timestamp: new Date(),
          status: 'filled',
        },
        {
          id: '2',
          symbol: 'MSFT',
          side: 'sell',
          quantity: 500,
          price: 342.25,
          timestamp: new Date(),
          status: 'filled',
        },
      ]);
    } catch (error) {
      console.error('Failed to load recent trades:', error);
    }
  };

  const loadAlerts = async () => {
    try {
      // Mock data for demonstration
      setAlerts([
        {
          id: '1',
          type: 'risk',
          severity: 'high',
          message: 'Portfolio VaR exceeded threshold',
          timestamp: new Date(),
        },
        {
          id: '2',
          type: 'compliance',
          severity: 'warning',
          message: 'Position limit breach detected',
          timestamp: new Date(),
        },
      ]);
    } catch (error) {
      console.error('Failed to load alerts:', error);
    }
  };

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await loadInitialData();
    setRefreshing(false);
  }, []);

  const renderTabBar = () => (
    <View style={[styles.tabBar, { backgroundColor: colors.surface, borderBottomColor: colors.border }]}>
      {[
        { key: 'overview', label: 'Overview', icon: 'dashboard' },
        { key: 'risk', label: 'Risk', icon: 'warning' },
        { key: 'compliance', label: 'Compliance', icon: 'verified-user' },
        { key: 'performance', label: 'Performance', icon: 'trending-up' },
      ].map((tab) => (
        <TouchableOpacity
          key={tab.key}
          style={[
            styles.tabItem,
            activeTab === tab.key && { borderBottomColor: colors.primary }
          ]}
          onPress={() => setActiveTab(tab.key as any)}
        >
          <Icon 
            name={tab.icon} 
            size={20} 
            color={activeTab === tab.key ? colors.primary : colors.textSecondary} 
          />
          <Text style={[
            styles.tabLabel,
            { color: activeTab === tab.key ? colors.primary : colors.textSecondary }
          ]}>
            {tab.label}
          </Text>
        </TouchableOpacity>
      ))}
    </View>
  );

  const renderOverview = () => (
    <ScrollView 
      style={styles.content}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
    >
      {/* Portfolio Summary */}
      <View style={[styles.section, { backgroundColor: colors.surface, borderColor: colors.border }]}>
        <Text style={[styles.sectionTitle, { color: colors.text }]}>Portfolio Summary</Text>
        <View style={styles.portfolioGrid}>
          {portfolios.map((portfolio) => (
            <View key={portfolio.id} style={[styles.portfolioCard, { backgroundColor: colors.background, borderColor: colors.border }]}>
              <Text style={[styles.portfolioName, { color: colors.text }]}>{portfolio.name}</Text>
              <Text style={[styles.portfolioAUM, { color: colors.textSecondary }]}>
                ${(portfolio.aum / 1000000).toFixed(0)}M AUM
              </Text>
              <Text style={[styles.portfolioReturn, { color: portfolio.return >= 0 ? colors.success : colors.danger }]}>
                {portfolio.return >= 0 ? '+' : ''}{portfolio.return.toFixed(1)}%
              </Text>
            </View>
          ))}
        </View>
      </View>

      {/* Key Metrics */}
      <View style={[styles.section, { backgroundColor: colors.surface, borderColor: colors.border }]}>
        <Text style={[styles.sectionTitle, { color: colors.text }]}>Key Metrics</Text>
        <View style={styles.metricsGrid}>
          <View style={styles.metricItem}>
            <Text style={[styles.metricValue, { color: colors.text }]}>$430M</Text>
            <Text style={[styles.metricLabel, { color: colors.textSecondary }]}>Total AUM</Text>
          </View>
          <View style={styles.metricItem}>
            <Text style={[styles.metricValue, { color: colors.success }]}>+9.8%</Text>
            <Text style={[styles.metricLabel, { color: colors.textSecondary }]}>YTD Return</Text>
          </View>
          <View style={styles.metricItem}>
            <Text style={[styles.metricValue, { color: colors.text }]}>1.45</Text>
            <Text style={[styles.metricLabel, { color: colors.textSecondary }]}>Sharpe Ratio</Text>
          </View>
          <View style={styles.metricItem}>
            <Text style={[styles.metricValue, { color: colors.warning }]}>-2.1%</Text>
            <Text style={[styles.metricLabel, { color: colors.textSecondary }]}>1D VaR (95%)</Text>
          </View>
        </View>
      </View>

      {/* Recent Alerts */}
      <View style={[styles.section, { backgroundColor: colors.surface, borderColor: colors.border }]}>
        <Text style={[styles.sectionTitle, { color: colors.text }]}>Recent Alerts</Text>
        {alerts.slice(0, 3).map((alert) => (
          <View key={alert.id} style={[styles.alertItem, { borderLeftColor: getSeverityColor(alert.severity) }]}>
            <View style={styles.alertHeader}>
              <Icon 
                name={alert.type === 'risk' ? 'warning' : 'verified-user'} 
                size={16} 
                color={getSeverityColor(alert.severity)} 
              />
              <Text style={[styles.alertType, { color: colors.textSecondary }]}>
                {alert.type.toUpperCase()}
              </Text>
              <Text style={[styles.alertTime, { color: colors.textSecondary }]}>
                {alert.timestamp.toLocaleTimeString()}
              </Text>
            </View>
            <Text style={[styles.alertMessage, { color: colors.text }]}>
              {alert.message}
            </Text>
          </View>
        ))}
      </View>

      {/* Recent Trades */}
      <View style={[styles.section, { backgroundColor: colors.surface, borderColor: colors.border }]}>
        <Text style={[styles.sectionTitle, { color: colors.text }]}>Recent Trades</Text>
        {recentTrades.map((trade) => (
          <View key={trade.id} style={styles.tradeItem}>
            <View style={styles.tradeInfo}>
              <Text style={[styles.tradeSymbol, { color: colors.text }]}>{trade.symbol}</Text>
              <Text style={[
                styles.tradeSide,
                { color: trade.side === 'buy' ? colors.success : colors.danger }
              ]}>
                {trade.side.toUpperCase()}
              </Text>
            </View>
            <View style={styles.tradeDetails}>
              <Text style={[styles.tradeQuantity, { color: colors.textSecondary }]}>
                {trade.quantity.toLocaleString()} @ ${trade.price.toFixed(2)}
              </Text>
              <Text style={[styles.tradeStatus, { color: colors.success }]}>
                {trade.status.toUpperCase()}
              </Text>
            </View>
          </View>
        ))}
      </View>
    </ScrollView>
  );

  const renderRisk = () => (
    <ScrollView 
      style={styles.content}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
    >
      {riskMetrics && (
        <>
          {/* VaR Metrics */}
          <View style={[styles.section, { backgroundColor: colors.surface, borderColor: colors.border }]}>
            <Text style={[styles.sectionTitle, { color: colors.text }]}>Value at Risk</Text>
            <View style={styles.varGrid}>
              <View style={styles.varItem}>
                <Text style={[styles.varPeriod, { color: colors.textSecondary }]}>1 Day</Text>
                <Text style={[styles.varValue, { color: colors.danger }]}>
                  {riskMetrics.var.oneDay.confidence95.toFixed(1)}%
                </Text>
                <Text style={[styles.varConfidence, { color: colors.textSecondary }]}>95% Conf</Text>
              </View>
              <View style={styles.varItem}>
                <Text style={[styles.varPeriod, { color: colors.textSecondary }]}>10 Day</Text>
                <Text style={[styles.varValue, { color: colors.danger }]}>
                  {riskMetrics.var.tenDay.confidence95.toFixed(1)}%
                </Text>
                <Text style={[styles.varConfidence, { color: colors.textSecondary }]}>95% Conf</Text>
              </View>
              <View style={styles.varItem}>
                <Text style={[styles.varPeriod, { color: colors.textSecondary }]}>20 Day</Text>
                <Text style={[styles.varValue, { color: colors.danger }]}>
                  {riskMetrics.var.twentyDay.confidence95.toFixed(1)}%
                </Text>
                <Text style={[styles.varConfidence, { color: colors.textSecondary }]}>95% Conf</Text>
              </View>
            </View>
          </View>

          {/* Risk Factors */}
          <View style={[styles.section, { backgroundColor: colors.surface, borderColor: colors.border }]}>
            <Text style={[styles.sectionTitle, { color: colors.text }]}>Risk Factors</Text>
            <View style={styles.riskFactors}>
              <View style={styles.riskFactor}>
                <Text style={[styles.riskFactorLabel, { color: colors.textSecondary }]}>Market Beta</Text>
                <Text style={[styles.riskFactorValue, { color: colors.text }]}>
                  {riskMetrics.beta.market.toFixed(2)}
                </Text>
              </View>
              <View style={styles.riskFactor}>
                <Text style={[styles.riskFactorLabel, { color: colors.textSecondary }]}>Volatility</Text>
                <Text style={[styles.riskFactorValue, { color: colors.text }]}>
                  {riskMetrics.volatility.realized.toFixed(1)}%
                </Text>
              </View>
              <View style={styles.riskFactor}>
                <Text style={[styles.riskFactorLabel, { color: colors.textSecondary }]}>Tracking Error</Text>
                <Text style={[styles.riskFactorValue, { color: colors.text }]}>
                  {riskMetrics.tracking.error.toFixed(1)}%
                </Text>
              </View>
              <View style={styles.riskFactor}>
                <Text style={[styles.riskFactorLabel, { color: colors.textSecondary }]}>Concentration</Text>
                <Text style={[styles.riskFactorValue, { color: colors.text }]}>
                  {riskMetrics.concentration.topTenWeight.toFixed(1)}%
                </Text>
              </View>
            </View>
          </View>

          {/* Stress Test */}
          <View style={[styles.section, { backgroundColor: colors.surface, borderColor: colors.border }]}>
            <Text style={[styles.sectionTitle, { color: colors.text }]}>Stress Test Scenarios</Text>
            {riskMetrics.stress.scenarios.map((scenario, index) => (
              <View key={index} style={styles.stressScenario}>
                <View style={styles.stressHeader}>
                  <Text style={[styles.stressName, { color: colors.text }]}>{scenario.name}</Text>
                  <Text style={[styles.stressProbability, { color: colors.textSecondary }]}>
                    {(scenario.probability * 100).toFixed(0)}% prob
                  </Text>
                </View>
                <Text style={[styles.stressPnL, { color: colors.danger }]}>
                  {scenario.pnl.toFixed(1)}%
                </Text>
              </View>
            ))}
          </View>
        </>
      )}
    </ScrollView>
  );

  const renderCompliance = () => (
    <ScrollView 
      style={styles.content}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
    >
      {/* Compliance Status */}
      <View style={[styles.section, { backgroundColor: colors.surface, borderColor: colors.border }]}>
        <Text style={[styles.sectionTitle, { color: colors.text }]}>Compliance Status</Text>
        <View style={styles.complianceStatus}>
          <View style={styles.complianceItem}>
            <Text style={[styles.complianceValue, { color: colors.success }]}>98.5%</Text>
            <Text style={[styles.complianceLabel, { color: colors.textSecondary }]}>Overall Score</Text>
          </View>
          <View style={styles.complianceItem}>
            <Text style={[styles.complianceValue, { color: colors.warning }]}>3</Text>
            <Text style={[styles.complianceLabel, { color: colors.textSecondary }]}>Open Issues</Text>
          </View>
          <View style={styles.complianceItem}>
            <Text style={[styles.complianceValue, { color: colors.text }]}>15</Text>
            <Text style={[styles.complianceLabel, { color: colors.textSecondary }]}>Rules Active</Text>
          </View>
        </View>
      </View>

      {/* Violations */}
      <View style={[styles.section, { backgroundColor: colors.surface, borderColor: colors.border }]}>
        <Text style={[styles.sectionTitle, { color: colors.text }]}>Active Violations</Text>
        {complianceViolations.map((violation) => (
          <View key={violation.id} style={[styles.violationItem, { borderLeftColor: getSeverityColor(violation.severity) }]}>
            <View style={styles.violationHeader}>
              <Text style={[styles.violationRule, { color: colors.text }]}>
                {violation.rule.name}
              </Text>
              <View style={[styles.severityBadge, { backgroundColor: getSeverityColor(violation.severity) }]}>
                <Text style={styles.severityText}>{violation.severity.toUpperCase()}</Text>
              </View>
            </View>
            <Text style={[styles.violationDescription, { color: colors.textSecondary }]}>
              {violation.description}
            </Text>
            <View style={styles.violationDetails}>
              <Text style={[styles.violationValue, { color: colors.text }]}>
                Current: {violation.currentValue.toFixed(1)}%
              </Text>
              <Text style={[styles.violationThreshold, { color: colors.textSecondary }]}>
                Threshold: {violation.thresholdValue.toFixed(1)}%
              </Text>
            </View>
          </View>
        ))}
      </View>
    </ScrollView>
  );

  const renderPerformance = () => (
    <ScrollView 
      style={styles.content}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
    >
      {performanceData && (
        <>
          {/* Performance Summary */}
          <View style={[styles.section, { backgroundColor: colors.surface, borderColor: colors.border }]}>
            <Text style={[styles.sectionTitle, { color: colors.text }]}>Performance Summary</Text>
            <View style={styles.performanceGrid}>
              <View style={styles.performanceItem}>
                <Text style={[styles.performanceLabel, { color: colors.textSecondary }]}>Portfolio</Text>
                <Text style={[styles.performanceValue, { color: colors.success }]}>
                  +{performanceData.totalReturn.portfolio.toFixed(1)}%
                </Text>
              </View>
              <View style={styles.performanceItem}>
                <Text style={[styles.performanceLabel, { color: colors.textSecondary }]}>Benchmark</Text>
                <Text style={[styles.performanceValue, { color: colors.text }]}>
                  +{performanceData.totalReturn.benchmark.toFixed(1)}%
                </Text>
              </View>
              <View style={styles.performanceItem}>
                <Text style={[styles.performanceLabel, { color: colors.textSecondary }]}>Active Return</Text>
                <Text style={[styles.performanceValue, { color: colors.success }]}>
                  +{performanceData.totalReturn.activeReturn.toFixed(1)}%
                </Text>
              </View>
            </View>
          </View>

          {/* Attribution Analysis */}
          <View style={[styles.section, { backgroundColor: colors.surface, borderColor: colors.border }]}>
            <Text style={[styles.sectionTitle, { color: colors.text }]}>Attribution Analysis</Text>
            <Text style={[styles.subsectionTitle, { color: colors.textSecondary }]}>Sector Allocation</Text>
            {performanceData.attribution.allocation.sector.map((item, index) => (
              <View key={index} style={styles.attributionItem}>
                <Text style={[styles.attributionName, { color: colors.text }]}>{item.name}</Text>
                <Text style={[
                  styles.attributionValue,
                  { color: item.contribution >= 0 ? colors.success : colors.danger }
                ]}>
                  {item.contribution >= 0 ? '+' : ''}{item.contribution.toFixed(2)}%
                </Text>
              </View>
            ))}
          </View>

          {/* Risk-Adjusted Metrics */}
          <View style={[styles.section, { backgroundColor: colors.surface, borderColor: colors.border }]}>
            <Text style={[styles.sectionTitle, { color: colors.text }]}>Risk-Adjusted Metrics</Text>
            <View style={styles.riskAdjustedGrid}>
              <View style={styles.riskAdjustedItem}>
                <Text style={[styles.riskAdjustedLabel, { color: colors.textSecondary }]}>Sharpe Ratio</Text>
                <Text style={[styles.riskAdjustedValue, { color: colors.text }]}>
                  {performanceData.riskAdjusted.sharpeRatio.portfolio.toFixed(2)}
                </Text>
              </View>
              <View style={styles.riskAdjustedItem}>
                <Text style={[styles.riskAdjustedLabel, { color: colors.textSecondary }]}>Information Ratio</Text>
                <Text style={[styles.riskAdjustedValue, { color: colors.text }]}>
                  {performanceData.riskAdjusted.informationRatio.toFixed(2)}
                </Text>
              </View>
              <View style={styles.riskAdjustedItem}>
                <Text style={[styles.riskAdjustedLabel, { color: colors.textSecondary }]}>Jensen Alpha</Text>
                <Text style={[styles.riskAdjustedValue, { color: colors.success }]}>
                  +{(performanceData.riskAdjusted.jensenAlpha * 100).toFixed(1)}%
                </Text>
              </View>
            </View>
          </View>
        </>
      )}
    </ScrollView>
  );

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return colors.danger;
      case 'high': return colors.danger;
      case 'warning': return colors.warning;
      case 'medium': return colors.warning;
      case 'low': return colors.success;
      case 'info': return colors.primary;
      default: return colors.textSecondary;
    }
  };

  const renderContent = () => {
    switch (activeTab) {
      case 'overview': return renderOverview();
      case 'risk': return renderRisk();
      case 'compliance': return renderCompliance();
      case 'performance': return renderPerformance();
      default: return null;
    }
  };

  return (
    <View style={[styles.container, { backgroundColor: colors.background }]}>
      {renderTabBar()}
      {renderContent()}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  tabBar: {
    flexDirection: 'row',
    borderBottomWidth: 1,
    paddingTop: 10,
  },
  tabItem: {
    flex: 1,
    alignItems: 'center',
    paddingVertical: 12,
    borderBottomWidth: 2,
    borderBottomColor: 'transparent',
  },
  tabLabel: {
    fontSize: 12,
    marginTop: 4,
    fontWeight: '500',
  },
  content: {
    flex: 1,
  },
  section: {
    margin: 16,
    padding: 16,
    borderRadius: 12,
    borderWidth: 1,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 16,
  },
  subsectionTitle: {
    fontSize: 14,
    fontWeight: '500',
    marginBottom: 8,
    marginTop: 8,
  },
  portfolioGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  portfolioCard: {
    flex: 1,
    minWidth: '45%',
    padding: 12,
    borderRadius: 8,
    borderWidth: 1,
  },
  portfolioName: {
    fontSize: 14,
    fontWeight: '600',
    marginBottom: 4,
  },
  portfolioAUM: {
    fontSize: 12,
    marginBottom: 8,
  },
  portfolioReturn: {
    fontSize: 16,
    fontWeight: '600',
  },
  metricsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 16,
  },
  metricItem: {
    flex: 1,
    minWidth: '45%',
    alignItems: 'center',
  },
  metricValue: {
    fontSize: 20,
    fontWeight: '600',
    marginBottom: 4,
  },
  metricLabel: {
    fontSize: 12,
    textAlign: 'center',
  },
  alertItem: {
    padding: 12,
    borderLeftWidth: 4,
    marginBottom: 8,
  },
  alertHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 4,
  },
  alertType: {
    fontSize: 10,
    fontWeight: '600',
    marginLeft: 6,
    flex: 1,
  },
  alertTime: {
    fontSize: 10,
  },
  alertMessage: {
    fontSize: 14,
  },
  tradeItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  tradeInfo: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  tradeSymbol: {
    fontSize: 16,
    fontWeight: '600',
    marginRight: 8,
  },
  tradeSide: {
    fontSize: 12,
    fontWeight: '600',
  },
  tradeDetails: {
    alignItems: 'flex-end',
  },
  tradeQuantity: {
    fontSize: 12,
  },
  tradeStatus: {
    fontSize: 10,
    fontWeight: '600',
  },
  varGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  varItem: {
    alignItems: 'center',
  },
  varPeriod: {
    fontSize: 12,
    marginBottom: 4,
  },
  varValue: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 2,
  },
  varConfidence: {
    fontSize: 10,
  },
  riskFactors: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 16,
  },
  riskFactor: {
    flex: 1,
    minWidth: '45%',
    alignItems: 'center',
  },
  riskFactorLabel: {
    fontSize: 12,
    marginBottom: 4,
  },
  riskFactorValue: {
    fontSize: 16,
    fontWeight: '600',
  },
  stressScenario: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 8,
  },
  stressHeader: {
    flex: 1,
  },
  stressName: {
    fontSize: 14,
    fontWeight: '500',
  },
  stressProbability: {
    fontSize: 12,
  },
  stressPnL: {
    fontSize: 16,
    fontWeight: '600',
  },
  complianceStatus: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  complianceItem: {
    alignItems: 'center',
  },
  complianceValue: {
    fontSize: 20,
    fontWeight: '600',
    marginBottom: 4,
  },
  complianceLabel: {
    fontSize: 12,
  },
  violationItem: {
    padding: 12,
    borderLeftWidth: 4,
    marginBottom: 12,
  },
  violationHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  },
  violationRule: {
    fontSize: 16,
    fontWeight: '600',
    flex: 1,
  },
  severityBadge: {
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 10,
  },
  severityText: {
    fontSize: 10,
    fontWeight: '600',
    color: '#ffffff',
  },
  violationDescription: {
    fontSize: 14,
    marginBottom: 8,
  },
  violationDetails: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  violationValue: {
    fontSize: 12,
    fontWeight: '500',
  },
  violationThreshold: {
    fontSize: 12,
  },
  performanceGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  performanceItem: {
    alignItems: 'center',
  },
  performanceLabel: {
    fontSize: 12,
    marginBottom: 4,
  },
  performanceValue: {
    fontSize: 18,
    fontWeight: '600',
  },
  attributionItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 4,
  },
  attributionName: {
    fontSize: 14,
    flex: 1,
  },
  attributionValue: {
    fontSize: 14,
    fontWeight: '600',
  },
  riskAdjustedGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  riskAdjustedItem: {
    alignItems: 'center',
  },
  riskAdjustedLabel: {
    fontSize: 12,
    marginBottom: 4,
    textAlign: 'center',
  },
  riskAdjustedValue: {
    fontSize: 16,
    fontWeight: '600',
  },
});

export default InstitutionalDashboard;