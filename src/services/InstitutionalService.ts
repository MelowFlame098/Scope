import axios from 'axios';
import { io, Socket } from 'socket.io-client';

// Types for institutional features
export interface InstitutionalClient {
  id: string;
  name: string;
  type: 'hedge_fund' | 'asset_manager' | 'pension_fund' | 'insurance' | 'bank' | 'family_office' | 'endowment' | 'sovereign_wealth';
  aum: number; // Assets Under Management
  jurisdiction: string;
  regulatoryStatus: string[];
  riskProfile: 'conservative' | 'moderate' | 'aggressive' | 'custom';
  investmentObjectives: string[];
  benchmarks: string[];
  reportingCurrency: string;
  contactInfo: {
    primaryContact: string;
    email: string;
    phone: string;
    address: string;
  };
  complianceRequirements: string[];
  createdAt: Date;
  lastReviewDate: Date;
}

export interface PortfolioMandate {
  id: string;
  clientId: string;
  name: string;
  description: string;
  investmentUniverse: string[];
  assetAllocationLimits: {
    assetClass: string;
    minWeight: number;
    maxWeight: number;
    targetWeight: number;
  }[];
  riskLimits: {
    maxVaR: number;
    maxDrawdown: number;
    maxConcentration: number;
    maxLeverage: number;
    maxVolatility: number;
  };
  performanceTargets: {
    benchmark: string;
    targetReturn: number;
    trackingError: number;
    informationRatio: number;
  };
  constraints: {
    esgRequirements?: string[];
    excludedSectors?: string[];
    excludedCountries?: string[];
    liquidityRequirements?: string;
    currencyHedging?: boolean;
  };
  rebalancingFrequency: 'daily' | 'weekly' | 'monthly' | 'quarterly';
  isActive: boolean;
  createdAt: Date;
  updatedAt: Date;
}

export interface RiskModel {
  id: string;
  name: string;
  type: 'factor' | 'statistical' | 'fundamental' | 'hybrid';
  description: string;
  factors: {
    id: string;
    name: string;
    type: 'market' | 'style' | 'industry' | 'country' | 'currency' | 'macro';
    weight: number;
    exposure: number;
  }[];
  correlationMatrix: number[][];
  volatilityModel: {
    type: 'garch' | 'ewma' | 'historical';
    parameters: any;
  };
  backtestResults: {
    period: string;
    accuracy: number;
    coverage: number;
    sharpeRatio: number;
  };
  lastUpdated: Date;
  isActive: boolean;
}

export interface RiskMetrics {
  portfolioId: string;
  asOfDate: Date;
  var: {
    oneDay: { confidence95: number; confidence99: number };
    tenDay: { confidence95: number; confidence99: number };
    twentyDay: { confidence95: number; confidence99: number };
  };
  expectedShortfall: {
    oneDay: { confidence95: number; confidence99: number };
    tenDay: { confidence95: number; confidence99: number };
  };
  volatility: {
    realized: number;
    implied: number;
    forecast: number;
  };
  beta: {
    market: number;
    sector: number;
    style: number;
  };
  tracking: {
    error: number;
    correlation: number;
    informationRatio: number;
  };
  concentration: {
    herfindahl: number;
    topTenWeight: number;
    effectiveNumber: number;
  };
  liquidity: {
    averageDaysToLiquidate: number;
    liquidityScore: number;
    illiquidPercentage: number;
  };
  leverage: {
    gross: number;
    net: number;
    adjustedGross: number;
  };
  stress: {
    scenarios: {
      name: string;
      pnl: number;
      probability: number;
    }[];
    worstCase: number;
    bestCase: number;
  };
}

export interface ComplianceRule {
  id: string;
  name: string;
  description: string;
  type: 'position_limit' | 'concentration' | 'liquidity' | 'esg' | 'regulatory' | 'custom';
  scope: 'portfolio' | 'mandate' | 'client' | 'firm';
  parameters: {
    threshold: number;
    operator: 'greater_than' | 'less_than' | 'equal_to' | 'between';
    value: number | number[];
    currency?: string;
    assetClass?: string;
    region?: string;
  };
  severity: 'info' | 'warning' | 'error' | 'critical';
  action: 'alert' | 'block' | 'auto_rebalance';
  isActive: boolean;
  createdBy: string;
  createdAt: Date;
  lastModified: Date;
}

export interface ComplianceViolation {
  id: string;
  ruleId: string;
  rule: ComplianceRule;
  portfolioId: string;
  clientId: string;
  severity: 'info' | 'warning' | 'error' | 'critical';
  status: 'open' | 'acknowledged' | 'resolved' | 'waived';
  description: string;
  currentValue: number;
  thresholdValue: number;
  deviation: number;
  detectedAt: Date;
  resolvedAt?: Date;
  resolvedBy?: string;
  resolution?: string;
  waiverReason?: string;
  impact: {
    financialImpact: number;
    riskImpact: string;
    clientImpact: string;
  };
}

export interface PerformanceAttribution {
  portfolioId: string;
  benchmarkId: string;
  period: string;
  totalReturn: {
    portfolio: number;
    benchmark: number;
    activeReturn: number;
  };
  attribution: {
    allocation: {
      sector: { name: string; contribution: number }[];
      country: { name: string; contribution: number }[];
      assetClass: { name: string; contribution: number }[];
    };
    selection: {
      sector: { name: string; contribution: number }[];
      country: { name: string; contribution: number }[];
      security: { name: string; contribution: number }[];
    };
    interaction: {
      total: number;
      breakdown: { factor: string; contribution: number }[];
    };
    currency: {
      hedged: number;
      unhedged: number;
      hedgingCost: number;
    };
  };
  riskAdjusted: {
    sharpeRatio: { portfolio: number; benchmark: number };
    informationRatio: number;
    treynorRatio: { portfolio: number; benchmark: number };
    jensenAlpha: number;
  };
  breakdown: {
    daily: { date: Date; portfolioReturn: number; benchmarkReturn: number; activeReturn: number }[];
    monthly: { month: string; portfolioReturn: number; benchmarkReturn: number; activeReturn: number }[];
  };
}

export interface ESGMetrics {
  portfolioId: string;
  asOfDate: Date;
  overallScore: {
    esg: number;
    environmental: number;
    social: number;
    governance: number;
  };
  breakdown: {
    holdings: {
      symbol: string;
      weight: number;
      esgScore: number;
      controversyScore: number;
      carbonIntensity: number;
    }[];
  };
  benchmarkComparison: {
    benchmark: string;
    portfolioScore: number;
    benchmarkScore: number;
    relativeDifference: number;
  }[];
  carbonFootprint: {
    totalEmissions: number;
    carbonIntensity: number;
    scope1: number;
    scope2: number;
    scope3: number;
  };
  alignment: {
    parisAgreement: {
      aligned: boolean;
      temperature: number;
      score: number;
    };
    sdg: {
      goal: string;
      score: number;
      contribution: number;
    }[];
  };
  exclusions: {
    tobacco: number;
    weapons: number;
    gambling: number;
    adultEntertainment: number;
    fossilFuels: number;
  };
}

export interface TradeOrder {
  id: string;
  portfolioId: string;
  clientId: string;
  symbol: string;
  side: 'buy' | 'sell';
  orderType: 'market' | 'limit' | 'stop' | 'stop_limit' | 'iceberg' | 'twap' | 'vwap';
  quantity: number;
  price?: number;
  stopPrice?: number;
  timeInForce: 'day' | 'gtc' | 'ioc' | 'fok';
  executionStrategy?: {
    algorithm: string;
    parameters: any;
    startTime?: Date;
    endTime?: Date;
  };
  status: 'pending' | 'working' | 'partially_filled' | 'filled' | 'cancelled' | 'rejected';
  fills: {
    id: string;
    quantity: number;
    price: number;
    timestamp: Date;
    venue: string;
    commission: number;
  }[];
  createdBy: string;
  createdAt: Date;
  updatedAt: Date;
  compliance: {
    preTradeChecks: { rule: string; status: 'pass' | 'fail'; message?: string }[];
    postTradeChecks: { rule: string; status: 'pass' | 'fail'; message?: string }[];
  };
}

export interface StressTest {
  id: string;
  name: string;
  description: string;
  portfolioId: string;
  scenario: {
    equityShock: number;
    bondShock: number;
    creditSpreadShock: number;
    volatilityShock: number;
    correlationShock: number;
  };
  results: {
    portfolioReturn: number;
    worstAssetReturn: number;
    timeToRecover: number;
    maxDrawdown: number;
    liquidityImpact: number;
  };
  runDate: Date;
  status: 'pending' | 'running' | 'completed' | 'failed';
  confidence: number;
}

export interface VaRCalculation {
  id: string;
  portfolioId: string;
  confidence: number;
  timeHorizon: number;
  value: number;
  method: 'historical' | 'parametric' | 'monte_carlo';
  calculatedAt: Date;
}

export interface ESGAnalytics {
  id: string;
  portfolioId: string;
  overallScore: number;
  environmentalScore: number;
  socialScore: number;
  governanceScore: number;
  calculatedAt: Date;
}

export interface RebalancingRecommendation {
  id: string;
  portfolioId: string;
  reason: string;
  recommendations: {
    symbol: string;
    currentWeight: number;
    targetWeight: number;
    action: 'buy' | 'sell' | 'hold';
  }[];
  expectedImpact: {
    riskReduction: number;
    costEstimate: number;
  };
  createdAt: Date;
}

export interface RebalancingPlan {
  id: string;
  portfolioId: string;
  mandateId: string;
  type: 'strategic' | 'tactical' | 'risk_driven' | 'compliance_driven';
  reason: string;
  targetAllocations: {
    assetClass: string;
    currentWeight: number;
    targetWeight: number;
    difference: number;
  }[];
  trades: {
    symbol: string;
    side: 'buy' | 'sell';
    quantity: number;
    estimatedPrice: number;
    estimatedCost: number;
    priority: number;
  }[];
  estimatedCosts: {
    totalCommissions: number;
    marketImpact: number;
    opportunityCost: number;
    totalCost: number;
  };
  riskImpact: {
    currentRisk: number;
    projectedRisk: number;
    riskReduction: number;
  };
  status: 'draft' | 'approved' | 'executing' | 'completed' | 'cancelled';
  approvedBy?: string;
  approvedAt?: Date;
  executedAt?: Date;
  createdAt: Date;
}

class InstitutionalService {
  private baseURL: string;
  private socket: Socket | null = null;
  private apiKey: string;

  constructor() {
    this.baseURL = process.env.REACT_APP_INSTITUTIONAL_API_URL || 'http://localhost:8003';
    this.apiKey = process.env.REACT_APP_API_KEY || '';
    this.initializeWebSocket();
  }

  private initializeWebSocket() {
    this.socket = io(`${this.baseURL}/institutional`, {
      auth: {
        token: this.apiKey,
      },
      transports: ['websocket'],
    });

    this.socket.on('connect', () => {
      console.log('Connected to institutional service');
    });

    this.socket.on('compliance_alert', (data: ComplianceViolation) => {
      this.handleComplianceAlert(data);
    });

    this.socket.on('risk_alert', (data: any) => {
      this.handleRiskAlert(data);
    });

    this.socket.on('trade_update', (data: TradeOrder) => {
      this.handleTradeUpdate(data);
    });
  }

  // Client Management
  async getClients(): Promise<InstitutionalClient[]> {
    try {
      const response = await axios.get(`${this.baseURL}/api/clients`, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get clients:', error);
      throw error;
    }
  }

  async getClient(clientId: string): Promise<InstitutionalClient> {
    try {
      const response = await axios.get(`${this.baseURL}/api/clients/${clientId}`, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get client:', error);
      throw error;
    }
  }

  async createClient(client: Omit<InstitutionalClient, 'id' | 'createdAt' | 'lastReviewDate'>): Promise<InstitutionalClient> {
    try {
      const response = await axios.post(`${this.baseURL}/api/clients`, client, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to create client:', error);
      throw error;
    }
  }

  async updateClient(clientId: string, updates: Partial<InstitutionalClient>): Promise<InstitutionalClient> {
    try {
      const response = await axios.put(`${this.baseURL}/api/clients/${clientId}`, updates, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to update client:', error);
      throw error;
    }
  }

  // Portfolio Mandates
  async getMandates(clientId?: string): Promise<PortfolioMandate[]> {
    try {
      const response = await axios.get(`${this.baseURL}/api/mandates`, {
        params: { clientId },
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get mandates:', error);
      throw error;
    }
  }

  async getMandate(mandateId: string): Promise<PortfolioMandate> {
    try {
      const response = await axios.get(`${this.baseURL}/api/mandates/${mandateId}`, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get mandate:', error);
      throw error;
    }
  }

  async createMandate(mandate: Omit<PortfolioMandate, 'id' | 'createdAt' | 'updatedAt'>): Promise<PortfolioMandate> {
    try {
      const response = await axios.post(`${this.baseURL}/api/mandates`, mandate, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to create mandate:', error);
      throw error;
    }
  }

  async updateMandate(mandateId: string, updates: Partial<PortfolioMandate>): Promise<PortfolioMandate> {
    try {
      const response = await axios.put(`${this.baseURL}/api/mandates/${mandateId}`, updates, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to update mandate:', error);
      throw error;
    }
  }

  // Risk Management
  async getRiskModels(): Promise<RiskModel[]> {
    try {
      const response = await axios.get(`${this.baseURL}/api/risk/models`, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get risk models:', error);
      throw error;
    }
  }

  async getRiskMetrics(portfolioId: string, date?: Date): Promise<RiskMetrics> {
    try {
      const response = await axios.get(`${this.baseURL}/api/risk/metrics/${portfolioId}`, {
        params: { date: date?.toISOString() },
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get risk metrics:', error);
      throw error;
    }
  }

  async calculateStressTest(portfolioId: string, scenarios: any[]): Promise<any> {
    try {
      const response = await axios.post(`${this.baseURL}/api/risk/stress-test/${portfolioId}`, {
        scenarios,
      }, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to calculate stress test:', error);
      throw error;
    }
  }

  async runVaRCalculation(portfolioId: string, confidence: number = 0.95, horizon: number = 1): Promise<any> {
    try {
      const response = await axios.post(`${this.baseURL}/api/risk/var/${portfolioId}`, {
        confidence,
        horizon,
      }, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to run VaR calculation:', error);
      throw error;
    }
  }

  // Compliance
  async getComplianceRules(scope?: string): Promise<ComplianceRule[]> {
    try {
      const response = await axios.get(`${this.baseURL}/api/compliance/rules`, {
        params: { scope },
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get compliance rules:', error);
      throw error;
    }
  }

  async createComplianceRule(rule: Omit<ComplianceRule, 'id' | 'createdAt' | 'lastModified'>): Promise<ComplianceRule> {
    try {
      const response = await axios.post(`${this.baseURL}/api/compliance/rules`, rule, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to create compliance rule:', error);
      throw error;
    }
  }

  async getComplianceViolations(filters?: any): Promise<ComplianceViolation[]> {
    try {
      const response = await axios.get(`${this.baseURL}/api/compliance/violations`, {
        params: filters,
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get compliance violations:', error);
      throw error;
    }
  }

  async resolveViolation(violationId: string, resolution: string): Promise<ComplianceViolation> {
    try {
      const response = await axios.put(`${this.baseURL}/api/compliance/violations/${violationId}/resolve`, {
        resolution,
      }, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to resolve violation:', error);
      throw error;
    }
  }

  async runComplianceCheck(portfolioId: string): Promise<ComplianceViolation[]> {
    try {
      const response = await axios.post(`${this.baseURL}/api/compliance/check/${portfolioId}`, {}, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to run compliance check:', error);
      throw error;
    }
  }

  // Performance Attribution
  async getPerformanceAttribution(portfolioId: string, benchmarkId: string, period: string): Promise<PerformanceAttribution> {
    try {
      const response = await axios.get(`${this.baseURL}/api/performance/attribution/${portfolioId}`, {
        params: { benchmarkId, period },
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get performance attribution:', error);
      throw error;
    }
  }

  async generatePerformanceReport(portfolioId: string, options: any): Promise<any> {
    try {
      const response = await axios.post(`${this.baseURL}/api/performance/report/${portfolioId}`, options, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
        responseType: 'blob',
      });
      return response.data;
    } catch (error) {
      console.error('Failed to generate performance report:', error);
      throw error;
    }
  }

  // ESG Analytics
  async getESGMetrics(portfolioId: string, date?: Date): Promise<ESGMetrics> {
    try {
      const response = await axios.get(`${this.baseURL}/api/esg/metrics/${portfolioId}`, {
        params: { date: date?.toISOString() },
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get ESG metrics:', error);
      throw error;
    }
  }

  async runESGScreening(portfolioId: string, criteria: any): Promise<any> {
    try {
      const response = await axios.post(`${this.baseURL}/api/esg/screening/${portfolioId}`, criteria, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to run ESG screening:', error);
      throw error;
    }
  }

  // Trade Management
  async createOrder(order: Omit<TradeOrder, 'id' | 'status' | 'fills' | 'createdAt' | 'updatedAt' | 'compliance'>): Promise<TradeOrder> {
    try {
      const response = await axios.post(`${this.baseURL}/api/trading/orders`, order, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to create order:', error);
      throw error;
    }
  }

  async getOrders(filters?: any): Promise<TradeOrder[]> {
    try {
      const response = await axios.get(`${this.baseURL}/api/trading/orders`, {
        params: filters,
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get orders:', error);
      throw error;
    }
  }

  async cancelOrder(orderId: string): Promise<TradeOrder> {
    try {
      const response = await axios.delete(`${this.baseURL}/api/trading/orders/${orderId}`, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to cancel order:', error);
      throw error;
    }
  }

  // Rebalancing
  async createRebalancingPlan(plan: Omit<RebalancingPlan, 'id' | 'status' | 'createdAt'>): Promise<RebalancingPlan> {
    try {
      const response = await axios.post(`${this.baseURL}/api/rebalancing/plans`, plan, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to create rebalancing plan:', error);
      throw error;
    }
  }

  async getRebalancingPlans(portfolioId?: string): Promise<RebalancingPlan[]> {
    try {
      const response = await axios.get(`${this.baseURL}/api/rebalancing/plans`, {
        params: { portfolioId },
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get rebalancing plans:', error);
      throw error;
    }
  }

  async approveRebalancingPlan(planId: string): Promise<RebalancingPlan> {
    try {
      const response = await axios.put(`${this.baseURL}/api/rebalancing/plans/${planId}/approve`, {}, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to approve rebalancing plan:', error);
      throw error;
    }
  }

  async executeRebalancingPlan(planId: string): Promise<RebalancingPlan> {
    try {
      const response = await axios.put(`${this.baseURL}/api/rebalancing/plans/${planId}/execute`, {}, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to execute rebalancing plan:', error);
      throw error;
    }
  }

  // Real-time subscriptions
  subscribeToComplianceAlerts(callback: (violation: ComplianceViolation) => void) {
    if (this.socket) {
      this.socket.emit('subscribe_compliance');
      this.socket.on('compliance_alert', callback);
    }
  }

  subscribeToRiskAlerts(callback: (alert: any) => void) {
    if (this.socket) {
      this.socket.emit('subscribe_risk');
      this.socket.on('risk_alert', callback);
    }
  }

  subscribeToTradeUpdates(callback: (order: TradeOrder) => void) {
    if (this.socket) {
      this.socket.emit('subscribe_trades');
      this.socket.on('trade_update', callback);
    }
  }

  // Event handlers
  private handleComplianceAlert(violation: ComplianceViolation) {
    console.log('Compliance alert:', violation);
  }

  private handleRiskAlert(alert: any) {
    console.log('Risk alert:', alert);
  }

  private handleTradeUpdate(order: TradeOrder) {
    console.log('Trade update:', order);
  }

  // Cleanup
  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
  }
}

export const institutionalService = new InstitutionalService();
export default institutionalService;