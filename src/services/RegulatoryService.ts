import axios from 'axios';
import { io, Socket } from 'socket.io-client';

// Types for regulatory compliance
export interface KYCProfile {
  id: string;
  userId: string;
  status: 'pending' | 'in_review' | 'approved' | 'rejected' | 'expired';
  tier: 'basic' | 'enhanced' | 'institutional';
  personalInfo: {
    firstName: string;
    lastName: string;
    dateOfBirth: Date;
    nationality: string;
    countryOfResidence: string;
    address: {
      street: string;
      city: string;
      state: string;
      postalCode: string;
      country: string;
    };
    phoneNumber: string;
    email: string;
  };
  identityDocuments: {
    type: 'passport' | 'drivers_license' | 'national_id' | 'utility_bill' | 'bank_statement';
    documentNumber: string;
    issuingCountry: string;
    expiryDate: Date;
    fileUrl: string;
    verificationStatus: 'pending' | 'verified' | 'rejected';
    uploadedAt: Date;
  }[];
  financialInfo: {
    employmentStatus: 'employed' | 'self_employed' | 'unemployed' | 'retired' | 'student';
    occupation: string;
    employer?: string;
    annualIncome: number;
    sourceOfFunds: string[];
    netWorth: number;
    investmentExperience: 'none' | 'limited' | 'good' | 'extensive';
    riskTolerance: 'conservative' | 'moderate' | 'aggressive';
  };
  pepStatus: {
    isPEP: boolean;
    pepType?: 'domestic' | 'foreign' | 'international';
    position?: string;
    country?: string;
    relatedPersons?: string[];
  };
  sanctions: {
    isOnSanctionsList: boolean;
    sanctionLists: string[];
    lastChecked: Date;
  };
  riskScore: {
    overall: number;
    factors: {
      geography: number;
      occupation: number;
      sourceOfFunds: number;
      transactionPattern: number;
      pepStatus: number;
    };
    lastCalculated: Date;
  };
  reviewHistory: {
    reviewedBy: string;
    reviewDate: Date;
    decision: 'approved' | 'rejected' | 'requires_additional_info';
    comments: string;
    documentsRequested?: string[];
  }[];
  createdAt: Date;
  updatedAt: Date;
  expiryDate: Date;
}

export interface AMLAlert {
  id: string;
  userId: string;
  alertType: 'suspicious_transaction' | 'unusual_pattern' | 'high_risk_jurisdiction' | 'sanctions_match' | 'threshold_breach';
  severity: 'low' | 'medium' | 'high' | 'critical';
  status: 'open' | 'investigating' | 'escalated' | 'closed' | 'false_positive';
  description: string;
  triggerData: {
    transactionIds?: string[];
    amount?: number;
    currency?: string;
    counterparty?: string;
    jurisdiction?: string;
    pattern?: string;
    threshold?: number;
  };
  riskFactors: {
    factor: string;
    score: number;
    description: string;
  }[];
  investigationNotes: {
    investigatorId: string;
    timestamp: Date;
    note: string;
    action: string;
  }[];
  assignedTo?: string;
  escalatedTo?: string;
  resolution?: {
    decision: 'no_action' | 'enhanced_monitoring' | 'account_restriction' | 'sar_filing' | 'account_closure';
    reason: string;
    actionTaken: string;
    resolvedBy: string;
    resolvedAt: Date;
  };
  createdAt: Date;
  updatedAt: Date;
  dueDate: Date;
}

export interface SARReport {
  id: string;
  userId: string;
  alertIds: string[];
  reportType: 'suspicious_activity' | 'currency_transaction' | 'cross_border';
  status: 'draft' | 'pending_review' | 'submitted' | 'acknowledged';
  filingJurisdiction: string;
  reportingPeriod: {
    startDate: Date;
    endDate: Date;
  };
  suspiciousActivity: {
    description: string;
    amountInvolved: number;
    currency: string;
    transactionDates: Date[];
    suspicionReasons: string[];
    redFlags: string[];
  };
  subjectInfo: {
    name: string;
    address: string;
    identification: string;
    relationship: 'customer' | 'beneficial_owner' | 'authorized_user' | 'other';
  }[];
  narrativeDescription: string;
  supportingDocuments: {
    type: string;
    description: string;
    fileUrl: string;
    uploadedAt: Date;
  }[];
  preparedBy: string;
  reviewedBy?: string;
  submittedBy?: string;
  submittedAt?: Date;
  acknowledgmentNumber?: string;
  createdAt: Date;
  updatedAt: Date;
}

export interface RegulatoryReport {
  id: string;
  reportType: 'mifid_ii' | 'emir' | 'sftr' | 'cftc' | 'sec' | 'finra' | 'custom';
  jurisdiction: string;
  reportingPeriod: {
    startDate: Date;
    endDate: Date;
  };
  status: 'generating' | 'ready' | 'submitted' | 'acknowledged' | 'rejected';
  data: {
    transactions: any[];
    positions: any[];
    exposures: any[];
    riskMetrics: any[];
  };
  validationResults: {
    isValid: boolean;
    errors: string[];
    warnings: string[];
  };
  submissionDetails?: {
    submittedAt: Date;
    submittedBy: string;
    confirmationNumber: string;
    acknowledgmentDate?: Date;
  };
  fileUrls: {
    xml?: string;
    csv?: string;
    pdf?: string;
  };
  createdAt: Date;
  updatedAt: Date;
  dueDate: Date;
}

export interface AuditTrail {
  id: string;
  userId: string;
  sessionId: string;
  action: string;
  resource: string;
  resourceId?: string;
  method: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';
  endpoint: string;
  requestData?: any;
  responseStatus: number;
  responseData?: any;
  ipAddress: string;
  userAgent: string;
  location?: {
    country: string;
    city: string;
    coordinates?: [number, number];
  };
  riskScore: number;
  flags: string[];
  timestamp: Date;
  duration: number; // milliseconds
}

export interface CompliancePolicy {
  id: string;
  name: string;
  description: string;
  type: 'kyc' | 'aml' | 'trading' | 'reporting' | 'data_protection' | 'operational';
  jurisdiction: string[];
  applicableUserTypes: string[];
  rules: {
    id: string;
    name: string;
    description: string;
    condition: string;
    action: string;
    parameters: any;
    isActive: boolean;
  }[];
  thresholds: {
    name: string;
    value: number;
    currency?: string;
    period?: string;
  }[];
  escalationMatrix: {
    level: number;
    condition: string;
    assignTo: string;
    timeLimit: number; // hours
    actions: string[];
  }[];
  isActive: boolean;
  version: string;
  effectiveDate: Date;
  expiryDate?: Date;
  createdBy: string;
  approvedBy?: string;
  createdAt: Date;
  updatedAt: Date;
}

export interface RegulatoryRequirement {
  id: string;
  regulation: string;
  jurisdiction: string;
  category: 'capital' | 'liquidity' | 'reporting' | 'conduct' | 'operational' | 'market';
  description: string;
  requirements: {
    id: string;
    title: string;
    description: string;
    mandatory: boolean;
    deadline?: Date;
    frequency?: 'daily' | 'weekly' | 'monthly' | 'quarterly' | 'annually';
    status: 'compliant' | 'non_compliant' | 'partially_compliant' | 'not_applicable';
    evidence?: string[];
    lastAssessed: Date;
    nextAssessment: Date;
  }[];
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  penalties: {
    type: 'fine' | 'suspension' | 'revocation' | 'warning';
    amount?: number;
    description: string;
  }[];
  contacts: {
    regulator: string;
    contactPerson: string;
    email: string;
    phone: string;
  }[];
  lastUpdated: Date;
  nextReview: Date;
}

export interface DataPrivacyRecord {
  id: string;
  userId: string;
  dataType: 'personal' | 'financial' | 'trading' | 'communication' | 'behavioral';
  purpose: string;
  legalBasis: 'consent' | 'contract' | 'legal_obligation' | 'vital_interests' | 'public_task' | 'legitimate_interests';
  consentGiven: boolean;
  consentDate?: Date;
  dataRetentionPeriod: number; // days
  dataLocation: string[];
  sharingAgreements: {
    thirdParty: string;
    purpose: string;
    legalBasis: string;
    dataTypes: string[];
    agreementDate: Date;
    expiryDate?: Date;
  }[];
  accessRequests: {
    requestId: string;
    requestDate: Date;
    requestType: 'access' | 'rectification' | 'erasure' | 'portability' | 'restriction';
    status: 'pending' | 'approved' | 'rejected' | 'completed';
    responseDate?: Date;
    reason?: string;
  }[];
  breachIncidents: {
    incidentId: string;
    incidentDate: Date;
    description: string;
    severity: 'low' | 'medium' | 'high' | 'critical';
    affectedData: string[];
    notificationRequired: boolean;
    notificationDate?: Date;
    mitigationActions: string[];
  }[];
  createdAt: Date;
  updatedAt: Date;
}

class RegulatoryService {
  private baseURL: string;
  private socket: Socket | null = null;
  private apiKey: string;

  constructor() {
    this.baseURL = process.env.REACT_APP_REGULATORY_API_URL || 'http://localhost:8004';
    this.apiKey = process.env.REACT_APP_API_KEY || '';
    this.initializeWebSocket();
  }

  private initializeWebSocket() {
    this.socket = io(`${this.baseURL}/regulatory`, {
      auth: {
        token: this.apiKey,
      },
      transports: ['websocket'],
    });

    this.socket.on('connect', () => {
      console.log('Connected to regulatory service');
    });

    this.socket.on('aml_alert', (data: AMLAlert) => {
      this.handleAMLAlert(data);
    });

    this.socket.on('kyc_update', (data: any) => {
      this.handleKYCUpdate(data);
    });

    this.socket.on('regulatory_deadline', (data: any) => {
      this.handleRegulatoryDeadline(data);
    });
  }

  // KYC Management
  async getKYCProfile(userId: string): Promise<KYCProfile> {
    try {
      const response = await axios.get(`${this.baseURL}/api/kyc/${userId}`, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get KYC profile:', error);
      throw error;
    }
  }

  async updateKYCProfile(userId: string, updates: Partial<KYCProfile>): Promise<KYCProfile> {
    try {
      const response = await axios.put(`${this.baseURL}/api/kyc/${userId}`, updates, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to update KYC profile:', error);
      throw error;
    }
  }

  async uploadKYCDocument(userId: string, file: File, documentType: string): Promise<any> {
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('documentType', documentType);

      const response = await axios.post(`${this.baseURL}/api/kyc/${userId}/documents`, formData, {
        headers: {
          Authorization: `Bearer ${this.apiKey}`,
          'Content-Type': 'multipart/form-data',
        },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to upload KYC document:', error);
      throw error;
    }
  }

  async reviewKYCProfile(userId: string, decision: string, comments: string): Promise<KYCProfile> {
    try {
      const response = await axios.post(`${this.baseURL}/api/kyc/${userId}/review`, {
        decision,
        comments,
      }, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to review KYC profile:', error);
      throw error;
    }
  }

  async runKYCScreening(userId: string): Promise<any> {
    try {
      const response = await axios.post(`${this.baseURL}/api/kyc/${userId}/screening`, {}, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to run KYC screening:', error);
      throw error;
    }
  }

  // AML Management
  async getAMLAlerts(filters?: any): Promise<AMLAlert[]> {
    try {
      const response = await axios.get(`${this.baseURL}/api/aml/alerts`, {
        params: filters,
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get AML alerts:', error);
      throw error;
    }
  }

  async getAMLAlert(alertId: string): Promise<AMLAlert> {
    try {
      const response = await axios.get(`${this.baseURL}/api/aml/alerts/${alertId}`, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get AML alert:', error);
      throw error;
    }
  }

  async updateAMLAlert(alertId: string, updates: Partial<AMLAlert>): Promise<AMLAlert> {
    try {
      const response = await axios.put(`${this.baseURL}/api/aml/alerts/${alertId}`, updates, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to update AML alert:', error);
      throw error;
    }
  }

  async addInvestigationNote(alertId: string, note: string, action: string): Promise<AMLAlert> {
    try {
      const response = await axios.post(`${this.baseURL}/api/aml/alerts/${alertId}/notes`, {
        note,
        action,
      }, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to add investigation note:', error);
      throw error;
    }
  }

  async escalateAMLAlert(alertId: string, escalateTo: string, reason: string): Promise<AMLAlert> {
    try {
      const response = await axios.post(`${this.baseURL}/api/aml/alerts/${alertId}/escalate`, {
        escalateTo,
        reason,
      }, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to escalate AML alert:', error);
      throw error;
    }
  }

  async runAMLScreening(userId: string, transactionData?: any): Promise<AMLAlert[]> {
    try {
      const response = await axios.post(`${this.baseURL}/api/aml/screening`, {
        userId,
        transactionData,
      }, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to run AML screening:', error);
      throw error;
    }
  }

  // SAR Reporting
  async createSARReport(report: Omit<SARReport, 'id' | 'status' | 'createdAt' | 'updatedAt'>): Promise<SARReport> {
    try {
      const response = await axios.post(`${this.baseURL}/api/sar/reports`, report, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to create SAR report:', error);
      throw error;
    }
  }

  async getSARReports(filters?: any): Promise<SARReport[]> {
    try {
      const response = await axios.get(`${this.baseURL}/api/sar/reports`, {
        params: filters,
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get SAR reports:', error);
      throw error;
    }
  }

  async submitSARReport(reportId: string): Promise<SARReport> {
    try {
      const response = await axios.post(`${this.baseURL}/api/sar/reports/${reportId}/submit`, {}, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to submit SAR report:', error);
      throw error;
    }
  }

  // Regulatory Reporting
  async generateRegulatoryReport(reportType: string, jurisdiction: string, period: any): Promise<RegulatoryReport> {
    try {
      const response = await axios.post(`${this.baseURL}/api/regulatory/reports/generate`, {
        reportType,
        jurisdiction,
        period,
      }, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to generate regulatory report:', error);
      throw error;
    }
  }

  async getRegulatoryReports(filters?: any): Promise<RegulatoryReport[]> {
    try {
      const response = await axios.get(`${this.baseURL}/api/regulatory/reports`, {
        params: filters,
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get regulatory reports:', error);
      throw error;
    }
  }

  async submitRegulatoryReport(reportId: string): Promise<RegulatoryReport> {
    try {
      const response = await axios.post(`${this.baseURL}/api/regulatory/reports/${reportId}/submit`, {}, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to submit regulatory report:', error);
      throw error;
    }
  }

  async validateRegulatoryReport(reportId: string): Promise<any> {
    try {
      const response = await axios.post(`${this.baseURL}/api/regulatory/reports/${reportId}/validate`, {}, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to validate regulatory report:', error);
      throw error;
    }
  }

  // Audit Trail
  async getAuditTrail(filters?: any): Promise<AuditTrail[]> {
    try {
      const response = await axios.get(`${this.baseURL}/api/audit/trail`, {
        params: filters,
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get audit trail:', error);
      throw error;
    }
  }

  async logAuditEvent(event: Omit<AuditTrail, 'id' | 'timestamp'>): Promise<AuditTrail> {
    try {
      const response = await axios.post(`${this.baseURL}/api/audit/log`, event, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to log audit event:', error);
      throw error;
    }
  }

  async exportAuditTrail(filters: any, format: 'csv' | 'json' | 'pdf'): Promise<Blob> {
    try {
      const response = await axios.post(`${this.baseURL}/api/audit/export`, {
        filters,
        format,
      }, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
        responseType: 'blob',
      });
      return response.data;
    } catch (error) {
      console.error('Failed to export audit trail:', error);
      throw error;
    }
  }

  // Compliance Policies
  async getCompliancePolicies(): Promise<CompliancePolicy[]> {
    try {
      const response = await axios.get(`${this.baseURL}/api/compliance/policies`, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get compliance policies:', error);
      throw error;
    }
  }

  async createCompliancePolicy(policy: Omit<CompliancePolicy, 'id' | 'createdAt' | 'updatedAt'>): Promise<CompliancePolicy> {
    try {
      const response = await axios.post(`${this.baseURL}/api/compliance/policies`, policy, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to create compliance policy:', error);
      throw error;
    }
  }

  async updateCompliancePolicy(policyId: string, updates: Partial<CompliancePolicy>): Promise<CompliancePolicy> {
    try {
      const response = await axios.put(`${this.baseURL}/api/compliance/policies/${policyId}`, updates, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to update compliance policy:', error);
      throw error;
    }
  }

  // Regulatory Requirements
  async getRegulatoryRequirements(jurisdiction?: string): Promise<RegulatoryRequirement[]> {
    try {
      const response = await axios.get(`${this.baseURL}/api/regulatory/requirements`, {
        params: { jurisdiction },
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get regulatory requirements:', error);
      throw error;
    }
  }

  async updateRequirementStatus(requirementId: string, status: string, evidence?: string[]): Promise<RegulatoryRequirement> {
    try {
      const response = await axios.put(`${this.baseURL}/api/regulatory/requirements/${requirementId}/status`, {
        status,
        evidence,
      }, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to update requirement status:', error);
      throw error;
    }
  }

  // Data Privacy
  async getDataPrivacyRecord(userId: string): Promise<DataPrivacyRecord> {
    try {
      const response = await axios.get(`${this.baseURL}/api/privacy/${userId}`, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get data privacy record:', error);
      throw error;
    }
  }

  async processDataRequest(userId: string, requestType: string, details: any): Promise<any> {
    try {
      const response = await axios.post(`${this.baseURL}/api/privacy/${userId}/request`, {
        requestType,
        details,
      }, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to process data request:', error);
      throw error;
    }
  }

  async reportDataBreach(incident: any): Promise<any> {
    try {
      const response = await axios.post(`${this.baseURL}/api/privacy/breach`, incident, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to report data breach:', error);
      throw error;
    }
  }

  // Real-time subscriptions
  subscribeToAMLAlerts(callback: (alert: AMLAlert) => void) {
    if (this.socket) {
      this.socket.emit('subscribe_aml');
      this.socket.on('aml_alert', callback);
    }
  }

  subscribeToKYCUpdates(callback: (update: any) => void) {
    if (this.socket) {
      this.socket.emit('subscribe_kyc');
      this.socket.on('kyc_update', callback);
    }
  }

  subscribeToRegulatoryDeadlines(callback: (deadline: any) => void) {
    if (this.socket) {
      this.socket.emit('subscribe_deadlines');
      this.socket.on('regulatory_deadline', callback);
    }
  }

  // Event handlers
  private handleAMLAlert(alert: AMLAlert) {
    console.log('AML alert:', alert);
  }

  private handleKYCUpdate(update: any) {
    console.log('KYC update:', update);
  }

  private handleRegulatoryDeadline(deadline: any) {
    console.log('Regulatory deadline:', deadline);
  }

  // Cleanup
  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
  }
}

export const regulatoryService = new RegulatoryService();
export default regulatoryService;