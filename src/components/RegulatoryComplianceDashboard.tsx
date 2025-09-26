import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Shield, 
  FileText, 
  AlertTriangle, 
  CheckCircle, 
  Clock, 
  Download,
  Eye,
  Settings
} from 'lucide-react';

interface ComplianceItem {
  id: string;
  title: string;
  status: 'compliant' | 'warning' | 'non-compliant' | 'pending';
  lastUpdated: string;
  description: string;
  priority: 'high' | 'medium' | 'low';
}

interface RegulatoryReport {
  id: string;
  name: string;
  type: string;
  dueDate: string;
  status: 'submitted' | 'draft' | 'overdue';
  completionRate: number;
}

const RegulatoryComplianceDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState('overview');

  // Mock data for compliance items
  const complianceItems: ComplianceItem[] = [
    {
      id: '1',
      title: 'KYC Documentation',
      status: 'compliant',
      lastUpdated: '2024-01-15',
      description: 'Customer identification and verification processes',
      priority: 'high'
    },
    {
      id: '2',
      title: 'AML Monitoring',
      status: 'warning',
      lastUpdated: '2024-01-14',
      description: 'Anti-money laundering transaction monitoring',
      priority: 'high'
    },
    {
      id: '3',
      title: 'Risk Assessment',
      status: 'compliant',
      lastUpdated: '2024-01-13',
      description: 'Periodic risk assessment and mitigation',
      priority: 'medium'
    },
    {
      id: '4',
      title: 'Data Protection',
      status: 'pending',
      lastUpdated: '2024-01-12',
      description: 'GDPR and data privacy compliance',
      priority: 'high'
    },
    {
      id: '5',
      title: 'Trade Reporting',
      status: 'non-compliant',
      lastUpdated: '2024-01-11',
      description: 'Regulatory trade reporting requirements',
      priority: 'high'
    }
  ];

  // Mock data for regulatory reports
  const regulatoryReports: RegulatoryReport[] = [
    {
      id: '1',
      name: 'Monthly AML Report',
      type: 'AML',
      dueDate: '2024-02-01',
      status: 'draft',
      completionRate: 75
    },
    {
      id: '2',
      name: 'Quarterly Risk Assessment',
      type: 'Risk',
      dueDate: '2024-01-31',
      status: 'submitted',
      completionRate: 100
    },
    {
      id: '3',
      name: 'Trade Surveillance Report',
      type: 'Trading',
      dueDate: '2024-01-28',
      status: 'overdue',
      completionRate: 45
    }
  ];

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'compliant':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'warning':
        return <AlertTriangle className="h-4 w-4 text-yellow-500" />;
      case 'non-compliant':
        return <AlertTriangle className="h-4 w-4 text-red-500" />;
      case 'pending':
        return <Clock className="h-4 w-4 text-blue-500" />;
      default:
        return <Clock className="h-4 w-4 text-gray-500" />;
    }
  };

  const getStatusBadge = (status: string) => {
    const variants: Record<string, "default" | "secondary" | "destructive" | "outline"> = {
      compliant: 'default',
      warning: 'secondary',
      'non-compliant': 'destructive',
      pending: 'outline'
    };
    return variants[status] || 'outline';
  };

  const getReportStatusBadge = (status: string) => {
    const variants: Record<string, "default" | "secondary" | "destructive" | "outline"> = {
      submitted: 'default',
      draft: 'secondary',
      overdue: 'destructive'
    };
    return variants[status] || 'outline';
  };

  const complianceScore = Math.round(
    (complianceItems.filter(item => item.status === 'compliant').length / complianceItems.length) * 100
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Shield className="h-6 w-6 text-blue-600" />
          <h2 className="text-2xl font-bold">Regulatory Compliance</h2>
        </div>
        <Button variant="outline" size="sm">
          <Settings className="h-4 w-4 mr-2" />
          Settings
        </Button>
      </div>

      {/* Compliance Score Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Compliance Score</p>
                <p className="text-2xl font-bold">{complianceScore}%</p>
              </div>
              <Shield className="h-8 w-8 text-green-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Active Issues</p>
                <p className="text-2xl font-bold">
                  {complianceItems.filter(item => item.status === 'non-compliant' || item.status === 'warning').length}
                </p>
              </div>
              <AlertTriangle className="h-8 w-8 text-yellow-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Pending Reviews</p>
                <p className="text-2xl font-bold">
                  {complianceItems.filter(item => item.status === 'pending').length}
                </p>
              </div>
              <Clock className="h-8 w-8 text-blue-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Reports Due</p>
                <p className="text-2xl font-bold">
                  {regulatoryReports.filter(report => report.status !== 'submitted').length}
                </p>
              </div>
              <FileText className="h-8 w-8 text-orange-500" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Content Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="reports">Reports</TabsTrigger>
          <TabsTrigger value="monitoring">Monitoring</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Compliance Status</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {complianceItems.map((item) => (
                  <div key={item.id} className="flex items-center justify-between p-4 border rounded-lg">
                    <div className="flex items-center space-x-3">
                      {getStatusIcon(item.status)}
                      <div>
                        <h4 className="font-medium">{item.title}</h4>
                        <p className="text-sm text-muted-foreground">{item.description}</p>
                        <p className="text-xs text-muted-foreground">Last updated: {item.lastUpdated}</p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Badge variant={getStatusBadge(item.status)}>
                        {item.status.replace('-', ' ')}
                      </Badge>
                      <Badge variant="outline" className="text-xs">
                        {item.priority}
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="reports" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Regulatory Reports</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {regulatoryReports.map((report) => (
                  <div key={report.id} className="flex items-center justify-between p-4 border rounded-lg">
                    <div className="flex-1">
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="font-medium">{report.name}</h4>
                        <Badge variant={getReportStatusBadge(report.status)}>
                          {report.status}
                        </Badge>
                      </div>
                      <div className="flex items-center space-x-4 text-sm text-muted-foreground mb-2">
                        <span>Type: {report.type}</span>
                        <span>Due: {report.dueDate}</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Progress value={report.completionRate} className="flex-1" />
                        <span className="text-sm font-medium">{report.completionRate}%</span>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2 ml-4">
                      <Button variant="outline" size="sm">
                        <Eye className="h-4 w-4 mr-1" />
                        View
                      </Button>
                      <Button variant="outline" size="sm">
                        <Download className="h-4 w-4 mr-1" />
                        Export
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="monitoring" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Real-time Monitoring</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Transaction Monitoring</span>
                    <Badge variant="default">Active</Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Risk Assessment</span>
                    <Badge variant="default">Active</Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Compliance Alerts</span>
                    <Badge variant="secondary">3 Pending</Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Audit Trail</span>
                    <Badge variant="default">Recording</Badge>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Recent Alerts</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex items-start space-x-2">
                    <AlertTriangle className="h-4 w-4 text-yellow-500 mt-0.5" />
                    <div className="text-sm">
                      <p className="font-medium">Unusual Trading Pattern</p>
                      <p className="text-muted-foreground">Detected in Account #12345</p>
                      <p className="text-xs text-muted-foreground">2 hours ago</p>
                    </div>
                  </div>
                  <div className="flex items-start space-x-2">
                    <AlertTriangle className="h-4 w-4 text-red-500 mt-0.5" />
                    <div className="text-sm">
                      <p className="font-medium">KYC Document Expired</p>
                      <p className="text-muted-foreground">Client ID: CL789</p>
                      <p className="text-xs text-muted-foreground">1 day ago</p>
                    </div>
                  </div>
                  <div className="flex items-start space-x-2">
                    <CheckCircle className="h-4 w-4 text-green-500 mt-0.5" />
                    <div className="text-sm">
                      <p className="font-medium">Report Submitted</p>
                      <p className="text-muted-foreground">Monthly AML Report</p>
                      <p className="text-xs text-muted-foreground">3 days ago</p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default RegulatoryComplianceDashboard;