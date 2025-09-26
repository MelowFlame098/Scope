"use client";

import React, { useState, useEffect } from 'react';
// UI Components - using inline styles
const Card = ({ children, className = '' }: { children: React.ReactNode; className?: string }) => (
  <div className={`bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 ${className}`}>
    {children}
  </div>
);

const CardHeader = ({ children }: { children: React.ReactNode }) => (
  <div className="p-6 pb-4">{children}</div>
);

const CardTitle = ({ children, className = '' }: { 
  children: React.ReactNode;
  className?: string;
}) => (
  <h3 className={`text-lg font-semibold text-gray-900 dark:text-white ${className}`}>{children}</h3>
);

const CardContent = ({ children, className = '' }: { children: React.ReactNode; className?: string }) => (
  <div className={`p-6 pt-0 ${className}`}>{children}</div>
);

const Button = ({ children, onClick, className = '', variant = 'default', size = 'default', disabled = false }: {
  children: React.ReactNode;
  onClick?: () => void;
  className?: string;
  variant?: 'default' | 'outline' | 'destructive';
  size?: 'default' | 'sm';
  disabled?: boolean;
}) => {
  const baseClasses = 'rounded-lg font-medium transition-colors';
  const sizeClasses = {
    default: 'px-4 py-2',
    sm: 'px-3 py-1.5 text-sm'
  };
  const variantClasses = {
    default: 'bg-blue-600 text-white hover:bg-blue-700 disabled:bg-gray-400',
    outline: 'border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700',
    destructive: 'bg-red-600 text-white hover:bg-red-700'
  };
  
  return (
    <button 
      onClick={onClick} 
      disabled={disabled}
      className={`${baseClasses} ${sizeClasses[size]} ${variantClasses[variant]} ${className}`}
    >
      {children}
    </button>
  );
};

const Input = ({ value, onChange, placeholder, className = '', type = 'text', id, step }: {
  value: string | number;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  placeholder?: string;
  className?: string;
  type?: string;
  id?: string;
  step?: string;
}) => (
  <input
    id={id}
    type={type}
    step={step}
    value={value}
    onChange={onChange}
    placeholder={placeholder}
    className={`px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white ${className}`}
  />
);

const Label = ({ children, htmlFor }: { children: React.ReactNode; htmlFor?: string }) => (
  <label htmlFor={htmlFor} className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
    {children}
  </label>
);

const Switch = ({ checked, onCheckedChange }: {
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
}) => (
  <button
    onClick={() => onCheckedChange(!checked)}
    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
      checked ? 'bg-blue-600' : 'bg-gray-200 dark:bg-gray-700'
    }`}
  >
    <span
      className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
        checked ? 'translate-x-6' : 'translate-x-1'
      }`}
    />
  </button>
);

const Select = ({ value, onValueChange, children }: {
  value: string;
  onValueChange: (value: string) => void;
  children: React.ReactNode;
}) => {
  const [isOpen, setIsOpen] = useState(false);
  
  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white text-left flex justify-between items-center"
      >
        <span>{value}</span>
        <span className="ml-2">▼</span>
      </button>
      {isOpen && (
        <div className="absolute z-10 w-full mt-1 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg shadow-lg">
          {React.Children.map(children, (child) => {
            if (React.isValidElement(child) && child.type === SelectItem) {
              return React.cloneElement(child as React.ReactElement<any>, {
                onClick: () => {
                  onValueChange(child.props.value);
                  setIsOpen(false);
                }
              });
            }
            return child;
          })}
        </div>
      )}
    </div>
  );
};

const SelectTrigger = ({ children, className = '' }: { 
  children: React.ReactNode;
  className?: string;
}) => (
  <div className={className}>{children}</div>
);

const SelectValue = ({ placeholder }: { placeholder?: string }) => (
  <span>{placeholder}</span>
);

const SelectContent = ({ children }: { children: React.ReactNode }) => (
  <div>{children}</div>
);

const SelectItem = ({ value, children, onClick }: {
  value: string;
  children: React.ReactNode;
  onClick?: () => void;
}) => (
  <button
    onClick={onClick}
    className="w-full px-3 py-2 text-left hover:bg-gray-100 dark:hover:bg-gray-600 text-gray-900 dark:text-white"
  >
    {children}
  </button>
);

const Tabs = ({ value, onValueChange, children }: {
  value: string;
  onValueChange: (value: string) => void;
  children: React.ReactNode;
}) => (
  <div className="w-full">
    {React.Children.map(children, (child) => {
      if (React.isValidElement(child)) {
        if (child.type === TabsList) {
          return React.cloneElement(child as React.ReactElement<any>, {
            activeTab: value,
            onTabChange: onValueChange
          });
        }
        if (child.type === TabsContent) {
          return React.cloneElement(child as React.ReactElement<any>, {
            isActive: child.props.value === value
          });
        }
      }
      return child;
    })}
  </div>
);

const TabsList = ({ children, activeTab, onTabChange, className = '' }: {
  children: React.ReactNode;
  activeTab?: string;
  onTabChange?: (value: string) => void;
  className?: string;
}) => (
  <div className={`flex space-x-1 bg-gray-100 dark:bg-gray-700 p-1 rounded-lg mb-4 ${className}`}>
    {React.Children.map(children, (child) => {
      if (React.isValidElement(child) && child.type === TabsTrigger) {
        return React.cloneElement(child as React.ReactElement<any>, {
          isActive: child.props.value === activeTab,
          onClick: () => onTabChange?.(child.props.value)
        });
      }
      return child;
    })}
  </div>
);

const TabsTrigger = ({ value, children, isActive, onClick }: {
  value: string;
  children: React.ReactNode;
  isActive?: boolean;
  onClick?: () => void;
}) => (
  <button
    onClick={onClick}
    className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${
      isActive
        ? 'bg-white dark:bg-gray-600 text-gray-900 dark:text-white shadow-sm'
        : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
    }`}
  >
    {children}
  </button>
);

const TabsContent = ({ value, children, isActive, className = '' }: {
  value: string;
  children: React.ReactNode;
  isActive?: boolean;
  className?: string;
}) => (
  <div className={`${isActive ? 'block' : 'hidden'} ${className}`}>
    {children}
  </div>
);

const Badge = ({ children, variant = 'default', className = '' }: {
  children: React.ReactNode;
  variant?: 'default' | 'secondary' | 'outline';
  className?: string;
}) => {
  const variantClasses = {
    default: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300',
    secondary: 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300',
    outline: 'border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300'
  };
  
  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${variantClasses[variant]} ${className}`}>
      {children}
    </span>
  );
};

const Alert = ({ children, variant = 'default' }: {
  children: React.ReactNode;
  variant?: 'default' | 'destructive';
}) => {
  const variantClasses = {
    default: 'bg-blue-50 border-blue-200 text-blue-800',
    destructive: 'bg-red-50 border-red-200 text-red-800'
  };
  
  return (
    <div className={`p-4 border rounded-lg ${variantClasses[variant]}`}>
      {children}
    </div>
  );
};

const AlertDescription = ({ children }: { children: React.ReactNode }) => (
  <div className="text-sm">{children}</div>
);
import {
  BellIcon,
  Cog6ToothIcon as SettingsIcon,
  ArrowTrendingUpIcon as TrendingUpIcon,
  ArrowTrendingDownIcon as TrendingDownIcon,
  ExclamationTriangleIcon as AlertTriangleIcon,
  InformationCircleIcon as InfoIcon,
  CheckIcon,
  XMarkIcon as XIcon,
  TrashIcon,
  PlusIcon,
  EnvelopeIcon as MailIcon,
  DevicePhoneMobileIcon as SmartphoneIcon,
  ComputerDesktopIcon as MonitorIcon,
  CurrencyDollarIcon as DollarSignIcon,
  ChartBarIcon as BarChart3Icon,
  NewspaperIcon as NewsIcon,
  UsersIcon,
  ClockIcon,
  FunnelIcon as FilterIcon
} from '@heroicons/react/24/outline';

interface Notification {
  id: string;
  type: 'PRICE_ALERT' | 'NEWS' | 'PORTFOLIO' | 'TRADING' | 'SYSTEM' | 'COMMUNITY';
  priority: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  title: string;
  message: string;
  data?: Record<string, any>;
  isRead: boolean;
  createdAt: string;
  expiresAt?: string;
}

interface PriceAlert {
  id: string;
  symbol: string;
  condition: 'ABOVE' | 'BELOW' | 'CHANGE_PERCENT' | 'VOLUME';
  value: number;
  currentValue: number;
  isActive: boolean;
  createdAt: string;
  triggeredAt?: string;
}

interface NotificationPreferences {
  channels: {
    email: boolean;
    push: boolean;
    inApp: boolean;
  };
  types: {
    priceAlerts: boolean;
    news: boolean;
    portfolio: boolean;
    trading: boolean;
    system: boolean;
    community: boolean;
  };
  frequency: {
    immediate: boolean;
    daily: boolean;
    weekly: boolean;
  };
  quietHours: {
    enabled: boolean;
    startTime: string;
    endTime: string;
  };
}

const NotificationCenter: React.FC = () => {
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [priceAlerts, setPriceAlerts] = useState<PriceAlert[]>([]);
  const [preferences, setPreferences] = useState<NotificationPreferences>({
    channels: { email: true, push: true, inApp: true },
    types: {
      priceAlerts: true,
      news: true,
      portfolio: true,
      trading: true,
      system: true,
      community: true
    },
    frequency: { immediate: true, daily: false, weekly: false },
    quietHours: { enabled: false, startTime: '22:00', endTime: '08:00' }
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('notifications');
  const [filterType, setFilterType] = useState<string>('ALL');
  const [filterPriority, setFilterPriority] = useState<string>('ALL');
  const [showOnlyUnread, setShowOnlyUnread] = useState(false);

  // New alert form
  const [newAlert, setNewAlert] = useState({
    symbol: '',
    condition: 'ABOVE' as const,
    value: 0
  });

  useEffect(() => {
    fetchNotifications();
    fetchPriceAlerts();
    fetchPreferences();
  }, []);

  const fetchNotifications = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/v2/notifications', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      
      if (!response.ok) throw new Error('Failed to fetch notifications');
      
      const data = await response.json();
      setNotifications(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch notifications');
    } finally {
      setLoading(false);
    }
  };

  const fetchPriceAlerts = async () => {
    try {
      const response = await fetch('/api/v2/notifications/price-alerts', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      
      if (!response.ok) throw new Error('Failed to fetch price alerts');
      
      const data = await response.json();
      setPriceAlerts(data);
    } catch (err) {
      console.error('Error fetching price alerts:', err);
    }
  };

  const fetchPreferences = async () => {
    try {
      const response = await fetch('/api/v2/notifications/preferences', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      
      if (!response.ok) throw new Error('Failed to fetch preferences');
      
      const data = await response.json();
      setPreferences(data);
    } catch (err) {
      console.error('Error fetching preferences:', err);
    }
  };

  const markAsRead = async (notificationId: string) => {
    try {
      const response = await fetch(`/api/v2/notifications/${notificationId}/read`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      
      if (!response.ok) throw new Error('Failed to mark as read');
      
      setNotifications(prev => 
        prev.map(notif => 
          notif.id === notificationId ? { ...notif, isRead: true } : notif
        )
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to mark as read');
    }
  };

  const markAllAsRead = async () => {
    try {
      const response = await fetch('/api/v2/notifications/read-all', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      
      if (!response.ok) throw new Error('Failed to mark all as read');
      
      setNotifications(prev => 
        prev.map(notif => ({ ...notif, isRead: true }))
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to mark all as read');
    }
  };

  const deleteNotification = async (notificationId: string) => {
    try {
      const response = await fetch(`/api/v2/notifications/${notificationId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      
      if (!response.ok) throw new Error('Failed to delete notification');
      
      setNotifications(prev => 
        prev.filter(notif => notif.id !== notificationId)
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete notification');
    }
  };

  const createPriceAlert = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/v2/notifications/price-alerts', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify(newAlert)
      });
      
      if (!response.ok) throw new Error('Failed to create price alert');
      
      const alert = await response.json();
      setPriceAlerts(prev => [alert, ...prev]);
      setNewAlert({ symbol: '', condition: 'ABOVE', value: 0 });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create price alert');
    } finally {
      setLoading(false);
    }
  };

  const deletePriceAlert = async (alertId: string) => {
    try {
      const response = await fetch(`/api/v2/notifications/price-alerts/${alertId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      
      if (!response.ok) throw new Error('Failed to delete price alert');
      
      setPriceAlerts(prev => prev.filter(alert => alert.id !== alertId));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete price alert');
    }
  };

  const updatePreferences = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/v2/notifications/preferences', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify(preferences)
      });
      
      if (!response.ok) throw new Error('Failed to update preferences');
      
      // Show success message
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update preferences');
    } finally {
      setLoading(false);
    }
  };

  const filteredNotifications = notifications.filter(notif => {
    if (filterType !== 'ALL' && notif.type !== filterType) return false;
    if (filterPriority !== 'ALL' && notif.priority !== filterPriority) return false;
    if (showOnlyUnread && notif.isRead) return false;
    return true;
  });

  const unreadCount = notifications.filter(n => !n.isRead).length;

  const getNotificationIcon = (type: string) => {
    switch (type) {
      case 'PRICE_ALERT': return <DollarSignIcon className="h-4 w-4" />;
      case 'NEWS': return <NewsIcon className="h-4 w-4" />;
      case 'PORTFOLIO': return <BarChart3Icon className="h-4 w-4" />;
      case 'TRADING': return <TrendingUpIcon className="h-4 w-4" />;
      case 'SYSTEM': return <SettingsIcon className="h-4 w-4" />;
      case 'COMMUNITY': return <UsersIcon className="h-4 w-4" />;
      default: return <InfoIcon className="h-4 w-4" />;
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'CRITICAL': return 'bg-red-100 text-red-800 border-red-200';
      case 'HIGH': return 'bg-orange-100 text-orange-800 border-orange-200';
      case 'MEDIUM': return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'LOW': return 'bg-blue-100 text-blue-800 border-blue-200';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const formatTimeAgo = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffInSeconds = Math.floor((now.getTime() - date.getTime()) / 1000);
    
    if (diffInSeconds < 60) return 'Just now';
    if (diffInSeconds < 3600) return `${Math.floor(diffInSeconds / 60)}m ago`;
    if (diffInSeconds < 86400) return `${Math.floor(diffInSeconds / 3600)}h ago`;
    return `${Math.floor(diffInSeconds / 86400)}d ago`;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div className="flex items-center space-x-4">
          <div className="relative">
            <BellIcon className="h-8 w-8" />
            {unreadCount > 0 && (
              <Badge className="absolute -top-2 -right-2 h-5 w-5 rounded-full p-0 flex items-center justify-center text-xs">
                {unreadCount}
              </Badge>
            )}
          </div>
          <div>
            <h1 className="text-3xl font-bold">Notifications</h1>
            <p className="text-gray-600">{unreadCount} unread notifications</p>
          </div>
        </div>
        <Button onClick={markAllAsRead} disabled={unreadCount === 0}>
          <CheckIcon className="h-4 w-4 mr-2" />
          Mark All Read
        </Button>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertTriangleIcon className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="notifications">Notifications</TabsTrigger>
          <TabsTrigger value="alerts">Price Alerts</TabsTrigger>
          <TabsTrigger value="settings">Settings</TabsTrigger>
        </TabsList>

        <TabsContent value="notifications" className="space-y-4">
          {/* Filters */}
          <Card>
            <CardContent className="p-4">
              <div className="flex flex-wrap gap-4 items-center">
                <Select value={filterType} onValueChange={setFilterType}>
                  <SelectTrigger className="w-40">
                    <SelectValue placeholder="Type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="ALL">All Types</SelectItem>
                    <SelectItem value="PRICE_ALERT">Price Alerts</SelectItem>
                    <SelectItem value="NEWS">News</SelectItem>
                    <SelectItem value="PORTFOLIO">Portfolio</SelectItem>
                    <SelectItem value="TRADING">Trading</SelectItem>
                    <SelectItem value="SYSTEM">System</SelectItem>
                    <SelectItem value="COMMUNITY">Community</SelectItem>
                  </SelectContent>
                </Select>
                <Select value={filterPriority} onValueChange={setFilterPriority}>
                  <SelectTrigger className="w-40">
                    <SelectValue placeholder="Priority" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="ALL">All Priorities</SelectItem>
                    <SelectItem value="CRITICAL">Critical</SelectItem>
                    <SelectItem value="HIGH">High</SelectItem>
                    <SelectItem value="MEDIUM">Medium</SelectItem>
                    <SelectItem value="LOW">Low</SelectItem>
                  </SelectContent>
                </Select>
                <div className="flex items-center space-x-2">
                  <Switch
                    checked={showOnlyUnread}
                    onCheckedChange={setShowOnlyUnread}
                  />
                  <Label>Unread only</Label>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Notifications List */}
          <div className="space-y-2">
            {filteredNotifications.map((notification) => (
              <Card key={notification.id} className={`${!notification.isRead ? 'border-l-4 border-l-blue-500' : ''}`}>
                <CardContent className="p-4">
                  <div className="flex items-start justify-between">
                    <div className="flex items-start space-x-3 flex-1">
                      <div className={`p-2 rounded-full ${getPriorityColor(notification.priority)}`}>
                        {getNotificationIcon(notification.type)}
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center space-x-2 mb-1">
                          <h3 className={`font-semibold ${!notification.isRead ? 'text-gray-900' : 'text-gray-600'}`}>
                            {notification.title}
                          </h3>
                          <Badge variant="outline" className={getPriorityColor(notification.priority)}>
                            {notification.priority}
                          </Badge>
                          <Badge variant="secondary">
                            {notification.type.replace('_', ' ')}
                          </Badge>
                        </div>
                        <p className={`text-sm ${!notification.isRead ? 'text-gray-700' : 'text-gray-500'}`}>
                          {notification.message}
                        </p>
                        <div className="flex items-center space-x-4 mt-2">
                          <span className="text-xs text-gray-500 flex items-center">
                            <ClockIcon className="h-3 w-3 mr-1" />
                            {formatTimeAgo(notification.createdAt)}
                          </span>
                          {notification.expiresAt && (
                            <span className="text-xs text-orange-500">
                              Expires: {new Date(notification.expiresAt).toLocaleDateString()}
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      {!notification.isRead && (
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => markAsRead(notification.id)}
                        >
                          <CheckIcon className="h-4 w-4" />
                        </Button>
                      )}
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => deleteNotification(notification.id)}
                      >
                        <TrashIcon className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
            
            {filteredNotifications.length === 0 && (
              <Card>
                <CardContent className="p-8 text-center">
                  <BellIcon className="h-12 w-12 mx-auto mb-4 text-gray-400" />
                  <p className="text-gray-500">No notifications found</p>
                </CardContent>
              </Card>
            )}
          </div>
        </TabsContent>

        <TabsContent value="alerts" className="space-y-4">
          {/* Create Alert */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <PlusIcon className="h-5 w-5 mr-2" />
                Create Price Alert
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div>
                  <Label htmlFor="symbol">Symbol</Label>
                  <Input
                    id="symbol"
                    placeholder="e.g., AAPL"
                    value={newAlert.symbol}
                    onChange={(e) => setNewAlert(prev => ({ ...prev, symbol: e.target.value.toUpperCase() }))}
                  />
                </div>
                <div>
                  <Label htmlFor="condition">Condition</Label>
                  <Select
                    value={newAlert.condition}
                    onValueChange={(value: any) => setNewAlert(prev => ({ ...prev, condition: value }))}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="ABOVE">Above</SelectItem>
                      <SelectItem value="BELOW">Below</SelectItem>
                      <SelectItem value="CHANGE_PERCENT">Change %</SelectItem>
                      <SelectItem value="VOLUME">Volume</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <Label htmlFor="value">Value</Label>
                  <Input
                    id="value"
                    type="number"
                    step="0.01"
                    value={newAlert.value}
                    onChange={(e) => setNewAlert(prev => ({ ...prev, value: parseFloat(e.target.value) || 0 }))}
                  />
                </div>
                <div className="flex items-end">
                  <Button
                    onClick={createPriceAlert}
                    disabled={loading || !newAlert.symbol || !newAlert.value}
                    className="w-full"
                  >
                    {loading ? 'Creating...' : 'Create Alert'}
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Active Alerts */}
          <Card>
            <CardHeader>
              <CardTitle>Active Price Alerts ({priceAlerts.filter(a => a.isActive).length})</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {priceAlerts.filter(alert => alert.isActive).map((alert) => (
                  <div key={alert.id} className="flex items-center justify-between p-3 border rounded-lg">
                    <div className="flex items-center space-x-4">
                      <div className="p-2 bg-blue-100 rounded-full">
                        <DollarSignIcon className="h-4 w-4 text-blue-600" />
                      </div>
                      <div>
                        <p className="font-semibold">{alert.symbol}</p>
                        <p className="text-sm text-gray-600">
                          {alert.condition.replace('_', ' ').toLowerCase()} {alert.value}
                        </p>
                        <p className="text-xs text-gray-500">
                          Current: {alert.currentValue} • Created {formatTimeAgo(alert.createdAt)}
                        </p>
                      </div>
                    </div>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => deletePriceAlert(alert.id)}
                    >
                      <TrashIcon className="h-4 w-4" />
                    </Button>
                  </div>
                ))}
                
                {priceAlerts.filter(a => a.isActive).length === 0 && (
                  <div className="text-center py-8">
                    <DollarSignIcon className="h-12 w-12 mx-auto mb-4 text-gray-400" />
                    <p className="text-gray-500">No active price alerts</p>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Triggered Alerts */}
          {priceAlerts.filter(alert => alert.triggeredAt).length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Recently Triggered</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {priceAlerts.filter(alert => alert.triggeredAt).map((alert) => (
                    <div key={alert.id} className="flex items-center justify-between p-3 border rounded-lg bg-green-50">
                      <div className="flex items-center space-x-4">
                        <div className="p-2 bg-green-100 rounded-full">
                          <CheckIcon className="h-4 w-4 text-green-600" />
                        </div>
                        <div>
                          <p className="font-semibold">{alert.symbol}</p>
                          <p className="text-sm text-gray-600">
                            {alert.condition.replace('_', ' ').toLowerCase()} {alert.value}
                          </p>
                          <p className="text-xs text-gray-500">
                            Triggered {formatTimeAgo(alert.triggeredAt!)}
                          </p>
                        </div>
                      </div>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => deletePriceAlert(alert.id)}
                      >
                        <TrashIcon className="h-4 w-4" />
                      </Button>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="settings" className="space-y-4">
          {/* Notification Channels */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <SettingsIcon className="h-5 w-5 mr-2" />
                Notification Channels
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <MailIcon className="h-5 w-5 text-gray-500" />
                  <div>
                    <p className="font-medium">Email Notifications</p>
                    <p className="text-sm text-gray-600">Receive notifications via email</p>
                  </div>
                </div>
                <Switch
                  checked={preferences.channels.email}
                  onCheckedChange={(checked) => 
                    setPreferences(prev => ({
                      ...prev,
                      channels: { ...prev.channels, email: checked }
                    }))
                  }
                />
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <SmartphoneIcon className="h-5 w-5 text-gray-500" />
                  <div>
                    <p className="font-medium">Push Notifications</p>
                    <p className="text-sm text-gray-600">Receive push notifications on your device</p>
                  </div>
                </div>
                <Switch
                  checked={preferences.channels.push}
                  onCheckedChange={(checked) => 
                    setPreferences(prev => ({
                      ...prev,
                      channels: { ...prev.channels, push: checked }
                    }))
                  }
                />
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <MonitorIcon className="h-5 w-5 text-gray-500" />
                  <div>
                    <p className="font-medium">In-App Notifications</p>
                    <p className="text-sm text-gray-600">Show notifications within the application</p>
                  </div>
                </div>
                <Switch
                  checked={preferences.channels.inApp}
                  onCheckedChange={(checked) => 
                    setPreferences(prev => ({
                      ...prev,
                      channels: { ...prev.channels, inApp: checked }
                    }))
                  }
                />
              </div>
            </CardContent>
          </Card>

          {/* Notification Types */}
          <Card>
            <CardHeader>
              <CardTitle>Notification Types</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {Object.entries(preferences.types).map(([type, enabled]) => (
                <div key={type} className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    {getNotificationIcon(type.toUpperCase())}
                    <div>
                      <p className="font-medium capitalize">{type.replace(/([A-Z])/g, ' $1').trim()}</p>
                      <p className="text-sm text-gray-600">
                        {type === 'priceAlerts' && 'Price movement and threshold alerts'}
                        {type === 'news' && 'Market news and updates'}
                        {type === 'portfolio' && 'Portfolio performance and changes'}
                        {type === 'trading' && 'Trade executions and order updates'}
                        {type === 'system' && 'System maintenance and updates'}
                        {type === 'community' && 'Forum posts and community activity'}
                      </p>
                    </div>
                  </div>
                  <Switch
                    checked={enabled}
                    onCheckedChange={(checked) => 
                      setPreferences(prev => ({
                        ...prev,
                        types: { ...prev.types, [type]: checked }
                      }))
                    }
                  />
                </div>
              ))}
            </CardContent>
          </Card>

          {/* Quiet Hours */}
          <Card>
            <CardHeader>
              <CardTitle>Quiet Hours</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium">Enable Quiet Hours</p>
                  <p className="text-sm text-gray-600">Suppress non-critical notifications during specified hours</p>
                </div>
                <Switch
                  checked={preferences.quietHours.enabled}
                  onCheckedChange={(checked) => 
                    setPreferences(prev => ({
                      ...prev,
                      quietHours: { ...prev.quietHours, enabled: checked }
                    }))
                  }
                />
              </div>
              {preferences.quietHours.enabled && (
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label htmlFor="startTime">Start Time</Label>
                    <Input
                      id="startTime"
                      type="time"
                      value={preferences.quietHours.startTime}
                      onChange={(e) => 
                        setPreferences(prev => ({
                          ...prev,
                          quietHours: { ...prev.quietHours, startTime: e.target.value }
                        }))
                      }
                    />
                  </div>
                  <div>
                    <Label htmlFor="endTime">End Time</Label>
                    <Input
                      id="endTime"
                      type="time"
                      value={preferences.quietHours.endTime}
                      onChange={(e) => 
                        setPreferences(prev => ({
                          ...prev,
                          quietHours: { ...prev.quietHours, endTime: e.target.value }
                        }))
                      }
                    />
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          <Button onClick={updatePreferences} disabled={loading} className="w-full">
            {loading ? 'Saving...' : 'Save Preferences'}
          </Button>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default NotificationCenter;