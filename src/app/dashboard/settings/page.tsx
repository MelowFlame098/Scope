"use client";

import { useState } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import { ProtectedRoute } from '@/components/ProtectedRoute';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { 
  CogIcon,
  UserIcon,
  BellIcon,
  ShieldCheckIcon,
  CreditCardIcon,
  EyeIcon,
  EyeSlashIcon
} from '@heroicons/react/24/outline';

const SettingsPage = () => {
  const { user } = useAuth();
  const [showPassword, setShowPassword] = useState(false);
  const [formData, setFormData] = useState({
    firstName: user?.firstName || '',
    lastName: user?.lastName || '',
    email: user?.email || '',
    phone: '',
    timezone: 'UTC-5',
    currency: 'USD',
    language: 'en'
  });

  const [notifications, setNotifications] = useState({
    priceAlerts: true,
    tradeExecutions: true,
    marketNews: false,
    systemUpdates: true,
    emailNotifications: true,
    pushNotifications: false
  });

  const [security, setSecurity] = useState({
    twoFactorAuth: false,
    loginAlerts: true,
    sessionTimeout: '30'
  });

  const handleInputChange = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const handleNotificationChange = (field: string, value: boolean) => {
    setNotifications(prev => ({ ...prev, [field]: value }));
  };

  const handleSecurityChange = (field: string, value: any) => {
    setSecurity(prev => ({ ...prev, [field]: value }));
  };

  const handleSaveProfile = () => {
    console.log('Saving profile:', formData);
    // Handle profile save
  };

  const handleSaveNotifications = () => {
    console.log('Saving notifications:', notifications);
    // Handle notifications save
  };

  const handleSaveSecurity = () => {
    console.log('Saving security:', security);
    // Handle security save
  };

  return (
    <ProtectedRoute requiredPlan="free">
      <div className="min-h-screen bg-gray-900 p-6">
        <div className="max-w-4xl mx-auto">
          {/* Header */}
          <div className="mb-8">
            <h1 className="text-3xl font-bold text-white mb-2 flex items-center">
              <CogIcon className="w-8 h-8 mr-3" />
              Settings
            </h1>
            <p className="text-gray-400">Manage your account preferences and security settings</p>
          </div>

          <Tabs defaultValue="profile" className="space-y-6">
            <TabsList className="bg-gray-800 border-gray-700">
              <TabsTrigger value="profile" className="data-[state=active]:bg-gray-700">
                <UserIcon className="w-4 h-4 mr-2" />
                Profile
              </TabsTrigger>
              <TabsTrigger value="notifications" className="data-[state=active]:bg-gray-700">
                <BellIcon className="w-4 h-4 mr-2" />
                Notifications
              </TabsTrigger>
              <TabsTrigger value="security" className="data-[state=active]:bg-gray-700">
                <ShieldCheckIcon className="w-4 h-4 mr-2" />
                Security
              </TabsTrigger>
              <TabsTrigger value="subscription" className="data-[state=active]:bg-gray-700">
                <CreditCardIcon className="w-4 h-4 mr-2" />
                Subscription
              </TabsTrigger>
            </TabsList>

            {/* Profile Settings */}
            <TabsContent value="profile">
              <Card className="bg-gray-800 border-gray-700">
                <CardHeader>
                  <CardTitle className="text-white">Profile Information</CardTitle>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <Label htmlFor="firstName" className="text-gray-300">First Name</Label>
                      <Input
                        id="firstName"
                        value={formData.firstName}
                        onChange={(e) => handleInputChange('firstName', e.target.value)}
                        className="bg-gray-700 border-gray-600 text-white mt-1"
                      />
                    </div>
                    <div>
                      <Label htmlFor="lastName" className="text-gray-300">Last Name</Label>
                      <Input
                        id="lastName"
                        value={formData.lastName}
                        onChange={(e) => handleInputChange('lastName', e.target.value)}
                        className="bg-gray-700 border-gray-600 text-white mt-1"
                      />
                    </div>
                  </div>

                  <div>
                    <Label htmlFor="email" className="text-gray-300">Email Address</Label>
                    <Input
                      id="email"
                      type="email"
                      value={formData.email}
                      onChange={(e) => handleInputChange('email', e.target.value)}
                      className="bg-gray-700 border-gray-600 text-white mt-1"
                    />
                  </div>

                  <div>
                    <Label htmlFor="phone" className="text-gray-300">Phone Number</Label>
                    <Input
                      id="phone"
                      type="tel"
                      value={formData.phone}
                      onChange={(e) => handleInputChange('phone', e.target.value)}
                      className="bg-gray-700 border-gray-600 text-white mt-1"
                      placeholder="+1 (555) 123-4567"
                    />
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div>
                      <Label className="text-gray-300">Timezone</Label>
                      <Select value={formData.timezone} onValueChange={(value) => handleInputChange('timezone', value)}>
                        <SelectTrigger className="bg-gray-700 border-gray-600 text-white mt-1">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent className="bg-gray-700 border-gray-600">
                          <SelectItem value="UTC-8">Pacific Time (UTC-8)</SelectItem>
                          <SelectItem value="UTC-7">Mountain Time (UTC-7)</SelectItem>
                          <SelectItem value="UTC-6">Central Time (UTC-6)</SelectItem>
                          <SelectItem value="UTC-5">Eastern Time (UTC-5)</SelectItem>
                          <SelectItem value="UTC+0">UTC</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    <div>
                      <Label className="text-gray-300">Currency</Label>
                      <Select value={formData.currency} onValueChange={(value) => handleInputChange('currency', value)}>
                        <SelectTrigger className="bg-gray-700 border-gray-600 text-white mt-1">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent className="bg-gray-700 border-gray-600">
                          <SelectItem value="USD">USD ($)</SelectItem>
                          <SelectItem value="EUR">EUR (€)</SelectItem>
                          <SelectItem value="GBP">GBP (£)</SelectItem>
                          <SelectItem value="JPY">JPY (¥)</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    <div>
                      <Label className="text-gray-300">Language</Label>
                      <Select value={formData.language} onValueChange={(value) => handleInputChange('language', value)}>
                        <SelectTrigger className="bg-gray-700 border-gray-600 text-white mt-1">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent className="bg-gray-700 border-gray-600">
                          <SelectItem value="en">English</SelectItem>
                          <SelectItem value="es">Spanish</SelectItem>
                          <SelectItem value="fr">French</SelectItem>
                          <SelectItem value="de">German</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>

                  <Button onClick={handleSaveProfile} className="bg-blue-600 hover:bg-blue-700">
                    Save Profile
                  </Button>
                </CardContent>
              </Card>
            </TabsContent>

            {/* Notification Settings */}
            <TabsContent value="notifications">
              <Card className="bg-gray-800 border-gray-700">
                <CardHeader>
                  <CardTitle className="text-white">Notification Preferences</CardTitle>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="space-y-4">
                    {Object.entries(notifications).map(([key, value]) => (
                      <div key={key} className="flex items-center justify-between p-3 bg-gray-700/50 rounded-lg">
                        <div>
                          <div className="font-medium text-white">
                            {key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
                          </div>
                          <div className="text-sm text-gray-400">
                            {key === 'priceAlerts' && 'Get notified when your price alerts are triggered'}
                            {key === 'tradeExecutions' && 'Receive notifications when trades are executed'}
                            {key === 'marketNews' && 'Stay updated with relevant market news'}
                            {key === 'systemUpdates' && 'Important system and platform updates'}
                            {key === 'emailNotifications' && 'Receive notifications via email'}
                            {key === 'pushNotifications' && 'Browser push notifications'}
                          </div>
                        </div>
                        <Button
                          variant={value ? "default" : "outline"}
                          size="sm"
                          onClick={() => handleNotificationChange(key, !value)}
                          className={value ? "bg-green-600 hover:bg-green-700" : "border-gray-600 text-gray-300 hover:bg-gray-700"}
                        >
                          {value ? 'On' : 'Off'}
                        </Button>
                      </div>
                    ))}
                  </div>

                  <Button onClick={handleSaveNotifications} className="bg-blue-600 hover:bg-blue-700">
                    Save Notification Settings
                  </Button>
                </CardContent>
              </Card>
            </TabsContent>

            {/* Security Settings */}
            <TabsContent value="security">
              <Card className="bg-gray-800 border-gray-700">
                <CardHeader>
                  <CardTitle className="text-white">Security Settings</CardTitle>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="space-y-4">
                    <div className="flex items-center justify-between p-3 bg-gray-700/50 rounded-lg">
                      <div>
                        <div className="font-medium text-white">Two-Factor Authentication</div>
                        <div className="text-sm text-gray-400">Add an extra layer of security to your account</div>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Badge className={security.twoFactorAuth ? "bg-green-600" : "bg-gray-600"}>
                          {security.twoFactorAuth ? 'Enabled' : 'Disabled'}
                        </Badge>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleSecurityChange('twoFactorAuth', !security.twoFactorAuth)}
                          className="border-gray-600 text-gray-300 hover:bg-gray-700"
                        >
                          {security.twoFactorAuth ? 'Disable' : 'Enable'}
                        </Button>
                      </div>
                    </div>

                    <div className="flex items-center justify-between p-3 bg-gray-700/50 rounded-lg">
                      <div>
                        <div className="font-medium text-white">Login Alerts</div>
                        <div className="text-sm text-gray-400">Get notified of new login attempts</div>
                      </div>
                      <Button
                        variant={security.loginAlerts ? "default" : "outline"}
                        size="sm"
                        onClick={() => handleSecurityChange('loginAlerts', !security.loginAlerts)}
                        className={security.loginAlerts ? "bg-green-600 hover:bg-green-700" : "border-gray-600 text-gray-300 hover:bg-gray-700"}
                      >
                        {security.loginAlerts ? 'On' : 'Off'}
                      </Button>
                    </div>

                    <div className="p-3 bg-gray-700/50 rounded-lg">
                      <div className="flex items-center justify-between mb-3">
                        <div>
                          <div className="font-medium text-white">Session Timeout</div>
                          <div className="text-sm text-gray-400">Automatically log out after inactivity</div>
                        </div>
                      </div>
                      <Select value={security.sessionTimeout} onValueChange={(value) => handleSecurityChange('sessionTimeout', value)}>
                        <SelectTrigger className="bg-gray-700 border-gray-600 text-white">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent className="bg-gray-700 border-gray-600">
                          <SelectItem value="15">15 minutes</SelectItem>
                          <SelectItem value="30">30 minutes</SelectItem>
                          <SelectItem value="60">1 hour</SelectItem>
                          <SelectItem value="120">2 hours</SelectItem>
                          <SelectItem value="never">Never</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>

                  <div className="border-t border-gray-700 pt-6">
                    <h3 className="text-lg font-medium text-white mb-4">Change Password</h3>
                    <div className="space-y-4">
                      <div>
                        <Label htmlFor="currentPassword" className="text-gray-300">Current Password</Label>
                        <div className="relative mt-1">
                          <Input
                            id="currentPassword"
                            type={showPassword ? "text" : "password"}
                            className="bg-gray-700 border-gray-600 text-white pr-10"
                          />
                          <button
                            type="button"
                            onClick={() => setShowPassword(!showPassword)}
                            className="absolute inset-y-0 right-0 pr-3 flex items-center"
                          >
                            {showPassword ? (
                              <EyeSlashIcon className="h-4 w-4 text-gray-400" />
                            ) : (
                              <EyeIcon className="h-4 w-4 text-gray-400" />
                            )}
                          </button>
                        </div>
                      </div>
                      <div>
                        <Label htmlFor="newPassword" className="text-gray-300">New Password</Label>
                        <Input
                          id="newPassword"
                          type="password"
                          className="bg-gray-700 border-gray-600 text-white mt-1"
                        />
                      </div>
                      <div>
                        <Label htmlFor="confirmPassword" className="text-gray-300">Confirm New Password</Label>
                        <Input
                          id="confirmPassword"
                          type="password"
                          className="bg-gray-700 border-gray-600 text-white mt-1"
                        />
                      </div>
                      <Button className="bg-blue-600 hover:bg-blue-700">
                        Update Password
                      </Button>
                    </div>
                  </div>

                  <Button onClick={handleSaveSecurity} className="bg-blue-600 hover:bg-blue-700">
                    Save Security Settings
                  </Button>
                </CardContent>
              </Card>
            </TabsContent>

            {/* Subscription Settings */}
            <TabsContent value="subscription">
              <Card className="bg-gray-800 border-gray-700">
                <CardHeader>
                  <CardTitle className="text-white">Subscription Management</CardTitle>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="p-4 bg-gray-700/50 rounded-lg">
                    <div className="flex items-center justify-between mb-4">
                      <div>
                        <h3 className="text-lg font-medium text-white">Current Plan</h3>
                        <p className="text-gray-400">Your current subscription details</p>
                      </div>
                      <Badge className="bg-blue-600 text-white">
                        {user?.subscriptionPlan || 'Free'}
                      </Badge>
                    </div>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="text-gray-400">Plan:</span>
                        <span className="text-white ml-2">{user?.subscriptionPlan || 'Free'}</span>
                      </div>
                      <div>
                        <span className="text-gray-400">Status:</span>
                        <span className="text-green-400 ml-2">Active</span>
                      </div>
                      <div>
                        <span className="text-gray-400">Next Billing:</span>
                        <span className="text-white ml-2">February 15, 2024</span>
                      </div>
                      <div>
                        <span className="text-gray-400">Amount:</span>
                        <span className="text-white ml-2">$0.00/month</span>
                      </div>
                    </div>
                  </div>

                  <div className="flex space-x-4">
                    <Button className="bg-blue-600 hover:bg-blue-700">
                      Upgrade Plan
                    </Button>
                    <Button variant="outline" className="border-gray-600 text-gray-300 hover:bg-gray-700">
                      View Billing History
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </ProtectedRoute>
  );
};

export default SettingsPage;