"use client";

import { useState } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import { ProtectedRoute } from '@/components/ProtectedRoute';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  BellIcon,
  ExclamationTriangleIcon,
  InformationCircleIcon,
  CheckCircleIcon,
  XMarkIcon,
  Cog6ToothIcon
} from '@heroicons/react/24/outline';

const NotificationsPage = () => {
  const { user } = useAuth();
  const [notifications, setNotifications] = useState([
    {
      id: '1',
      type: 'alert',
      title: 'Price Alert: AAPL',
      message: 'Apple Inc. has reached your target price of $155.00',
      timestamp: '2024-01-15 10:30:00',
      read: false,
      priority: 'high'
    },
    {
      id: '2',
      type: 'trade',
      title: 'Order Filled',
      message: 'Your buy order for 10 shares of MSFT has been executed at $295.40',
      timestamp: '2024-01-15 09:15:00',
      read: false,
      priority: 'medium'
    },
    {
      id: '3',
      type: 'news',
      title: 'Market Update',
      message: 'S&P 500 reaches new all-time high amid strong earnings reports',
      timestamp: '2024-01-15 08:00:00',
      read: true,
      priority: 'low'
    },
    {
      id: '4',
      type: 'system',
      title: 'Account Security',
      message: 'New login detected from Windows device in New York',
      timestamp: '2024-01-14 18:45:00',
      read: true,
      priority: 'high'
    },
    {
      id: '5',
      type: 'alert',
      title: 'Stop Loss Triggered',
      message: 'Your stop loss order for TSLA has been triggered at $215.00',
      timestamp: '2024-01-14 15:30:00',
      read: true,
      priority: 'high'
    }
  ]);

  const markAsRead = (id: string) => {
    setNotifications(prev => 
      prev.map(notif => 
        notif.id === id ? { ...notif, read: true } : notif
      )
    );
  };

  const markAllAsRead = () => {
    setNotifications(prev => 
      prev.map(notif => ({ ...notif, read: true }))
    );
  };

  const deleteNotification = (id: string) => {
    setNotifications(prev => prev.filter(notif => notif.id !== id));
  };

  const getNotificationIcon = (type: string) => {
    switch (type) {
      case 'alert':
        return <ExclamationTriangleIcon className="w-5 h-5 text-yellow-400" />;
      case 'trade':
        return <CheckCircleIcon className="w-5 h-5 text-green-400" />;
      case 'news':
        return <InformationCircleIcon className="w-5 h-5 text-blue-400" />;
      case 'system':
        return <Cog6ToothIcon className="w-5 h-5 text-purple-400" />;
      default:
        return <BellIcon className="w-5 h-5 text-gray-400" />;
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high':
        return 'bg-red-600';
      case 'medium':
        return 'bg-yellow-600';
      case 'low':
        return 'bg-green-600';
      default:
        return 'bg-gray-600';
    }
  };

  const unreadCount = notifications.filter(n => !n.read).length;
  const alertNotifications = notifications.filter(n => n.type === 'alert');
  const tradeNotifications = notifications.filter(n => n.type === 'trade');
  const newsNotifications = notifications.filter(n => n.type === 'news');
  const systemNotifications = notifications.filter(n => n.type === 'system');

  const NotificationItem = ({ notification }: { notification: any }) => (
    <div className={`p-4 border-l-4 ${
      notification.read ? 'bg-gray-800/50 border-gray-600' : 'bg-gray-800 border-blue-500'
    } rounded-r-lg`}>
      <div className="flex items-start justify-between">
        <div className="flex items-start space-x-3 flex-1">
          {getNotificationIcon(notification.type)}
          <div className="flex-1">
            <div className="flex items-center space-x-2 mb-1">
              <h3 className={`font-medium ${notification.read ? 'text-gray-300' : 'text-white'}`}>
                {notification.title}
              </h3>
              <Badge className={`${getPriorityColor(notification.priority)} text-white text-xs`}>
                {notification.priority}
              </Badge>
            </div>
            <p className={`text-sm ${notification.read ? 'text-gray-400' : 'text-gray-300'} mb-2`}>
              {notification.message}
            </p>
            <p className="text-xs text-gray-500">{notification.timestamp}</p>
          </div>
        </div>
        <div className="flex items-center space-x-2 ml-4">
          {!notification.read && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => markAsRead(notification.id)}
              className="border-gray-600 text-gray-300 hover:bg-gray-700"
            >
              Mark Read
            </Button>
          )}
          <Button
            variant="outline"
            size="sm"
            onClick={() => deleteNotification(notification.id)}
            className="border-gray-600 text-gray-300 hover:bg-gray-700"
          >
            <XMarkIcon className="w-4 h-4" />
          </Button>
        </div>
      </div>
    </div>
  );

  return (
    <ProtectedRoute requiredPlan="free">
      <div className="min-h-screen bg-gray-900 p-6">
        <div className="max-w-4xl mx-auto">
          {/* Header */}
          <div className="mb-8">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-3xl font-bold text-white mb-2 flex items-center">
                  <BellIcon className="w-8 h-8 mr-3" />
                  Notifications
                  {unreadCount > 0 && (
                    <Badge className="ml-3 bg-red-600 text-white">
                      {unreadCount} new
                    </Badge>
                  )}
                </h1>
                <p className="text-gray-400">Stay updated with alerts and system messages</p>
              </div>
              {unreadCount > 0 && (
                <Button
                  onClick={markAllAsRead}
                  className="bg-blue-600 hover:bg-blue-700"
                >
                  Mark All Read
                </Button>
              )}
            </div>
          </div>

          {/* Notification Stats */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
            <Card className="bg-gray-800 border-gray-700">
              <CardContent className="p-4">
                <div className="flex items-center space-x-2">
                  <ExclamationTriangleIcon className="w-5 h-5 text-yellow-400" />
                  <div>
                    <div className="text-lg font-bold text-white">{alertNotifications.length}</div>
                    <div className="text-sm text-gray-400">Alerts</div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-gray-800 border-gray-700">
              <CardContent className="p-4">
                <div className="flex items-center space-x-2">
                  <CheckCircleIcon className="w-5 h-5 text-green-400" />
                  <div>
                    <div className="text-lg font-bold text-white">{tradeNotifications.length}</div>
                    <div className="text-sm text-gray-400">Trades</div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-gray-800 border-gray-700">
              <CardContent className="p-4">
                <div className="flex items-center space-x-2">
                  <InformationCircleIcon className="w-5 h-5 text-blue-400" />
                  <div>
                    <div className="text-lg font-bold text-white">{newsNotifications.length}</div>
                    <div className="text-sm text-gray-400">News</div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-gray-800 border-gray-700">
              <CardContent className="p-4">
                <div className="flex items-center space-x-2">
                  <Cog6ToothIcon className="w-5 h-5 text-purple-400" />
                  <div>
                    <div className="text-lg font-bold text-white">{systemNotifications.length}</div>
                    <div className="text-sm text-gray-400">System</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Notifications */}
          <Tabs defaultValue="all" className="space-y-4">
            <TabsList className="bg-gray-800 border-gray-700">
              <TabsTrigger value="all" className="data-[state=active]:bg-gray-700">
                All ({notifications.length})
              </TabsTrigger>
              <TabsTrigger value="unread" className="data-[state=active]:bg-gray-700">
                Unread ({unreadCount})
              </TabsTrigger>
              <TabsTrigger value="alerts" className="data-[state=active]:bg-gray-700">
                Alerts ({alertNotifications.length})
              </TabsTrigger>
              <TabsTrigger value="trades" className="data-[state=active]:bg-gray-700">
                Trades ({tradeNotifications.length})
              </TabsTrigger>
            </TabsList>

            <TabsContent value="all">
              <div className="space-y-4">
                {notifications.map((notification) => (
                  <NotificationItem key={notification.id} notification={notification} />
                ))}
              </div>
            </TabsContent>

            <TabsContent value="unread">
              <div className="space-y-4">
                {notifications.filter(n => !n.read).map((notification) => (
                  <NotificationItem key={notification.id} notification={notification} />
                ))}
              </div>
            </TabsContent>

            <TabsContent value="alerts">
              <div className="space-y-4">
                {alertNotifications.map((notification) => (
                  <NotificationItem key={notification.id} notification={notification} />
                ))}
              </div>
            </TabsContent>

            <TabsContent value="trades">
              <div className="space-y-4">
                {tradeNotifications.map((notification) => (
                  <NotificationItem key={notification.id} notification={notification} />
                ))}
              </div>
            </TabsContent>
          </Tabs>

          {notifications.length === 0 && (
            <Card className="bg-gray-800 border-gray-700">
              <CardContent className="p-8 text-center">
                <BellIcon className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-white mb-2">No notifications</h3>
                <p className="text-gray-400">You're all caught up! New notifications will appear here.</p>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </ProtectedRoute>
  );
};

export default NotificationsPage;