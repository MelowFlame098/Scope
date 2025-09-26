"use client";

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  ChartBarIcon,
  NewspaperIcon,
  ChartPieIcon,
  ChatBubbleLeftRightIcon,
  ArchiveBoxIcon,
  EyeIcon,
  Bars3Icon,
  XMarkIcon
} from '@heroicons/react/24/outline';

// Import area components
import { HubArea } from './areas/HubArea';
import { InformationTerminal } from './areas/InformationTerminal';
import { DepthAnalysis } from './areas/DepthAnalysis';
import { ChatArea } from './areas/ChatArea';
import { ArchiveArea } from './areas/ArchiveArea';

interface FiveAreasLayoutProps {
  user: any;
}

export const FiveAreasLayout: React.FC<FiveAreasLayoutProps> = ({ user }) => {
  const [activeArea, setActiveArea] = useState('hub');
  const [isCompactMode, setIsCompactMode] = useState(false);
  const [visibleAreas, setVisibleAreas] = useState(['hub', 'information']);

  const areas = [
    {
      id: 'hub',
      name: 'Hub',
      icon: ChartBarIcon,
      description: 'Watchlist, Charts, AI & Indicators',
      color: 'blue',
      component: HubArea
    },
    {
      id: 'information',
      name: 'Information Terminal',
      icon: NewspaperIcon,
      description: 'Global News & Social Feeds',
      color: 'green',
      component: InformationTerminal
    },
    {
      id: 'depth',
      name: 'Depth Analysis',
      icon: ChartPieIcon,
      description: 'Order Flow & Volume Analysis',
      color: 'purple',
      component: DepthAnalysis
    },
    {
      id: 'chat',
      name: 'Chat Area',
      icon: ChatBubbleLeftRightIcon,
      description: 'Trade Ideas & Community',
      color: 'orange',
      component: ChatArea
    },
    {
      id: 'archive',
      name: 'Archive',
      icon: ArchiveBoxIcon,
      description: 'Personal Journals & Reports',
      color: 'gray',
      component: ArchiveArea
    }
  ];

  const getAreaColor = (color: string) => {
    switch (color) {
      case 'blue': return 'border-blue-500 bg-blue-500/10';
      case 'green': return 'border-green-500 bg-green-500/10';
      case 'purple': return 'border-purple-500 bg-purple-500/10';
      case 'orange': return 'border-orange-500 bg-orange-500/10';
      case 'gray': return 'border-gray-500 bg-gray-500/10';
      default: return 'border-gray-600 bg-gray-800';
    }
  };

  const toggleAreaVisibility = (areaId: string) => {
    setVisibleAreas(prev => 
      prev.includes(areaId) 
        ? prev.filter(id => id !== areaId)
        : [...prev, areaId]
    );
  };

  const renderArea = (area: any) => {
    const AreaComponent = area.component;
    return (
      <Card key={area.id} className={`${getAreaColor(area.color)} h-full`}>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <area.icon className="w-5 h-5" />
              <CardTitle className="text-lg text-white">{area.name}</CardTitle>
            </div>
            <div className="flex items-center space-x-2">
              <Badge variant="outline" className="text-xs">
                {area.description}
              </Badge>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => toggleAreaVisibility(area.id)}
                className="h-6 w-6 p-0"
              >
                <EyeIcon className="w-4 h-4" />
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent className="pt-0 h-full">
          <AreaComponent user={user} />
        </CardContent>
      </Card>
    );
  };

  if (isCompactMode) {
    // Tabbed view for compact mode
    return (
      <div className="h-full bg-gray-900">
        <div className="flex items-center justify-between p-4 border-b border-gray-700">
          <h2 className="text-xl font-bold text-white">Trading Platform</h2>
          <div className="flex items-center space-x-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setIsCompactMode(false)}
            >
              <Bars3Icon className="w-4 h-4 mr-2" />
              Grid View
            </Button>
          </div>
        </div>

        <Tabs value={activeArea} onValueChange={setActiveArea} className="h-full">
          <TabsList className="grid w-full grid-cols-5 bg-gray-800">
            {areas.map((area) => (
              <TabsTrigger
                key={area.id}
                value={area.id}
                className="flex items-center space-x-2"
              >
                <area.icon className="w-4 h-4" />
                <span className="hidden sm:inline">{area.name}</span>
              </TabsTrigger>
            ))}
          </TabsList>

          {areas.map((area) => (
            <TabsContent key={area.id} value={area.id} className="h-full mt-0">
              <div className="p-4 h-full">
                <area.component user={user} />
              </div>
            </TabsContent>
          ))}
        </Tabs>
      </div>
    );
  }

  // Grid layout for full view
  return (
    <div className="h-full bg-gray-900 p-4">
      {/* Header Controls */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-2xl font-bold text-white">Trading Platform - 5 Areas</h2>
        <div className="flex items-center space-x-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setIsCompactMode(true)}
          >
            <XMarkIcon className="w-4 h-4 mr-2" />
            Compact View
          </Button>
          
          {/* Area visibility toggles */}
          <div className="flex items-center space-x-1">
            {areas.map((area) => (
              <Button
                key={area.id}
                variant={visibleAreas.includes(area.id) ? "default" : "outline"}
                size="sm"
                onClick={() => toggleAreaVisibility(area.id)}
                className="h-8"
              >
                <area.icon className="w-3 h-3 mr-1" />
                {area.name}
              </Button>
            ))}
          </div>
        </div>
      </div>

      {/* Dynamic Grid Layout */}
      <div className={`grid gap-4 h-full ${
        visibleAreas.length === 1 ? 'grid-cols-1' :
        visibleAreas.length === 2 ? 'grid-cols-1 lg:grid-cols-2' :
        visibleAreas.length === 3 ? 'grid-cols-1 lg:grid-cols-3' :
        visibleAreas.length === 4 ? 'grid-cols-1 lg:grid-cols-2 xl:grid-cols-4' :
        'grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-5'
      }`}>
        {areas
          .filter(area => visibleAreas.includes(area.id))
          .map(renderArea)}
      </div>
    </div>
  );
};

export default FiveAreasLayout;