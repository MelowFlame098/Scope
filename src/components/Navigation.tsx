'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  BarChart3,
  Brain,
  TrendingUp,
  Newspaper,
  MessageSquare,
  Settings,
  User,
  Menu,
  X,
  Home,
  PieChart,
  Activity
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Sheet, SheetContent, SheetTrigger } from '@/components/ui/sheet';

const Navigation: React.FC = () => {
  const pathname = usePathname();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const navigationItems = [
    {
      name: 'Dashboard',
      href: '/',
      icon: Home,
      description: 'Overview and portfolio summary'
    },
    {
      name: 'Market Data',
      href: '/market',
      icon: BarChart3,
      description: 'Real-time market information'
    },
    {
      name: 'Portfolio',
      href: '/portfolio',
      icon: PieChart,
      description: 'Manage your investments'
    },
    {
      name: 'Analytics',
      href: '/analytics',
      icon: Activity,
      description: 'Advanced market analysis'
    },
    {
      name: 'News & Sentiment',
      href: '/news',
      icon: Newspaper,
      description: 'Market news and sentiment analysis'
    },
    {
      name: 'AI Assistant',
      href: '/ai-chat',
      icon: Brain,
      description: 'AI-powered financial insights',
      badge: 'New'
    },
    {
      name: 'Trading',
      href: '/trading',
      icon: TrendingUp,
      description: 'Execute trades and strategies'
    }
  ];

  const isActive = (href: string) => {
    if (href === '/') {
      return pathname === '/';
    }
    return pathname.startsWith(href);
  };

  const NavItems = ({ mobile = false }: { mobile?: boolean }) => (
    <>
      {navigationItems.map((item) => {
        const Icon = item.icon;
        const active = isActive(item.href);
        
        return (
          <Link
            key={item.name}
            href={item.href}
            className={`flex items-center gap-3 px-3 py-2 rounded-lg transition-colors group ${
              mobile ? 'w-full' : ''
            } ${
              active
                ? 'bg-primary text-primary-foreground'
                : 'text-muted-foreground hover:text-foreground hover:bg-muted'
            }`}
            onClick={() => mobile && setIsMobileMenuOpen(false)}
          >
            <Icon className={`h-5 w-5 ${active ? 'text-primary-foreground' : 'text-muted-foreground group-hover:text-foreground'}`} />
            <div className="flex-1">
              <div className="flex items-center gap-2">
                <span className={`font-medium ${mobile ? 'text-base' : 'text-sm'}`}>
                  {item.name}
                </span>
                {item.badge && (
                  <Badge variant="secondary" className="text-xs">
                    {item.badge}
                  </Badge>
                )}
              </div>
              {mobile && (
                <p className="text-xs text-muted-foreground mt-1">
                  {item.description}
                </p>
              )}
            </div>
          </Link>
        );
      })}
    </>
  );

  return (
    <>
      {/* Desktop Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 border-b">
        <div className="container mx-auto px-4">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <Link href="/" className="flex items-center gap-2">
              <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
                <BarChart3 className="h-5 w-5 text-primary-foreground" />
              </div>
              <span className="font-bold text-xl">FinScope</span>
            </Link>

            {/* Desktop Navigation Items */}
            <div className="hidden lg:flex items-center gap-1">
              <NavItems />
            </div>

            {/* Right Side Actions */}
            <div className="flex items-center gap-3">
              {/* User Menu */}
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="ghost" size="sm" className="relative">
                    <User className="h-5 w-5" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end" className="w-56">
                  <DropdownMenuLabel>My Account</DropdownMenuLabel>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem>
                    <User className="mr-2 h-4 w-4" />
                    <span>Profile</span>
                  </DropdownMenuItem>
                  <DropdownMenuItem>
                    <Settings className="mr-2 h-4 w-4" />
                    <span>Settings</span>
                  </DropdownMenuItem>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem>
                    <span>Log out</span>
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>

              {/* Mobile Menu Trigger */}
              <Sheet open={isMobileMenuOpen} onOpenChange={setIsMobileMenuOpen}>
                <SheetTrigger asChild>
                  <Button variant="ghost" size="sm" className="lg:hidden">
                    <Menu className="h-5 w-5" />
                  </Button>
                </SheetTrigger>
                <SheetContent side="left" className="w-80">
                  <div className="flex items-center justify-between mb-6">
                    <Link href="/" className="flex items-center gap-2">
                      <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
                        <BarChart3 className="h-5 w-5 text-primary-foreground" />
                      </div>
                      <span className="font-bold text-xl">FinScope</span>
                    </Link>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setIsMobileMenuOpen(false)}
                    >
                      <X className="h-5 w-5" />
                    </Button>
                  </div>
                  
                  <div className="space-y-2">
                    <NavItems mobile />
                  </div>
                  
                  <div className="mt-8 pt-6 border-t">
                    <div className="space-y-2">
                      <Button variant="ghost" className="w-full justify-start">
                        <User className="mr-2 h-4 w-4" />
                        Profile
                      </Button>
                      <Button variant="ghost" className="w-full justify-start">
                        <Settings className="mr-2 h-4 w-4" />
                        Settings
                      </Button>
                      <Button variant="ghost" className="w-full justify-start">
                        Log out
                      </Button>
                    </div>
                  </div>
                </SheetContent>
              </Sheet>
            </div>
          </div>
        </div>
      </nav>

      {/* Quick Access Floating Button for AI Chat */}
      {!isActive('/ai-chat') && (
        <div className="fixed bottom-6 right-6 z-40">
          <Link href="/ai-chat">
            <Button size="lg" className="rounded-full shadow-lg hover:shadow-xl transition-shadow">
              <MessageSquare className="h-5 w-5 mr-2" />
              Ask AI
            </Button>
          </Link>
        </div>
      )}
    </>
  );
};

export default Navigation;