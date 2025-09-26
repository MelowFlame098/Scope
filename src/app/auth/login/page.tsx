"use client";

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { useAuth } from '@/contexts/AuthContext';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  ChartBarIcon, 
  EyeIcon, 
  EyeSlashIcon,
  ExclamationTriangleIcon
} from '@heroicons/react/24/outline';

export default function LoginPage() {
  const router = useRouter();
  const { login, isLoading } = useAuth();
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    rememberMe: false
  });
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (!formData.email || !formData.password) {
      setError('Please fill in all fields');
      return;
    }

    try {
      const result = await login(formData.email, formData.password);
      if (result.success) {
        router.push('/dashboard');
      } else {
        setError(result.error || 'Login failed');
      }
    } catch (err) {
      setError('An unexpected error occurred');
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value, type, checked } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Logo */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center space-x-2 mb-4">
            <ChartBarIcon className="h-10 w-10 text-blue-400" />
            <span className="text-3xl font-bold text-white">FinScope</span>
          </div>
          <p className="text-gray-300">Sign in to your account</p>
        </div>

        {/* Login Form */}
        <Card className="bg-gray-800/50 border-gray-700 backdrop-blur-sm">
          <CardHeader>
            <CardTitle className="text-white text-center">Welcome Back</CardTitle>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4" data-testid="login-form">
              {error && (
                <Alert className="border-red-500/50 bg-red-500/10" data-testid="login-error">
                  <ExclamationTriangleIcon className="h-4 w-4 text-red-400" />
                  <AlertDescription className="text-red-400">
                    {error}
                  </AlertDescription>
                </Alert>
              )}

              <div className="space-y-2">
                <Label htmlFor="email" className="text-gray-300">Email</Label>
                <Input
                  id="email"
                  name="email"
                  type="email"
                  value={formData.email}
                  onChange={handleInputChange}
                  placeholder="Enter your email"
                  className="bg-gray-700/50 border-gray-600 text-white placeholder-gray-400"
                  data-testid="email-input"
                  required
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="password" className="text-gray-300">Password</Label>
                <div className="relative">
                  <Input
                    id="password"
                    name="password"
                    type={showPassword ? 'text' : 'password'}
                    value={formData.password}
                    onChange={handleInputChange}
                    placeholder="Enter your password"
                    className="bg-gray-700/50 border-gray-600 text-white placeholder-gray-400 pr-10"
                    data-testid="password-input"
                    required
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-300"
                  >
                    {showPassword ? (
                      <EyeSlashIcon className="h-4 w-4" />
                    ) : (
                      <EyeIcon className="h-4 w-4" />
                    )}
                  </button>
                </div>
              </div>

              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <input
                    id="rememberMe"
                    name="rememberMe"
                    type="checkbox"
                    checked={formData.rememberMe}
                    onChange={handleInputChange}
                    className="rounded border-gray-600 bg-gray-700 text-blue-600 focus:ring-blue-500"
                  />
                  <Label htmlFor="rememberMe" className="text-sm text-gray-300">
                    Remember me
                  </Label>
                </div>
                <Link 
                  href="/auth/forgot-password" 
                  className="text-sm text-blue-400 hover:text-blue-300"
                  data-testid="forgot-password-link"
                >
                  Forgot password?
                </Link>
              </div>

              <Button
                type="submit"
                disabled={isLoading}
                className="w-full bg-blue-600 hover:bg-blue-700 text-white"
                data-testid="login-button"
              >
                {isLoading ? 'Signing in...' : 'Sign In'}
              </Button>
            </form>

            <div className="mt-6 text-center">
              <p className="text-gray-400">
                Don't have an account?{' '}
                <Link 
                  href="/auth/register" 
                  className="text-blue-400 hover:text-blue-300 font-medium"
                  data-testid="register-link"
                >
                  Sign up
                </Link>
              </p>
            </div>

            <div className="mt-4 text-center">
              <Link 
                href="/landing" 
                className="text-sm text-gray-400 hover:text-gray-300"
              >
                ← Back to home
              </Link>
            </div>
          </CardContent>
        </Card>

        {/* Demo Credentials */}
        <div className="mt-6 p-4 bg-gray-800/30 rounded-lg border border-gray-700">
          <p className="text-sm text-gray-400 text-center mb-2">Demo Credentials:</p>
          <p className="text-xs text-gray-500 text-center">
            Email: demo@finscope.com | Password: demo123456
          </p>
        </div>
      </div>
    </div>
  );
}