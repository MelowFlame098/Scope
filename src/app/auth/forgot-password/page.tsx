"use client";

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  ChartBarIcon, 
  ExclamationTriangleIcon,
  CheckCircleIcon,
  ArrowLeftIcon
} from '@heroicons/react/24/outline';

export default function ForgotPasswordPage() {
  const router = useRouter();
  const [email, setEmail] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);

  const validateEmail = (email: string) => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setSuccess(false);

    if (!email) {
      setError('Please enter your email address');
      return;
    }

    if (!validateEmail(email)) {
      setError('Please enter a valid email address');
      return;
    }

    setIsLoading(true);

    try {
      // Simulate API call for password reset
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // For demo purposes, always show success
      setSuccess(true);
    } catch (err) {
      setError('Failed to send reset email. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  if (success) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900 flex items-center justify-center p-4">
        <Card className="w-full max-w-md bg-gray-800/50 border-gray-700 backdrop-blur-sm" data-testid="reset-success">
          <CardContent className="pt-6">
            <div className="text-center">
              <CheckCircleIcon className="h-16 w-16 text-green-400 mx-auto mb-4" />
              <h2 className="text-2xl font-bold text-white mb-2">Check Your Email</h2>
              <p className="text-gray-300 mb-6">
                We've sent a password reset link to <strong>{email}</strong>
              </p>
              <p className="text-sm text-gray-400 mb-6">
                If you don't see the email, check your spam folder or try again with a different email address.
              </p>
              <div className="space-y-3">
                <Link href="/auth/login">
                  <Button className="w-full bg-blue-600 hover:bg-blue-700 text-white">
                    <ArrowLeftIcon className="h-4 w-4 mr-2" />
                    Back to Login
                  </Button>
                </Link>
                <Button
                  variant="outline"
                  onClick={() => {
                    setSuccess(false);
                    setEmail('');
                  }}
                  className="w-full border-gray-600 text-gray-300 hover:bg-gray-700"
                >
                  Try Different Email
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Logo */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center space-x-2 mb-4">
            <ChartBarIcon className="h-10 w-10 text-blue-400" />
            <span className="text-3xl font-bold text-white">FinScope</span>
          </div>
          <p className="text-gray-300">Reset your password</p>
        </div>

        {/* Reset Form */}
        <Card className="bg-gray-800/50 border-gray-700 backdrop-blur-sm">
          <CardHeader>
            <CardTitle className="text-white text-center">Forgot Password?</CardTitle>
            <p className="text-gray-400 text-center text-sm">
              Enter your email address and we'll send you a link to reset your password.
            </p>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4" data-testid="forgot-password-form">
              {error && (
                <Alert className="border-red-500/50 bg-red-500/10">
                  <ExclamationTriangleIcon className="h-4 w-4 text-red-400" />
                  <AlertDescription className="text-red-400">
                    {error}
                  </AlertDescription>
                </Alert>
              )}

              <div className="space-y-2">
                <Label htmlFor="email" className="text-gray-300">Email Address</Label>
                <Input
                  id="email"
                  name="email"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="Enter your email address"
                  className="bg-gray-700/50 border-gray-600 text-white placeholder-gray-400"
                  data-testid="email-input"
                  required
                />
              </div>

              <Button
                type="submit"
                disabled={isLoading}
                className="w-full bg-blue-600 hover:bg-blue-700 text-white"
                data-testid="reset-button"
              >
                {isLoading ? 'Sending Reset Link...' : 'Send Reset Link'}
              </Button>
            </form>

            <div className="mt-6 text-center space-y-4">
              <p className="text-gray-400">
                Remember your password?{' '}
                <Link 
                  href="/auth/login" 
                  className="text-blue-400 hover:text-blue-300 font-medium"
                >
                  Sign in
                </Link>
              </p>
              
              <p className="text-gray-400">
                Don't have an account?{' '}
                <Link 
                  href="/auth/register" 
                  className="text-blue-400 hover:text-blue-300 font-medium"
                >
                  Create one
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
      </div>
    </div>
  );
}