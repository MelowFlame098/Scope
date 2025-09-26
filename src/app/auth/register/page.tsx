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
  ExclamationTriangleIcon,
  CheckCircleIcon
} from '@heroicons/react/24/outline';

export default function RegisterPage() {
  const router = useRouter();
  const { register, isLoading } = useAuth();
  const [formData, setFormData] = useState({
    firstName: '',
    lastName: '',
    email: '',
    password: '',
    confirmPassword: '',
    acceptTerms: false,
    acceptPrivacy: false,
    marketingEmails: false
  });
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);

  const validateForm = () => {
    if (!formData.firstName || !formData.lastName || !formData.email || !formData.password) {
      setError('Please fill in all required fields');
      return false;
    }

    if (formData.password.length < 8) {
      setError('Password must be at least 8 characters long');
      return false;
    }

    if (formData.password !== formData.confirmPassword) {
      setError('Passwords do not match');
      return false;
    }

    if (!formData.acceptTerms) {
      setError('You must accept the Terms of Service');
      return false;
    }

    if (!formData.acceptPrivacy) {
      setError('You must accept the Privacy Policy');
      return false;
    }

    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(formData.email)) {
      setError('Please enter a valid email address');
      return false;
    }

    return true;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setSuccess(false);

    if (!validateForm()) {
      return;
    }

    try {
      const result = await register(formData);
      if (result.success) {
        setSuccess(true);
        setTimeout(() => {
          router.push('/dashboard');
        }, 2000);
      } else {
        setError(result.error || 'Registration failed');
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

  if (success) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900 flex items-center justify-center p-4">
        <Card className="w-full max-w-md bg-gray-800/50 border-gray-700 backdrop-blur-sm" data-testid="registration-success">
          <CardContent className="pt-6">
            <div className="text-center">
              <CheckCircleIcon className="h-16 w-16 text-green-400 mx-auto mb-4" />
              <h2 className="text-2xl font-bold text-white mb-2">Welcome to FinScope!</h2>
              <p className="text-gray-300 mb-4">
                Your account has been created successfully. Redirecting to dashboard...
              </p>
              <div className="w-8 h-8 border-4 border-blue-500/30 border-t-blue-500 rounded-full animate-spin mx-auto"></div>
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
          <p className="text-gray-300">Create your account</p>
        </div>

        {/* Registration Form */}
        <Card className="bg-gray-800/50 border-gray-700 backdrop-blur-sm">
          <CardHeader>
            <CardTitle className="text-white text-center">Join FinScope</CardTitle>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4" data-testid="register-form">
              {error && (
                <Alert className="border-red-500/50 bg-red-500/10">
                  <ExclamationTriangleIcon className="h-4 w-4 text-red-400" />
                  <AlertDescription className="text-red-400">
                    {error}
                  </AlertDescription>
                </Alert>
              )}

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="firstName" className="text-gray-300">First Name</Label>
                  <Input
                    id="firstName"
                    name="firstName"
                    type="text"
                    value={formData.firstName}
                    onChange={handleInputChange}
                    placeholder="John"
                    className="bg-gray-700/50 border-gray-600 text-white placeholder-gray-400"
                    required
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="lastName" className="text-gray-300">Last Name</Label>
                  <Input
                    id="lastName"
                    name="lastName"
                    type="text"
                    value={formData.lastName}
                    onChange={handleInputChange}
                    placeholder="Doe"
                    className="bg-gray-700/50 border-gray-600 text-white placeholder-gray-400"
                    required
                  />
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="email" className="text-gray-300">Email</Label>
                <Input
                  id="email"
                  name="email"
                  type="email"
                  value={formData.email}
                  onChange={handleInputChange}
                  placeholder="john@example.com"
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
                    placeholder="At least 8 characters"
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

              <div className="space-y-2">
                <Label htmlFor="confirmPassword" className="text-gray-300">Confirm Password</Label>
                <div className="relative">
                  <Input
                    id="confirmPassword"
                    name="confirmPassword"
                    type={showConfirmPassword ? 'text' : 'password'}
                    value={formData.confirmPassword}
                    onChange={handleInputChange}
                    placeholder="Confirm your password"
                    className="bg-gray-700/50 border-gray-600 text-white placeholder-gray-400 pr-10"
                    data-testid="confirm-password-input"
                    required
                  />
                  <button
                    type="button"
                    onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                    className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-300"
                  >
                    {showConfirmPassword ? (
                      <EyeSlashIcon className="h-4 w-4" />
                    ) : (
                      <EyeIcon className="h-4 w-4" />
                    )}
                  </button>
                </div>
              </div>

              <div className="space-y-3">
                <div className="flex items-start space-x-2">
                  <input
                    id="acceptTerms"
                    name="acceptTerms"
                    type="checkbox"
                    checked={formData.acceptTerms}
                    onChange={handleInputChange}
                    className="mt-1 rounded border-gray-600 bg-gray-700 text-blue-600 focus:ring-blue-500"
                    required
                  />
                  <Label htmlFor="acceptTerms" className="text-sm text-gray-300">
                    I agree to the{' '}
                    <Link href="/terms" className="text-blue-400 hover:text-blue-300">
                      Terms of Service
                    </Link>
                  </Label>
                </div>

                <div className="flex items-start space-x-2">
                  <input
                    id="acceptPrivacy"
                    name="acceptPrivacy"
                    type="checkbox"
                    checked={formData.acceptPrivacy}
                    onChange={handleInputChange}
                    className="mt-1 rounded border-gray-600 bg-gray-700 text-blue-600 focus:ring-blue-500"
                    required
                  />
                  <Label htmlFor="acceptPrivacy" className="text-sm text-gray-300">
                    I agree to the{' '}
                    <Link href="/privacy" className="text-blue-400 hover:text-blue-300">
                      Privacy Policy
                    </Link>
                  </Label>
                </div>

                <div className="flex items-start space-x-2">
                  <input
                    id="marketingEmails"
                    name="marketingEmails"
                    type="checkbox"
                    checked={formData.marketingEmails}
                    onChange={handleInputChange}
                    className="mt-1 rounded border-gray-600 bg-gray-700 text-blue-600 focus:ring-blue-500"
                  />
                  <Label htmlFor="marketingEmails" className="text-sm text-gray-300">
                    I want to receive marketing emails and updates
                  </Label>
                </div>
              </div>

              <Button
                type="submit"
                disabled={isLoading}
                className="w-full bg-blue-600 hover:bg-blue-700 text-white"
                data-testid="register-button"
              >
                {isLoading ? 'Creating Account...' : 'Create Account'}
              </Button>
            </form>

            <div className="mt-6 text-center">
              <p className="text-gray-400">
                Already have an account?{' '}
                <Link 
                  href="/auth/login" 
                  className="text-blue-400 hover:text-blue-300 font-medium"
                >
                  Sign in
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