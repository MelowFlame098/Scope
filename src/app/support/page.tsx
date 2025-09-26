"use client";

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  ChartBarIcon,
  QuestionMarkCircleIcon,
  ChatBubbleLeftRightIcon,
  PhoneIcon,
  EnvelopeIcon,
  ClockIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  ArrowLeftIcon
} from '@heroicons/react/24/outline';

export default function SupportPage() {
  const router = useRouter();
  const [activeTab, setActiveTab] = useState('contact');
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    subject: '',
    message: '',
    priority: 'medium'
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    
    // Simulate form submission
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    setSubmitted(true);
    setIsSubmitting(false);
  };

  const faqItems = [
    {
      question: "How do I get started with trading?",
      answer: "Start by creating an account, completing your profile, and exploring our paper trading feature to practice without real money."
    },
    {
      question: "What are the subscription plans?",
      answer: "We offer Basic (Free), Pro ($29/mo), and Enterprise ($99/mo) plans with different features and capabilities."
    },
    {
      question: "How secure is my data?",
      answer: "We use bank-level encryption, two-factor authentication, and comply with all financial data protection regulations."
    },
    {
      question: "Can I cancel my subscription anytime?",
      answer: "Yes, you can cancel your subscription at any time from your account settings. No cancellation fees apply."
    },
    {
      question: "How do AI insights work?",
      answer: "Our AI analyzes market data, news sentiment, and technical indicators to provide personalized trading insights and predictions."
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900">
      {/* Header */}
      <header className="relative z-10 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <Button 
              variant="ghost" 
              onClick={() => router.back()}
              className="text-white hover:text-blue-400"
            >
              <ArrowLeftIcon className="h-5 w-5 mr-2" />
              Back
            </Button>
            <div className="flex items-center space-x-2">
              <ChartBarIcon className="h-8 w-8 text-blue-400" />
              <span className="text-2xl font-bold text-white">FinScope Support</span>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-6xl mx-auto px-6 py-12">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-white mb-4">
            How can we help you?
          </h1>
          <p className="text-xl text-gray-300">
            Get the support you need to make the most of FinScope
          </p>
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-4 mb-8">
            <TabsTrigger value="contact">Contact Us</TabsTrigger>
            <TabsTrigger value="faq">FAQ</TabsTrigger>
            <TabsTrigger value="resources">Resources</TabsTrigger>
            <TabsTrigger value="status">System Status</TabsTrigger>
          </TabsList>

          <TabsContent value="contact" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Contact Form */}
              <Card className="bg-gray-800/50 border-gray-700">
                <CardHeader>
                  <CardTitle className="text-white flex items-center">
                    <ChatBubbleLeftRightIcon className="h-6 w-6 mr-2" />
                    Send us a message
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {submitted ? (
                    <Alert className="border-green-500 bg-green-500/10">
                      <CheckCircleIcon className="h-4 w-4 text-green-500" />
                      <AlertDescription className="text-green-400">
                        Thank you for your message! We'll get back to you within 24 hours.
                      </AlertDescription>
                    </Alert>
                  ) : (
                    <form onSubmit={handleSubmit} className="space-y-4">
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <Label htmlFor="name" className="text-gray-300">Name</Label>
                          <Input
                            id="name"
                            name="name"
                            value={formData.name}
                            onChange={handleInputChange}
                            required
                            className="bg-gray-700 border-gray-600 text-white"
                          />
                        </div>
                        <div>
                          <Label htmlFor="email" className="text-gray-300">Email</Label>
                          <Input
                            id="email"
                            name="email"
                            type="email"
                            value={formData.email}
                            onChange={handleInputChange}
                            required
                            className="bg-gray-700 border-gray-600 text-white"
                          />
                        </div>
                      </div>
                      <div>
                        <Label htmlFor="subject" className="text-gray-300">Subject</Label>
                        <Input
                          id="subject"
                          name="subject"
                          value={formData.subject}
                          onChange={handleInputChange}
                          required
                          className="bg-gray-700 border-gray-600 text-white"
                        />
                      </div>
                      <div>
                        <Label htmlFor="message" className="text-gray-300">Message</Label>
                        <Textarea
                          id="message"
                          name="message"
                          value={formData.message}
                          onChange={handleInputChange}
                          required
                          rows={5}
                          className="bg-gray-700 border-gray-600 text-white"
                        />
                      </div>
                      <Button 
                        type="submit" 
                        disabled={isSubmitting}
                        className="w-full bg-blue-600 hover:bg-blue-700"
                      >
                        {isSubmitting ? 'Sending...' : 'Send Message'}
                      </Button>
                    </form>
                  )}
                </CardContent>
              </Card>

              {/* Contact Information */}
              <div className="space-y-6">
                <Card className="bg-gray-800/50 border-gray-700">
                  <CardHeader>
                    <CardTitle className="text-white">Contact Information</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="flex items-center space-x-3">
                      <EnvelopeIcon className="h-5 w-5 text-blue-400" />
                      <span className="text-gray-300">support@finscope.com</span>
                    </div>
                    <div className="flex items-center space-x-3">
                      <PhoneIcon className="h-5 w-5 text-blue-400" />
                      <span className="text-gray-300">+1 (555) 123-4567</span>
                    </div>
                    <div className="flex items-center space-x-3">
                      <ClockIcon className="h-5 w-5 text-blue-400" />
                      <span className="text-gray-300">Mon-Fri 9AM-6PM EST</span>
                    </div>
                  </CardContent>
                </Card>

                <Card className="bg-gray-800/50 border-gray-700">
                  <CardHeader>
                    <CardTitle className="text-white">Response Times</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-gray-300">General Inquiries</span>
                      <Badge variant="secondary">24 hours</Badge>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-300">Technical Issues</span>
                      <Badge variant="secondary">4 hours</Badge>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-300">Account Issues</span>
                      <Badge variant="secondary">2 hours</Badge>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="faq" className="space-y-6">
            <div className="grid gap-4">
              {faqItems.map((item, index) => (
                <Card key={index} className="bg-gray-800/50 border-gray-700">
                  <CardHeader>
                    <CardTitle className="text-white flex items-center">
                      <QuestionMarkCircleIcon className="h-5 w-5 mr-2 text-blue-400" />
                      {item.question}
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-gray-300">{item.answer}</p>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          <TabsContent value="resources" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              <Card className="bg-gray-800/50 border-gray-700">
                <CardHeader>
                  <CardTitle className="text-white">Getting Started Guide</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-gray-300 mb-4">
                    Learn the basics of using FinScope for trading and investment analysis.
                  </p>
                  <Button variant="outline" className="w-full">
                    View Guide
                  </Button>
                </CardContent>
              </Card>

              <Card className="bg-gray-800/50 border-gray-700">
                <CardHeader>
                  <CardTitle className="text-white">API Documentation</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-gray-300 mb-4">
                    Integrate FinScope data and features into your own applications.
                  </p>
                  <Button variant="outline" className="w-full">
                    View Docs
                  </Button>
                </CardContent>
              </Card>

              <Card className="bg-gray-800/50 border-gray-700">
                <CardHeader>
                  <CardTitle className="text-white">Video Tutorials</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-gray-300 mb-4">
                    Watch step-by-step tutorials on using advanced features.
                  </p>
                  <Button variant="outline" className="w-full">
                    Watch Videos
                  </Button>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="status" className="space-y-6">
            <Card className="bg-gray-800/50 border-gray-700">
              <CardHeader>
                <CardTitle className="text-white">System Status</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-gray-300">Trading Platform</span>
                  <Badge className="bg-green-500 text-white">Operational</Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-300">Market Data Feed</span>
                  <Badge className="bg-green-500 text-white">Operational</Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-300">AI Insights</span>
                  <Badge className="bg-green-500 text-white">Operational</Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-300">User Authentication</span>
                  <Badge className="bg-green-500 text-white">Operational</Badge>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}