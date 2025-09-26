import type { Metadata, Viewport } from "next";
import { Inter } from "next/font/google";
import { AuthProvider } from "@/contexts/AuthContext";
import { SubscriptionProvider } from "@/contexts/SubscriptionContext";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#1f2937",
};

export const metadata: Metadata = {
  title: "FinScope - Advanced Financial Analytics & Trading Platform",
  description: "Professional-grade financial analytics, portfolio management, and algorithmic trading platform with real-time market data and AI-powered insights.",
  keywords: [
    "financial analytics",
    "trading platform",
    "portfolio management",
    "algorithmic trading",
    "market data",
    "investment tools",
    "financial technology",
    "fintech"
  ],
  authors: [{ name: "FinScope Team" }],
  robots: "index, follow",
  openGraph: {
    title: "FinScope - Advanced Financial Analytics & Trading Platform",
    description: "Professional-grade financial analytics, portfolio management, and algorithmic trading platform with real-time market data and AI-powered insights.",
    type: "website",
    locale: "en_US",
  },
  twitter: {
    card: "summary_large_image",
    title: "FinScope - Advanced Financial Analytics & Trading Platform",
    description: "Professional-grade financial analytics, portfolio management, and algorithmic trading platform with real-time market data and AI-powered insights.",
  },
  icons: {
    icon: "/favicon.ico",
    apple: "/apple-touch-icon.png",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className={inter.className}>
        <AuthProvider>
          <SubscriptionProvider>
            {children}
          </SubscriptionProvider>
        </AuthProvider>
      </body>
    </html>
  );
}