import { NextRequest, NextResponse } from 'next/server';
import { jwtVerify } from 'jose';

interface UserPayload {
  user_id: string;
  email: string;
  subscription_plan: 'free' | 'basic' | 'premium';
  subscription_status: 'active' | 'trial' | 'expired' | 'cancelled';
  exp: number;
}

// Define protected routes and their required subscription levels
// Only include routes that require paid subscriptions (basic/premium)
const PROTECTED_ROUTES = {
  '/dashboard/analytics': 'basic',
  '/dashboard/ai-insights': 'basic',
  '/dashboard/social-trading': 'basic',
  '/dashboard/institutional': 'premium',
  '/dashboard/advanced-charts': 'basic',
  '/dashboard/portfolio/advanced': 'basic',
  '/dashboard/watchlist/premium': 'premium',
  '/api/analytics': 'basic',
  '/api/ai-insights': 'basic',
  '/api/social-trading': 'basic',
  '/api/institutional': 'premium',
} as const;

// Features that require specific subscription levels
const FEATURE_ACCESS = {
  'real_time_data': 'basic',
  'advanced_charts': 'basic',
  'price_alerts': 'basic',
  'technical_indicators': 'basic',
  'ai_insights': 'basic',
  'social_trading': 'basic',
  'institutional_tools': 'premium',
  'priority_support': 'premium',
  'unlimited_portfolios': 'premium',
  'unlimited_watchlists': 'premium',
} as const;

export async function subscriptionMiddleware(request: NextRequest): Promise<NextResponse> {
  const { pathname } = request.nextUrl;
  
  // Check if the route requires subscription access
  const requiredPlan = getRequiredPlanForRoute(pathname);
  if (!requiredPlan) {
    return NextResponse.next();
  }

  // Get the access token from cookies or headers
  const token = request.cookies.get('access_token')?.value || 
                request.headers.get('authorization')?.replace('Bearer ', '');

  if (!token) {
    return createUnauthorizedResponse('Authentication required');
  }

  try {
    // Verify and decode the JWT token
    const secret = new TextEncoder().encode(process.env.JWT_SECRET || 'your-secret-key-here-change-in-production');
    const { payload } = await jwtVerify(token, secret) as { payload: UserPayload };

    // Check if user has required subscription level
    if (!hasSubscriptionAccess(payload, requiredPlan)) {
      return createSubscriptionRequiredResponse(requiredPlan, pathname);
    }

    // Add user subscription info to headers for downstream use
    const requestHeaders = new Headers(request.headers);
    requestHeaders.set('x-user-subscription-plan', payload.subscription_plan);
    requestHeaders.set('x-user-subscription-status', payload.subscription_status);
    requestHeaders.set('x-user-id', payload.user_id);

    return NextResponse.next({
      request: {
        headers: requestHeaders,
      },
    });
  } catch (error) {
    console.error('Subscription middleware error:', error);
    return createUnauthorizedResponse('Invalid token');
  }
}

function getRequiredPlanForRoute(pathname: string): 'free' | 'basic' | 'premium' | null {
  // Check exact matches first
  if (pathname in PROTECTED_ROUTES) {
    return PROTECTED_ROUTES[pathname as keyof typeof PROTECTED_ROUTES] as 'free' | 'basic' | 'premium';
  }

  // Check for partial matches (e.g., API routes with parameters)
  for (const [route, plan] of Object.entries(PROTECTED_ROUTES)) {
    if (pathname.startsWith(route)) {
      return plan as 'free' | 'basic' | 'premium';
    }
  }

  return null;
}

function hasSubscriptionAccess(
  user: UserPayload, 
  requiredPlan: 'free' | 'basic' | 'premium'
): boolean {
  // Check if subscription is active
  if (user.subscription_status === 'expired' || user.subscription_status === 'cancelled') {
    return user.subscription_plan === 'free' && requiredPlan === 'free';
  }

  const planHierarchy = { free: 0, basic: 1, premium: 2 };
  const userPlanLevel = planHierarchy[user.subscription_plan];
  const requiredPlanLevel = planHierarchy[requiredPlan];

  return userPlanLevel >= requiredPlanLevel;
}

function createUnauthorizedResponse(message: string) {
  return new NextResponse(
    JSON.stringify({ error: message, code: 'UNAUTHORIZED' }),
    {
      status: 401,
      headers: {
        'Content-Type': 'application/json',
        'Cache-Control': 'no-cache, no-store, must-revalidate',
      },
    }
  );
}

function createSubscriptionRequiredResponse(requiredPlan: string, pathname: string) {
  return new NextResponse(
    JSON.stringify({
      error: 'Subscription upgrade required',
      code: 'SUBSCRIPTION_REQUIRED',
      required_plan: requiredPlan,
      current_path: pathname,
      upgrade_url: '/pricing',
    }),
    {
      status: 403,
      headers: {
        'Content-Type': 'application/json',
        'Cache-Control': 'no-cache, no-store, must-revalidate',
      },
    }
  );
}

// Helper function to check feature access
export function checkFeatureAccess(
  userPlan: 'free' | 'basic' | 'premium',
  userStatus: 'active' | 'trial' | 'expired' | 'cancelled',
  feature: keyof typeof FEATURE_ACCESS
): boolean {
  if (userStatus === 'expired' || userStatus === 'cancelled') {
    return userPlan === 'free' && !FEATURE_ACCESS[feature];
  }

  const requiredPlan = FEATURE_ACCESS[feature];
  const planHierarchy = { free: 0, basic: 1, premium: 2 };
  const userPlanLevel = planHierarchy[userPlan];
  const requiredPlanLevel = planHierarchy[requiredPlan];

  return userPlanLevel >= requiredPlanLevel;
}

export { PROTECTED_ROUTES, FEATURE_ACCESS };