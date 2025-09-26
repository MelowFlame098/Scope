import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';
import { subscriptionMiddleware } from './middleware/subscription';
import { sessionService } from './services/SessionService';

// Define protected routes that require authentication
const protectedRoutes = [
  '/dashboard',
  '/portfolio',
  '/trading',
  '/analytics',
  '/settings',
  '/profile'
];

// Define routes that require authentication but not subscription checks
const freeProtectedRoutes = [
  '/dashboard',
  '/portfolio',
  '/settings',
  '/profile'
];

// Define public routes that don't require authentication
const publicRoutes = [
  '/',
  '/landing',
  '/pricing',
  '/demo',
  '/contact',
  '/about',
  '/features',
  '/blog',
  '/careers',
  '/help-center',
  '/documentation',
  '/security',
  '/status',
  '/tutorial',
  '/auth/login',
  '/auth/register',
  '/auth/forgot-password',
  '/auth/reset-password',
  '/auth/verify-email',
  '/legal/terms',
  '/legal/privacy',
  '/support',
  '/api/auth/captcha',
  '/api/auth/register',
  '/api/auth/login',
  '/api/auth/password-reset',
  '/api/auth/verify-email',
  '/api/auth/reset-password-confirm'
];

// Define auth routes that authenticated users shouldn't access
const authRoutes = [
  '/auth/login',
  '/auth/register',
  '/auth/forgot-password'
];

export async function middleware(request: NextRequest) {
  const pathname = request.nextUrl.pathname;
  const fullUrl = request.url;
  
  // Immediately bypass for RSC requests, static files, and API routes
  const url = new URL(request.url);
  const hasRscParam = url.searchParams.has('_rsc') || fullUrl.includes('_rsc=');
  
  if (hasRscParam || 
      fullUrl.includes('_next=') ||
      pathname.startsWith('/_next/') ||
      pathname.startsWith('/api/') ||
      pathname.startsWith('/static/') ||
      pathname.startsWith('/_vercel/') ||
      pathname.includes('.') || // files with extensions
      request.headers.get('RSC') === '1' ||
      request.headers.get('Next-Router-Prefetch') === '1' ||
      request.headers.get('accept')?.includes('text/x-component') ||
      request.headers.get('accept')?.includes('application/rsc') ||
      request.headers.get('next-router-state-tree') ||
      request.headers.get('next-url') ||
      request.headers.get('x-middleware-prefetch') ||
      request.headers.get('x-middleware-invoke') ||
      request.method === 'HEAD'
  ) {
    return NextResponse.next();
  }
  
  const token = request.cookies.get('access_token')?.value;
  const sessionId = request.cookies.get('session_id')?.value;
  
  // Validate token and session if they exist
  let isValidToken = false;
  let sessionData = null;
  
  if (token && sessionId) {
    try {
      // Validate JWT token
      const { jwtVerify } = await import('jose');
      const secret = new TextEncoder().encode(process.env.JWT_SECRET || 'your-secret-key-here-change-in-production');
      const { payload } = await jwtVerify(token, secret);
      
      // Validate Redis session
      sessionData = await sessionService.getSession(sessionId);
      
      if (sessionData && sessionData.userId === payload.sub) {
        isValidToken = true;
        
        // Update session activity for protected routes
        if (protectedRoutes.some(route => pathname.startsWith(route))) {
          await sessionService.updateActivity(sessionId);
        }
      }
    } catch (error) {
      // Token or session is invalid, clear them
      isValidToken = false;
      sessionData = null;
    }
  }
  
  // Handle root path
  if (pathname === '/') {
    if (isValidToken) {
      return NextResponse.redirect(new URL('/dashboard', request.url));
    } else {
      return NextResponse.redirect(new URL('/landing', request.url));
    }
  }

  // Check if the route is protected
  const isProtectedRoute = protectedRoutes.some(route => 
    pathname.startsWith(route)
  );
  
  // Check if the route is an auth route
  const isAuthRoute = authRoutes.some(route => 
    pathname.startsWith(route)
  );

  // Check if the route is public
  const isPublicRoute = publicRoutes.some(route => 
    pathname.startsWith(route)
  );

  // If user is authenticated and trying to access auth routes, redirect to dashboard
  if (isValidToken && isAuthRoute) {
    const redirectResponse = NextResponse.redirect(new URL('/dashboard', request.url));
    // Add headers to ensure clean redirect
    redirectResponse.headers.set('Cache-Control', 'no-cache, no-store, must-revalidate');
    redirectResponse.headers.set('Pragma', 'no-cache');
    redirectResponse.headers.set('Expires', '0');
    return redirectResponse;
  }

  // If user is not authenticated and trying to access protected routes, redirect to login
  if (!isValidToken && isProtectedRoute) {
    const loginUrl = new URL('/auth/login', request.url);
    loginUrl.searchParams.set('redirect', pathname);
    return NextResponse.redirect(loginUrl);
  }

  // For email verification and password reset, allow access regardless of auth status
  if (pathname.startsWith('/auth/verify-email') || pathname.startsWith('/auth/reset-password')) {
    return NextResponse.next();
  }

  // If route is not explicitly public or protected, and user is not authenticated,
  // redirect to landing page
  if (!isValidToken && !isPublicRoute && !isProtectedRoute) {
    return NextResponse.redirect(new URL('/landing', request.url));
  }

  // Check if route is free (requires auth but no subscription)
  const isFreeProtectedRoute = freeProtectedRoutes.some(route => 
    pathname.startsWith(route)
  );

  // If user is authenticated and accessing protected routes, check subscription access
  // Only apply subscription middleware to routes that require paid subscriptions
  if (isValidToken && isProtectedRoute && !isPublicRoute && !isFreeProtectedRoute) {
    const subscriptionResponse = await subscriptionMiddleware(request);
    
    // If subscription middleware returns a response (error), handle it
    if (subscriptionResponse.status === 403) {
      // Subscription upgrade required - redirect to pricing with context
      const pricingUrl = new URL('/pricing', request.url);
      pricingUrl.searchParams.set('upgrade', 'true');
      pricingUrl.searchParams.set('feature', pathname);
      return NextResponse.redirect(pricingUrl);
    } else if (subscriptionResponse.status === 401) {
      // Invalid token - redirect to login
      const loginUrl = new URL('/auth/login', request.url);
      loginUrl.searchParams.set('redirect', pathname);
      return NextResponse.redirect(loginUrl);
    }
    
    // If subscription check passed, return the response with headers
    if (subscriptionResponse.status === 200) {
      return subscriptionResponse;
    }
  }

  // Add headers to prevent caching of auth-related pages
  const response = NextResponse.next();
  if (pathname.startsWith('/auth/')) {
    response.headers.set('Cache-Control', 'no-cache, no-store, must-revalidate');
    response.headers.set('Pragma', 'no-cache');
    response.headers.set('Expires', '0');
  }

  return response;
}

// Configure which routes the middleware should run on
export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - api (API routes)
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     * - public folder files
     */
    '/((?!api|_next/static|_next/image|favicon.ico|public).*)',
  ],
};