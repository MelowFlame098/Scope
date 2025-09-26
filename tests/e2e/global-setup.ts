import { chromium, FullConfig } from '@playwright/test';

async function globalSetup(config: FullConfig) {
  console.log('🚀 Starting global setup for E2E tests...');
  
  // Launch browser for setup
  const browser = await chromium.launch();
  const page = await browser.newPage();
  
  try {
    // Wait for the application to be ready
    console.log('⏳ Waiting for application to be ready...');
    await page.goto('http://localhost:3000', { waitUntil: 'networkidle' });
    
    // Setup test user authentication if needed
    console.log('👤 Setting up test user authentication...');
    
    // You can add authentication setup here
    // For example, login with test credentials and save auth state
    /*
    await page.fill('[data-testid="email"]', 'test@example.com');
    await page.fill('[data-testid="password"]', 'testpassword');
    await page.click('[data-testid="login-button"]');
    
    // Wait for successful login
    await page.waitForSelector('[data-testid="dashboard"]');
    
    // Save authentication state
    await page.context().storageState({ path: 'tests/e2e/auth.json' });
    */
    
    console.log('✅ Global setup completed successfully');
    
  } catch (error) {
    console.error('❌ Global setup failed:', error);
    throw error;
  } finally {
    await browser.close();
  }
}

export default globalSetup;