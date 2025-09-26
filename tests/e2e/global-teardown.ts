import { FullConfig } from '@playwright/test';

async function globalTeardown(config: FullConfig) {
  console.log('🧹 Starting global teardown for E2E tests...');
  
  try {
    // Clean up test data
    console.log('🗑️ Cleaning up test data...');
    
    // You can add cleanup logic here
    // For example, delete test users, reset database state, etc.
    
    // Clean up authentication files
    const fs = require('fs');
    const path = require('path');
    
    const authFile = path.join(__dirname, 'auth.json');
    if (fs.existsSync(authFile)) {
      fs.unlinkSync(authFile);
      console.log('🔐 Cleaned up authentication state');
    }
    
    console.log('✅ Global teardown completed successfully');
    
  } catch (error) {
    console.error('❌ Global teardown failed:', error);
    // Don't throw error in teardown to avoid masking test failures
  }
}

export default globalTeardown;