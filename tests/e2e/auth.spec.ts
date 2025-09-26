import { test, expect } from '@playwright/test';

test.describe('Authentication Flow', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should display login form on homepage', async ({ page }) => {
    // Check if login form is visible
    await expect(page.locator('[data-testid="login-form"]')).toBeVisible();
    await expect(page.locator('[data-testid="email-input"]')).toBeVisible();
    await expect(page.locator('[data-testid="password-input"]')).toBeVisible();
    await expect(page.locator('[data-testid="login-button"]')).toBeVisible();
  });

  test('should show validation errors for empty form submission', async ({ page }) => {
    // Click login button without filling form
    await page.click('[data-testid="login-button"]');
    
    // Check for validation errors
    await expect(page.locator('[data-testid="email-error"]')).toBeVisible();
    await expect(page.locator('[data-testid="password-error"]')).toBeVisible();
  });

  test('should show error for invalid credentials', async ({ page }) => {
    // Fill form with invalid credentials
    await page.fill('[data-testid="email-input"]', 'invalid@example.com');
    await page.fill('[data-testid="password-input"]', 'wrongpassword');
    await page.click('[data-testid="login-button"]');
    
    // Check for error message
    await expect(page.locator('[data-testid="login-error"]')).toBeVisible();
    await expect(page.locator('[data-testid="login-error"]')).toContainText('Invalid credentials');
  });

  test('should successfully login with valid credentials', async ({ page }) => {
    // Fill form with valid credentials
    await page.fill('[data-testid="email-input"]', 'test@example.com');
    await page.fill('[data-testid="password-input"]', 'testpassword');
    await page.click('[data-testid="login-button"]');
    
    // Wait for redirect to dashboard
    await page.waitForURL('/dashboard');
    
    // Check if dashboard is loaded
    await expect(page.locator('[data-testid="dashboard"]')).toBeVisible();
    await expect(page.locator('[data-testid="user-menu"]')).toBeVisible();
  });

  test('should navigate to registration form', async ({ page }) => {
    // Click register link
    await page.click('[data-testid="register-link"]');
    
    // Check if registration form is visible
    await expect(page.locator('[data-testid="register-form"]')).toBeVisible();
    await expect(page.locator('[data-testid="username-input"]')).toBeVisible();
    await expect(page.locator('[data-testid="email-input"]')).toBeVisible();
    await expect(page.locator('[data-testid="password-input"]')).toBeVisible();
    await expect(page.locator('[data-testid="confirm-password-input"]')).toBeVisible();
  });

  test('should successfully register new user', async ({ page }) => {
    // Navigate to registration
    await page.click('[data-testid="register-link"]');
    
    // Fill registration form
    await page.fill('[data-testid="username-input"]', 'newuser');
    await page.fill('[data-testid="email-input"]', 'newuser@example.com');
    await page.fill('[data-testid="password-input"]', 'newpassword');
    await page.fill('[data-testid="confirm-password-input"]', 'newpassword');
    await page.click('[data-testid="register-button"]');
    
    // Check for success message or redirect
    await expect(page.locator('[data-testid="registration-success"]')).toBeVisible();
  });

  test('should logout successfully', async ({ page }) => {
    // Login first
    await page.fill('[data-testid="email-input"]', 'test@example.com');
    await page.fill('[data-testid="password-input"]', 'testpassword');
    await page.click('[data-testid="login-button"]');
    
    // Wait for dashboard
    await page.waitForURL('/dashboard');
    
    // Click user menu and logout
    await page.click('[data-testid="user-menu"]');
    await page.click('[data-testid="logout-button"]');
    
    // Check if redirected to login page
    await page.waitForURL('/');
    await expect(page.locator('[data-testid="login-form"]')).toBeVisible();
  });

  test('should handle password reset flow', async ({ page }) => {
    // Click forgot password link
    await page.click('[data-testid="forgot-password-link"]');
    
    // Fill email for password reset
    await page.fill('[data-testid="reset-email-input"]', 'test@example.com');
    await page.click('[data-testid="reset-password-button"]');
    
    // Check for success message
    await expect(page.locator('[data-testid="reset-success"]')).toBeVisible();
    await expect(page.locator('[data-testid="reset-success"]')).toContainText('Password reset email sent');
  });
});