import { test, expect } from '@playwright/test';

test.describe('Dashboard Functionality', () => {
  test.beforeEach(async ({ page }) => {
    // Login before each test
    await page.goto('/');
    await page.fill('[data-testid="email-input"]', 'test@example.com');
    await page.fill('[data-testid="password-input"]', 'testpassword');
    await page.click('[data-testid="login-button"]');
    await page.waitForURL('/dashboard');
  });

  test('should display all five main areas', async ({ page }) => {
    // Check if all five areas are visible
    await expect(page.locator('[data-testid="hub-area"]')).toBeVisible();
    await expect(page.locator('[data-testid="info-terminal-area"]')).toBeVisible();
    await expect(page.locator('[data-testid="depth-analysis-area"]')).toBeVisible();
    await expect(page.locator('[data-testid="chat-area"]')).toBeVisible();
    await expect(page.locator('[data-testid="archive-area"]')).toBeVisible();
  });

  test('should navigate between areas using navigation', async ({ page }) => {
    // Test navigation to each area
    await page.click('[data-testid="nav-hub"]');
    await expect(page.locator('[data-testid="hub-area"]')).toBeVisible();
    
    await page.click('[data-testid="nav-info-terminal"]');
    await expect(page.locator('[data-testid="info-terminal-area"]')).toBeVisible();
    
    await page.click('[data-testid="nav-depth-analysis"]');
    await expect(page.locator('[data-testid="depth-analysis-area"]')).toBeVisible();
    
    await page.click('[data-testid="nav-chat"]');
    await expect(page.locator('[data-testid="chat-area"]')).toBeVisible();
    
    await page.click('[data-testid="nav-archive"]');
    await expect(page.locator('[data-testid="archive-area"]')).toBeVisible();
  });

  test('should display watchlist in hub area', async ({ page }) => {
    // Navigate to hub area
    await page.click('[data-testid="nav-hub"]');
    
    // Check if watchlist is visible
    await expect(page.locator('[data-testid="watchlist"]')).toBeVisible();
    await expect(page.locator('[data-testid="watchlist-item"]').first()).toBeVisible();
  });

  test('should open chart when clicking on watchlist item', async ({ page }) => {
    // Navigate to hub area
    await page.click('[data-testid="nav-hub"]');
    
    // Click on first watchlist item
    await page.click('[data-testid="watchlist-item"]').first();
    
    // Check if chart is displayed
    await expect(page.locator('[data-testid="trading-chart"]')).toBeVisible();
    await expect(page.locator('[data-testid="chart-canvas"]')).toBeVisible();
  });

  test('should display AI insights area', async ({ page }) => {
    // Navigate to hub area
    await page.click('[data-testid="nav-hub"]');
    
    // Check if AI insights area is visible
    await expect(page.locator('[data-testid="ai-insights"]')).toBeVisible();
    await expect(page.locator('[data-testid="ai-predictions"]')).toBeVisible();
  });

  test('should allow adding indicators to chart', async ({ page }) => {
    // Navigate to hub area and open chart
    await page.click('[data-testid="nav-hub"]');
    await page.click('[data-testid="watchlist-item"]').first();
    
    // Click add indicator button
    await page.click('[data-testid="add-indicator-button"]');
    
    // Select an indicator
    await page.click('[data-testid="indicator-sma"]');
    
    // Check if indicator is added to chart
    await expect(page.locator('[data-testid="chart-indicator-sma"]')).toBeVisible();
  });

  test('should display news feed in info terminal', async ({ page }) => {
    // Navigate to info terminal area
    await page.click('[data-testid="nav-info-terminal"]');
    
    // Check if news feed is visible
    await expect(page.locator('[data-testid="news-feed"]')).toBeVisible();
    await expect(page.locator('[data-testid="news-item"]').first()).toBeVisible();
  });

  test('should open news article in full view', async ({ page }) => {
    // Navigate to info terminal area
    await page.click('[data-testid="nav-info-terminal"]');
    
    // Click on first news item
    await page.click('[data-testid="news-item"]').first();
    
    // Check if news article modal is opened
    await expect(page.locator('[data-testid="news-modal"]')).toBeVisible();
    await expect(page.locator('[data-testid="news-content"]')).toBeVisible();
  });

  test('should display order flow data in depth analysis', async ({ page }) => {
    // Navigate to depth analysis area
    await page.click('[data-testid="nav-depth-analysis"]');
    
    // Check if order flow data is visible
    await expect(page.locator('[data-testid="order-flow"]')).toBeVisible();
    await expect(page.locator('[data-testid="volume-analysis"]')).toBeVisible();
    await expect(page.locator('[data-testid="market-depth"]')).toBeVisible();
  });

  test('should allow sending messages in chat area', async ({ page }) => {
    // Navigate to chat area
    await page.click('[data-testid="nav-chat"]');
    
    // Type and send a message
    await page.fill('[data-testid="chat-input"]', 'Test message');
    await page.click('[data-testid="send-message-button"]');
    
    // Check if message appears in chat
    await expect(page.locator('[data-testid="chat-message"]').last()).toContainText('Test message');
  });

  test('should allow uploading images in chat', async ({ page }) => {
    // Navigate to chat area
    await page.click('[data-testid="nav-chat"]');
    
    // Click upload image button
    await page.click('[data-testid="upload-image-button"]');
    
    // Check if file input is triggered (mock file upload)
    await expect(page.locator('[data-testid="file-input"]')).toBeVisible();
  });

  test('should display trading journal in archive area', async ({ page }) => {
    // Navigate to archive area
    await page.click('[data-testid="nav-archive"]');
    
    // Check if trading journal is visible
    await expect(page.locator('[data-testid="trading-journal"]')).toBeVisible();
    await expect(page.locator('[data-testid="journal-entries"]')).toBeVisible();
  });

  test('should allow creating new journal entry', async ({ page }) => {
    // Navigate to archive area
    await page.click('[data-testid="nav-archive"]');
    
    // Click new entry button
    await page.click('[data-testid="new-journal-entry"]');
    
    // Fill journal entry form
    await page.fill('[data-testid="journal-title"]', 'Test Trade Entry');
    await page.fill('[data-testid="journal-content"]', 'This is a test trading journal entry.');
    await page.click('[data-testid="save-journal-entry"]');
    
    // Check if entry is saved
    await expect(page.locator('[data-testid="journal-entry"]').last()).toContainText('Test Trade Entry');
  });

  test('should display live news feed for current chart symbol', async ({ page }) => {
    // Navigate to hub area and select a symbol
    await page.click('[data-testid="nav-hub"]');
    await page.click('[data-testid="watchlist-item"]').first();
    
    // Check if live news feed is visible for the selected symbol
    await expect(page.locator('[data-testid="live-news-feed"]')).toBeVisible();
    await expect(page.locator('[data-testid="symbol-specific-news"]')).toBeVisible();
  });

  test('should change chart timeframes', async ({ page }) => {
    // Navigate to hub area and open chart
    await page.click('[data-testid="nav-hub"]');
    await page.click('[data-testid="watchlist-item"]').first();
    
    // Test different timeframes
    const timeframes = ['1m', '5m', '15m', '30m', '1h', '2h', '6h', '12h', '1d', '1w', '1M'];
    
    for (const timeframe of timeframes.slice(0, 3)) { // Test first 3 for speed
      await page.click(`[data-testid="timeframe-${timeframe}"]`);
      await expect(page.locator('[data-testid="active-timeframe"]')).toContainText(timeframe);
    }
  });
});