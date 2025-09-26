const lighthouse = require('lighthouse');
const chromeLauncher = require('chrome-launcher');
const fs = require('fs');
const path = require('path');

async function runPerformanceAudit() {
  console.log('🚀 Starting performance audit...');
  
  // Launch Chrome
  const chrome = await chromeLauncher.launch({
    chromeFlags: ['--headless', '--no-sandbox', '--disable-gpu']
  });
  
  const options = {
    logLevel: 'info',
    output: 'html',
    onlyCategories: ['performance', 'accessibility', 'best-practices', 'seo'],
    port: chrome.port,
  };
  
  try {
    // Run Lighthouse audit
    console.log('📊 Running Lighthouse audit...');
    const runnerResult = await lighthouse('http://localhost:3000', options);
    
    // Generate report
    const reportHtml = runnerResult.report;
    const reportPath = path.join(__dirname, '../reports/lighthouse-report.html');
    
    // Ensure reports directory exists
    const reportsDir = path.dirname(reportPath);
    if (!fs.existsSync(reportsDir)) {
      fs.mkdirSync(reportsDir, { recursive: true });
    }
    
    // Save report
    fs.writeFileSync(reportPath, reportHtml);
    console.log(`📋 Report saved to: ${reportPath}`);
    
    // Extract scores
    const scores = runnerResult.lhr.categories;
    console.log('\n📈 Performance Scores:');
    console.log(`Performance: ${Math.round(scores.performance.score * 100)}/100`);
    console.log(`Accessibility: ${Math.round(scores.accessibility.score * 100)}/100`);
    console.log(`Best Practices: ${Math.round(scores['best-practices'].score * 100)}/100`);
    console.log(`SEO: ${Math.round(scores.seo.score * 100)}/100`);
    
    // Check for performance issues
    const performanceScore = scores.performance.score * 100;
    if (performanceScore < 90) {
      console.log('\n⚠️  Performance issues detected:');
      const audits = runnerResult.lhr.audits;
      
      // Check specific metrics
      if (audits['first-contentful-paint'].score < 0.9) {
        console.log(`- First Contentful Paint: ${audits['first-contentful-paint'].displayValue}`);
      }
      if (audits['largest-contentful-paint'].score < 0.9) {
        console.log(`- Largest Contentful Paint: ${audits['largest-contentful-paint'].displayValue}`);
      }
      if (audits['cumulative-layout-shift'].score < 0.9) {
        console.log(`- Cumulative Layout Shift: ${audits['cumulative-layout-shift'].displayValue}`);
      }
      if (audits['total-blocking-time'].score < 0.9) {
        console.log(`- Total Blocking Time: ${audits['total-blocking-time'].displayValue}`);
      }
    } else {
      console.log('\n✅ Performance is excellent!');
    }
    
    // Generate JSON report for CI/CD
    const jsonReport = {
      timestamp: new Date().toISOString(),
      url: 'http://localhost:3000',
      scores: {
        performance: Math.round(scores.performance.score * 100),
        accessibility: Math.round(scores.accessibility.score * 100),
        bestPractices: Math.round(scores['best-practices'].score * 100),
        seo: Math.round(scores.seo.score * 100),
      },
      metrics: {
        firstContentfulPaint: audits['first-contentful-paint'].numericValue,
        largestContentfulPaint: audits['largest-contentful-paint'].numericValue,
        cumulativeLayoutShift: audits['cumulative-layout-shift'].numericValue,
        totalBlockingTime: audits['total-blocking-time'].numericValue,
      }
    };
    
    const jsonReportPath = path.join(__dirname, '../reports/lighthouse-report.json');
    fs.writeFileSync(jsonReportPath, JSON.stringify(jsonReport, null, 2));
    console.log(`📊 JSON report saved to: ${jsonReportPath}`);
    
  } catch (error) {
    console.error('❌ Performance audit failed:', error);
    process.exit(1);
  } finally {
    await chrome.kill();
  }
}

// Run the audit
if (require.main === module) {
  runPerformanceAudit().catch(console.error);
}

module.exports = { runPerformanceAudit };