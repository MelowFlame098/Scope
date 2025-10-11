pub struct SOPRResult {
    pub current_sopr: f64,
    pub sopr_trend: String,
    pub profit_loss_ratio: f64,
    pub market_sentiment: String,
    pub historical_sopr: Vec<f64>,
    pub sopr_ma: Vec<f64>,
}

pub fn compute_sopr(sopr_series: &[f64]) -> SOPRResult {
    let n = sopr_series.len();
    let historical = sopr_series.to_vec();
    let current = if n == 0 { 1.0 } else { sopr_series[n - 1] };
    let ma = rolling_mean(&historical, usize::min(7, n.max(1)));
    let current_ma = if ma.is_empty() { 1.0 } else { ma[ma.len() - 1] };
    let trend = calculate_trend(&historical);
    let sentiment = determine_market_sentiment(current, current_ma);
    let profit_loss_ratio = current; // approximation consistent with Python

    SOPRResult {
        current_sopr: current,
        sopr_trend: trend,
        profit_loss_ratio,
        market_sentiment: sentiment,
        historical_sopr: historical,
        sopr_ma: ma,
    }
}

fn rolling_mean(series: &[f64], window: usize) -> Vec<f64> {
    if series.is_empty() { return vec![]; }
    let mut out = Vec::with_capacity(series.len());
    for i in 0..series.len() {
        let start = if i + 1 >= window { i + 1 - window } else { 0 };
        let slice = &series[start..=i];
        let sum: f64 = slice.iter().copied().sum();
        out.push(sum / (slice.len() as f64));
    }
    out
}

fn calculate_trend(series: &[f64]) -> String {
    let window = 7usize;
    if series.len() < window {
        return "Insufficient Data".to_string();
    }
    let y: Vec<f64> = series[series.len() - window..].to_vec();
    let x: Vec<f64> = (0..window).map(|v| v as f64).collect();
    let (slope, r) = linreg_slope_r(&x, &y);
    if r.abs() < 0.3 {
        "Sideways".to_string()
    } else if slope > 0.0 {
        if r > 0.5 { "Rising".to_string() } else { "Weakly Rising".to_string() }
    } else {
        if r < -0.5 { "Falling".to_string() } else { "Weakly Falling".to_string() }
    }
}

fn linreg_slope_r(x: &[f64], y: &[f64]) -> (f64, f64) {
    // Simple linear regression: slope and correlation coefficient
    let n = x.len().min(y.len()) as f64;
    if n == 0.0 { return (0.0, 0.0); }
    let mean_x = mean(x);
    let mean_y = mean(y);
    let mut num = 0.0;
    let mut den_x = 0.0;
    let mut den_y = 0.0;
    for i in 0..(n as usize) {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        num += dx * dy;
        den_x += dx * dx;
        den_y += dy * dy;
    }
    let slope = if den_x == 0.0 { 0.0 } else { num / den_x };
    let r = if den_x == 0.0 || den_y == 0.0 { 0.0 } else { num / (den_x.sqrt() * den_y.sqrt()) };
    (slope, r)
}

fn determine_market_sentiment(sopr: f64, sopr_ma: f64) -> String {
    if sopr > 1.05 && sopr > sopr_ma {
        "Strong Greed - High Profit Taking".to_string()
    } else if sopr > 1.02 {
        "Greed - Moderate Profit Taking".to_string()
    } else if sopr > 0.98 {
        "Neutral - Balanced Market".to_string()
    } else if sopr > 0.95 {
        "Fear - Some Capitulation".to_string()
    } else {
        "Extreme Fear - Heavy Capitulation".to_string()
    }
}

fn mean(series: &[f64]) -> f64 {
    if series.is_empty() { return 0.0; }
    let sum: f64 = series.iter().copied().sum();
    sum / (series.len() as f64)
}