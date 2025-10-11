use std::f64;

#[derive(Clone, Debug)]
pub struct HashRibbonsResult {
    pub hash_ribbon_signal: String,
    pub miner_capitulation: bool,
    pub hash_rate_trend: String,
    pub difficulty_adjustment: f64,
    pub mining_health: String,
    pub hash_rate_ma_30: Vec<f64>,
    pub hash_rate_ma_60: Vec<f64>,
}

pub fn compute_hash_ribbons(hash_rate: &[f64], difficulty: Option<&[f64]>) -> HashRibbonsResult {
    let ma30 = rolling_mean(hash_rate, 30);
    let ma60 = rolling_mean(hash_rate, 60);

    let current_ma30 = ma30.last().copied().unwrap_or(0.0);
    let current_ma60 = ma60.last().copied().unwrap_or(0.0);

    let hash_ribbon_signal = if current_ma30 > current_ma60 {
        "Buy Signal - Hash Rate Recovery".to_string()
    } else {
        "Sell Signal - Hash Rate Decline".to_string()
    };

    // recent changes over ~14 periods
    let hash_rate_change = if hash_rate.len() >= 14 && hash_rate[hash_rate.len()-14] != 0.0 {
        (hash_rate[hash_rate.len()-1] - hash_rate[hash_rate.len()-14]) / hash_rate[hash_rate.len()-14]
    } else { 0.0 };

    let difficulty_adjustment = match difficulty {
        Some(diff) if diff.len() >= 14 && diff[diff.len()-14] != 0.0 => {
            (diff[diff.len()-1] - diff[diff.len()-14]) / diff[diff.len()-14]
        }
        _ => 0.0,
    };

    let miner_capitulation = detect_miner_capitulation(hash_rate_change, difficulty_adjustment);
    let hash_rate_trend = calculate_hash_rate_trend(hash_rate);
    let mining_health = assess_mining_health(&hash_ribbon_signal, miner_capitulation, &hash_rate_trend);

    HashRibbonsResult {
        hash_ribbon_signal,
        miner_capitulation,
        hash_rate_trend,
        difficulty_adjustment,
        mining_health,
        hash_rate_ma_30: ma30,
        hash_rate_ma_60: ma60,
    }
}

fn rolling_mean(series: &[f64], window: usize) -> Vec<f64> {
    if series.is_empty() || window == 0 { return vec![]; }
    let mut out = Vec::with_capacity(series.len());
    let mut sum = 0.0;
    for i in 0..series.len() {
        sum += series[i];
        if i >= window { sum -= series[i - window]; }
        let count = if i + 1 < window { i + 1 } else { window } as f64;
        out.push(sum / count);
    }
    out
}

fn detect_miner_capitulation(hash_rate_change: f64, difficulty_change: f64) -> bool {
    hash_rate_change < -0.15 && difficulty_change > -0.05
}

fn calculate_hash_rate_trend(recent_hash_rates: &[f64]) -> String {
    if recent_hash_rates.len() < 7 { return "Insufficient Data".to_string(); }
    let n = recent_hash_rates.len().min(14);
    let start = recent_hash_rates.len() - n;
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = recent_hash_rates[start..].to_vec();
    let (slope, r) = linreg_slope_r(&x, &y);
    if r.abs() < 0.3 { return "Stable".to_string(); }
    if slope > 0.0 { if r > 0.5 { "Growing".to_string() } else { "Slowly Growing".to_string() } }
    else { if r < -0.5 { "Declining".to_string() } else { "Slowly Declining".to_string() } }
}

fn assess_mining_health(signal: &str, miner_capitulation: bool, trend: &str) -> String {
    if miner_capitulation { return "Poor - Miner Capitulation Event".to_string(); }
    if signal.contains("Buy") && trend.contains("Growing") { return "Excellent - Strong Network Growth".to_string(); }
    if signal.contains("Buy") { return "Good - Network Recovery".to_string(); }
    if trend.contains("Declining") { return "Concerning - Network Stress".to_string(); }
    "Fair - Stable Network".to_string()
}

fn linreg_slope_r(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len().min(y.len());
    if n == 0 { return (0.0, 0.0); }
    let xm = mean(&x[..n]);
    let ym = mean(&y[..n]);
    let mut num = 0.0;
    let mut den_x = 0.0;
    let mut den_y = 0.0;
    for i in 0..n {
        let dx = x[i] - xm;
        let dy = y[i] - ym;
        num += dx * dy;
        den_x += dx * dx;
        den_y += dy * dy;
    }
    let slope = if den_x == 0.0 { 0.0 } else { num / den_x };
    let r = if den_x == 0.0 || den_y == 0.0 {
        0.0
    } else {
        num / (den_x.sqrt() * den_y.sqrt())
    };
    (slope, r)
}

fn mean(series: &[f64]) -> f64 {
    if series.is_empty() { 0.0 }
    else { series.iter().sum::<f64>() / series.len() as f64 }
}