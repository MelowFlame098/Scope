pub struct MVRVResult {
    pub current_mvrv: f64,
    pub mvrv_z_score: f64,
    pub mvrv_percentile: f64,
    pub market_phase: String,
    pub historical_mvrv: Vec<f64>,
    pub bottom: f64,
    pub low: f64,
    pub fair: f64,
    pub high: f64,
    pub top: f64,
}

pub fn compute_mvrv(market_cap: &[f64], realized_cap: &[f64]) -> MVRVResult {
    let n = market_cap.len().min(realized_cap.len());
    let mut ratios: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        let rc = realized_cap[i];
        let r = if rc <= 0.0 { 1.0 } else { market_cap[i] / rc };
        ratios.push(r);
    }

    let current_mvrv = if ratios.is_empty() { 1.0 } else { *ratios.last().unwrap() };
    let mvrv_z_score = z_score(current_mvrv, &ratios);
    let mvrv_percentile = percentile_of_score(&ratios, current_mvrv) * 100.0;
    let market_phase = determine_market_phase(mvrv_z_score);
    let (bottom, low, fair, high, top) = if ratios.len() < 10 {
        (0.0, 0.0, 0.0, 0.0, 0.0)
    } else {
        let mut sorted = ratios.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (
            percentile(&sorted, 5.0),
            percentile(&sorted, 25.0),
            percentile(&sorted, 50.0),
            percentile(&sorted, 75.0),
            percentile(&sorted, 95.0),
        )
    };

    MVRVResult {
        current_mvrv,
        mvrv_z_score,
        mvrv_percentile,
        market_phase,
        historical_mvrv: ratios,
        bottom,
        low,
        fair,
        high,
        top,
    }
}

fn z_score(current: f64, series: &[f64]) -> f64 {
    if series.len() < 30 {
        return 0.0;
    }
    let mean = mean(series);
    let std = stddev(series, mean);
    if std == 0.0 { 0.0 } else { (current - mean) / std }
}

fn mean(series: &[f64]) -> f64 {
    if series.is_empty() { return 0.0; }
    let sum: f64 = series.iter().copied().sum();
    sum / (series.len() as f64)
}

fn stddev(series: &[f64], mean: f64) -> f64 {
    if series.is_empty() { return 0.0; }
    let var: f64 = series.iter().map(|v| {
        let d = *v - mean;
        d * d
    }).sum::<f64>() / (series.len() as f64);
    var.sqrt()
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() { return 0.0; }
    let n = sorted.len();
    let pos = (p / 100.0) * ((n - 1) as f64);
    let lower = pos.floor() as usize;
    let upper = pos.ceil() as usize;
    if lower == upper { return sorted[lower]; }
    let weight = pos - (lower as f64);
    sorted[lower] * (1.0 - weight) + sorted[upper] * weight
}

fn percentile_of_score(series: &[f64], score: f64) -> f64 {
    if series.is_empty() { return 0.0; }
    let mut count = 0usize;
    for v in series.iter() {
        if v <= &score { count += 1; }
    }
    (count as f64) / (series.len() as f64)
}

fn determine_market_phase(z: f64) -> String {
    if z > 7.0 {
        "Extreme Euphoria - Major Top Signal".to_string()
    } else if z > 3.5 {
        "Euphoria - Top Formation".to_string()
    } else if z > 1.0 {
        "Optimism - Bull Market".to_string()
    } else if z > -0.5 {
        "Neutral - Consolidation".to_string()
    } else if z > -1.5 {
        "Pessimism - Bear Market".to_string()
    } else {
        "Extreme Fear - Major Bottom Signal".to_string()
    }
}