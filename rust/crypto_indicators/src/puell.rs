use std::f64;

#[derive(Clone, Debug)]
pub struct PuellResult {
    pub current_puell: f64,
    pub puell_percentile: f64,
    pub mining_profitability: String,
    pub market_cycle_phase: String,
    pub historical_puell: Vec<f64>,
    // bands
    pub bottom: f64,
    pub low: f64,
    pub fair: f64,
    pub high: f64,
    pub top: f64,
}

pub fn compute_puell(daily_issuance_usd: &[f64], issuance_ma_365: &[f64]) -> PuellResult {
    let n = daily_issuance_usd.len().min(issuance_ma_365.len());
    let mut puell_values: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        let ma = issuance_ma_365[i];
        let val = if ma <= 0.0 { 1.0 } else { daily_issuance_usd[i] / ma };
        puell_values.push(val);
    }

    let current_puell = puell_values.last().copied().unwrap_or(1.0);
    let puell_percentile = percentile_of_score(&puell_values, current_puell);
    let mining_profitability = determine_mining_profitability(current_puell).to_string();
    let market_cycle_phase = determine_market_cycle_phase(puell_percentile).to_string();

    let (bottom, low, fair, high, top) = puell_bands(&puell_values);

    PuellResult {
        current_puell,
        puell_percentile,
        mining_profitability,
        market_cycle_phase,
        historical_puell: puell_values,
        bottom,
        low,
        fair,
        high,
        top,
    }
}

fn percentile_of_score(series: &[f64], value: f64) -> f64 {
    if series.is_empty() { return 50.0; }
    let mut sorted = series.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut count = 0usize;
    for v in &sorted { if *v <= value { count += 1; } }
    (count as f64 / sorted.len() as f64) * 100.0
}

fn determine_mining_profitability(puell_multiple: f64) -> &'static str {
    if puell_multiple > 4.0 {
        "Extremely High Profitability - Potential Top"
    } else if puell_multiple > 2.0 {
        "High Profitability - Bull Market"
    } else if puell_multiple > 0.5 {
        "Normal Profitability - Stable Market"
    } else if puell_multiple > 0.3 {
        "Low Profitability - Bear Market"
    } else {
        "Extremely Low Profitability - Potential Bottom"
    }
}

fn determine_market_cycle_phase(puell_percentile: f64) -> &'static str {
    if puell_percentile > 95.0 {
        "Cycle Top - Extreme Overheating"
    } else if puell_percentile > 80.0 {
        "Late Bull Market - Overheating"
    } else if puell_percentile > 60.0 {
        "Bull Market - Healthy Growth"
    } else if puell_percentile > 40.0 {
        "Neutral - Consolidation"
    } else if puell_percentile > 20.0 {
        "Bear Market - Cooling Down"
    } else {
        "Cycle Bottom - Extreme Undervaluation"
    }
}

fn puell_bands(series: &[f64]) -> (f64, f64, f64, f64, f64) {
    if series.len() < 10 { return (0.0, 0.0, 0.0, 0.0, 0.0); }
    (
        percentile(series, 10.0),
        percentile(series, 30.0),
        percentile(series, 50.0),
        percentile(series, 70.0),
        percentile(series, 90.0),
    )
}

fn percentile(series: &[f64], p: f64) -> f64 {
    if series.is_empty() { return 0.0; }
    let mut sorted = series.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let rank = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[rank.min(sorted.len() - 1)]
}