use pyo3::prelude::*;

#[pyclass]
#[derive(Clone)]
pub struct MeanReversionResult {
    #[pyo3(get)]
    pub mean_reversion_scores: Vec<f64>,
    #[pyo3(get)]
    pub reversion_signals: Vec<String>,
    #[pyo3(get)]
    pub bollinger_upper: Vec<f64>,
    #[pyo3(get)]
    pub bollinger_lower: Vec<f64>,
    #[pyo3(get)]
    pub bollinger_middle: Vec<f64>,
    #[pyo3(get)]
    pub z_scores: Vec<f64>,
    #[pyo3(get)]
    pub adf_pvalue: f64,
    #[pyo3(get)]
    pub half_life: f64,
    #[pyo3(get)]
    pub reversion_probability: Vec<f64>,
    #[pyo3(get)]
    pub oversold_levels: Vec<bool>,
    #[pyo3(get)]
    pub overbought_levels: Vec<bool>,
}

#[pyfunction]
pub fn analyze_mean_reversion(high: Vec<f64>, low: Vec<f64>, close: Vec<f64>) -> PyResult<MeanReversionResult> {
    let (bb_upper, bb_middle, bb_lower) = bollinger_bands(&close, 20, 2.0);
    let z_scores = z_scores(&close, 20);
    let half_life = half_life_mean_reversion(&close);
    let mean_reversion_scores = mean_reversion_scores(&close, &bb_upper, &bb_lower, &bb_middle);
    let reversion_signals = generate_reversion_signals(&mean_reversion_scores, &z_scores);
    let reversion_probability = z_scores.iter().map(|z| (-(z.abs())).exp().clamp(0.0, 1.0)).collect();
    let oversold_levels: Vec<bool> = z_scores.iter().map(|z| *z < -2.0).collect();
    let overbought_levels: Vec<bool> = z_scores.iter().map(|z| *z > 2.0).collect();
    let adf_pvalue = 0.5; // Placeholder; requires statsmodels equivalent

    Ok(MeanReversionResult {
        mean_reversion_scores,
        reversion_signals,
        bollinger_upper: bb_upper,
        bollinger_lower: bb_lower,
        bollinger_middle: bb_middle,
        z_scores,
        adf_pvalue,
        half_life,
        reversion_probability,
        oversold_levels,
        overbought_levels,
    })
}

fn rolling_mean(data: &[f64], window: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![0.0; n];
    let mut sum = 0.0;
    for i in 0..n {
        sum += data[i];
        if i >= window { sum -= data[i - window]; }
        let count = if i + 1 < window { i + 1 } else { window } as f64;
        out[i] = sum / count;
    }
    out
}

fn rolling_std(data: &[f64], window: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![0.0; n];
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    for i in 0..n {
        let x = data[i];
        sum += x;
        sum_sq += x * x;
        if i >= window { 
            let x_old = data[i - window];
            sum -= x_old;
            sum_sq -= x_old * x_old;
        }
        let count = if i + 1 < window { i + 1 } else { window } as f64;
        let mean = sum / count;
        let var = (sum_sq / count) - mean * mean;
        out[i] = var.max(0.0).sqrt();
    }
    out
}

fn bollinger_bands(close: &[f64], window: usize, num_std: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let m = rolling_mean(close, window);
    let s = rolling_std(close, window);
    let upper: Vec<f64> = m.iter().zip(&s).map(|(mm, ss)| mm + num_std * ss).collect();
    let lower: Vec<f64> = m.iter().zip(&s).map(|(mm, ss)| mm - num_std * ss).collect();
    (upper, m, lower)
}

fn z_scores(close: &[f64], window: usize) -> Vec<f64> {
    let m = rolling_mean(close, window);
    let s = rolling_std(close, window);
    close.iter().enumerate().map(|(i, &c)| {
        if s[i] <= 1e-12 { 0.0 } else { (c - m[i]) / s[i] }
    }).collect()
}

fn half_life_mean_reversion(close: &[f64]) -> f64 {
    if close.len() < 3 { return 0.0; }
    // Approximate AR(1) phi via OLS on delta vs lagged
    let mut x: Vec<f64> = Vec::new();
    let mut y: Vec<f64> = Vec::new();
    for i in 1..close.len() {
        x.push(close[i - 1]);
        y.push(close[i] - close[i - 1]);
    }
    let n = x.len() as f64;
    let x_mean = x.iter().sum::<f64>() / n;
    let y_mean = y.iter().sum::<f64>() / n;
    let cov = x.iter().zip(y.iter()).map(|(xi, yi)| (xi - x_mean) * (yi - y_mean)).sum::<f64>();
    let var_x = x.iter().map(|xi| (xi - x_mean).powi(2)).sum::<f64>();
    let phi = if var_x <= 1e-12 { 0.0 } else { cov / var_x };
    let phi_abs = phi.abs().min(0.999999);
    let hl = if phi_abs <= 1e-9 { 0.0 } else { - (2.0f64.ln()) / ((1.0 + phi_abs).ln()) };
    hl.max(0.0)
}

fn mean_reversion_scores(close: &[f64], bb_upper: &[f64], bb_lower: &[f64], bb_middle: &[f64]) -> Vec<f64> {
    let n = close.len();
    let mut scores = Vec::with_capacity(n);
    for i in 0..n {
        let dist_upper = (bb_upper[i] - close[i]).max(1e-12);
        let dist_lower = (close[i] - bb_lower[i]).max(1e-12);
        // Positive score if closer to lower band (expect reversion up), negative if near upper band
        let score = (dist_lower - dist_upper) / (dist_lower + dist_upper);
        scores.push(score.clamp(-1.0, 1.0));
    }
    scores
}

fn generate_reversion_signals(scores: &[f64], z_scores: &[f64]) -> Vec<String> {
    let n = scores.len().min(z_scores.len());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let s = scores[i];
        let z = z_scores[i];
        let signal = if z < -1.0 || s > 0.3 { "BUY" } else if z > 1.0 || s < -0.3 { "SELL" } else { "HOLD" };
        out.push(signal.to_string());
    }
    out
}