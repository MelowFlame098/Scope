use pyo3::prelude::*;

use crate::tech;

#[pyclass]
#[derive(Clone)]
pub struct MomentumAnalysis {
    #[pyo3(get)]
    pub momentum_scores: Vec<f64>,
    #[pyo3(get)]
    pub momentum_signals: Vec<String>,
    #[pyo3(get)]
    pub momentum_strength: Vec<f64>,
    #[pyo3(get)]
    pub trend_direction: Vec<String>,
    #[pyo3(get)]
    pub momentum_divergence: Vec<bool>,
    #[pyo3(get)]
    pub rsi_values: Vec<f64>,
    #[pyo3(get)]
    pub macd_values: Vec<f64>,
    #[pyo3(get)]
    pub macd_signal: Vec<f64>,
    #[pyo3(get)]
    pub stochastic_k: Vec<f64>,
    #[pyo3(get)]
    pub stochastic_d: Vec<f64>,
    #[pyo3(get)]
    pub williams_r: Vec<f64>,
}

pub fn analyze(high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> MomentumAnalysis {
    let rsi_values = tech::rsi(close, 14);
    let (macd_values, macd_signal) = tech::macd(close);
    let (stochastic_k, stochastic_d) = tech::stochastic(high, low, close, 14);
    let williams_r = tech::williams_r(high, low, close, 14);

    let momentum_scores = composite_scores(&rsi_values, &macd_values, &macd_signal, &stochastic_k, &williams_r);
    let momentum_signals = generate_signals(&momentum_scores);
    let momentum_strength = calculate_strength(&momentum_scores, volume);
    let trend_direction = determine_trend(close, &macd_values);
    let momentum_divergence = detect_divergence(close, &rsi_values);

    MomentumAnalysis {
        momentum_scores,
        momentum_signals,
        momentum_strength,
        trend_direction,
        momentum_divergence,
        rsi_values,
        macd_values,
        macd_signal,
        stochastic_k,
        stochastic_d,
        williams_r,
    }
}

fn composite_scores(rsi: &[f64], macd: &[f64], signal: &[f64], k: &[f64], wr: &[f64]) -> Vec<f64> {
    let n = rsi.len().min(macd.len()).min(signal.len()).min(k.len()).min(wr.len());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let rsi_score = (rsi[i] - 50.0) / 50.0;
        let macd_score = if macd[i] > signal[i] { 1.0 } else { -1.0 };
        let stoch_score = (k[i] - 50.0) / 50.0;
        let williams_score = (wr[i] + 50.0) / 50.0;
        let composite = 0.3 * rsi_score + 0.3 * macd_score + 0.2 * stoch_score + 0.2 * williams_score;
        out.push(composite.clamp(-1.0, 1.0));
    }
    out
}

fn generate_signals(scores: &[f64]) -> Vec<String> {
    scores
        .iter()
        .map(|s| if *s > 0.3 { "BUY" } else if *s < -0.3 { "SELL" } else { "HOLD" }.to_string())
        .collect()
}

fn calculate_strength(scores: &[f64], volume: &[f64]) -> Vec<f64> {
    let n = scores.len().min(volume.len());
    let mut out = Vec::with_capacity(n);
    // normalize volume effect by z-scoring over a rolling window (simple): scale factor
    let vol_mean = if volume.is_empty() { 0.0 } else { volume.iter().sum::<f64>() / volume.len() as f64 };
    let vol_std = if volume.is_empty() {
        1.0
    } else {
        let mean = vol_mean;
        let var = volume.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / volume.len().max(1) as f64;
        var.sqrt().max(1e-9)
    };
    for i in 0..n {
        let vol_z = (volume[i] - vol_mean) / vol_std;
        let strength = (scores[i].abs() * (1.0 + 0.1 * vol_z)).clamp(0.0, 1.0);
        out.push(strength);
    }
    out
}

fn determine_trend(close: &[f64], macd: &[f64]) -> Vec<String> {
    let n = close.len().min(macd.len());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let slope = if i == 0 { 0.0 } else { macd[i] - macd[i - 1] };
        let trend = if slope > 0.0 { "UP" } else if slope < 0.0 { "DOWN" } else { "SIDEWAYS" };
        out.push(trend.to_string());
    }
    out
}

fn detect_divergence(close: &[f64], rsi: &[f64]) -> Vec<bool> {
    let n = close.len().min(rsi.len());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let diverge = if i < 3 {
            false
        } else {
            let price_trend = close[i] - close[i - 3];
            let rsi_trend = rsi[i] - rsi[i - 3];
            (price_trend > 0.0 && rsi_trend < 0.0) || (price_trend < 0.0 && rsi_trend > 0.0)
        };
        out.push(diverge);
    }
    out
}