use pyo3::prelude::*;

use crate::momentum;
use crate::mean_reversion;
use crate::samuelson;

#[pyclass]
#[derive(Clone)]
pub struct FuturesUnifiedResult {
    #[pyo3(get)]
    pub consensus_signal: f64,
    #[pyo3(get)]
    pub consensus_confidence: f64,
    #[pyo3(get)]
    pub technical_score: f64,
    #[pyo3(get)]
    pub term_structure_score: f64,
    #[pyo3(get)]
    pub trading_signals: Vec<String>,
}

#[pyfunction]
pub fn unified_analyze(high: Vec<f64>, low: Vec<f64>, close: Vec<f64>, volume: Vec<f64>, basis: Option<Vec<f64>>) -> PyResult<FuturesUnifiedResult> {
    let momentum_res = momentum::analyze(&high, &low, &close, &volume);
    let mean_rev_res = mean_reversion::analyze_mean_reversion(high.clone(), low.clone(), close.clone())?;
    let sam_res = samuelson::analyze_samuelson(close.clone(), basis.clone())?;

    // Technical score: average of last momentum score and reversion signal mapping
    let tech_score = if momentum_res.momentum_scores.is_empty() { 0.0 } else { *momentum_res.momentum_scores.last().unwrap() };
    let rev_signal_score = match mean_rev_res.reversion_signals.last() { Some(s) if s == "BUY" => 0.5, Some(s) if s == "SELL" => -0.5, _ => 0.0 };
    let technical_score = (tech_score + rev_signal_score).clamp(-1.0, 1.0);

    // Term structure score derived from Samuelson/backwardation
    let term_structure_score = ((sam_res.samuelson_effect - 0.5) * 0.5 + (sam_res.backwardation_signal - 0.5) * 0.5).clamp(-0.5, 0.5);

    // Consensus signal is weighted sum
    let consensus_signal = (technical_score * 0.7 + term_structure_score * 0.3).clamp(-1.0, 1.0);

    // Confidence from strength and volatility stability
    let strength = if momentum_res.momentum_strength.is_empty() { 0.0 } else { *momentum_res.momentum_strength.last().unwrap() };
    let confidence = (0.5 + 0.5 * strength).clamp(0.0, 1.0);

    // Trading signals join
    let mut signals = Vec::new();
    if let Some(s) = momentum_res.momentum_signals.last() { signals.push(format!("Momentum: {}", s)); }
    if let Some(s) = mean_rev_res.reversion_signals.last() { signals.push(format!("MeanReversion: {}", s)); }
    signals.push(format!("TermStructure: {:.2}", term_structure_score));

    Ok(FuturesUnifiedResult {
        consensus_signal,
        consensus_confidence: confidence,
        technical_score,
        term_structure_score,
        trading_signals: signals,
    })
}