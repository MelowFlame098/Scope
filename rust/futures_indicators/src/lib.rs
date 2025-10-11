use pyo3::prelude::*;
mod tech;
mod momentum;
mod mean_reversion;
mod samuelson;
mod unified;

/// Python module: futures_indicators
#[pymodule]
fn futures_indicators(_py: Python, m: &PyModule) -> PyResult<()> {
    // Expose technical indicator primitives
    m.add_function(wrap_pyfunction!(rsi, m)?)?;
    m.add_function(wrap_pyfunction!(macd, m)?)?;
    m.add_function(wrap_pyfunction!(stochastic, m)?)?;
    m.add_function(wrap_pyfunction!(williams_r, m)?)?;

    // Expose momentum composite analysis
    m.add_function(wrap_pyfunction!(analyze_momentum, m)?)?;

    // Expose mean reversion and Samuelson/backwardation analyses
    m.add_function(wrap_pyfunction!(analyze_mean_reversion, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_samuelson, m)?)?;
    m.add_function(wrap_pyfunction!(unified_analyze, m)?)?;
    Ok(())
}

#[pyfunction]
fn rsi(prices: Vec<f64>, period: Option<usize>) -> PyResult<Vec<f64>> {
    Ok(tech::rsi(&prices, period.unwrap_or(14)))
}

#[pyfunction]
fn macd(prices: Vec<f64>) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let (macd, signal) = tech::macd(&prices);
    Ok((macd, signal))
}

#[pyfunction]
fn stochastic(high: Vec<f64>, low: Vec<f64>, close: Vec<f64>, period: Option<usize>) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let p = period.unwrap_or(14);
    let (k, d) = tech::stochastic(&high, &low, &close, p);
    Ok((k, d))
}

#[pyfunction]
fn williams_r(high: Vec<f64>, low: Vec<f64>, close: Vec<f64>, period: Option<usize>) -> PyResult<Vec<f64>> {
    Ok(tech::williams_r(&high, &low, &close, period.unwrap_or(14)))
}

#[pyfunction]
fn analyze_momentum(high: Vec<f64>, low: Vec<f64>, close: Vec<f64>, volume: Vec<f64>) -> PyResult<momentum::MomentumAnalysis> {
    Ok(momentum::analyze(&high, &low, &close, &volume))
}

#[pyfunction]
fn analyze_mean_reversion(high: Vec<f64>, low: Vec<f64>, close: Vec<f64>) -> PyResult<mean_reversion::MeanReversionResult> {
    mean_reversion::analyze_mean_reversion(high, low, close)
}

#[pyfunction]
fn analyze_samuelson(close: Vec<f64>, basis: Option<Vec<f64>>) -> PyResult<samuelson::SamuelsonBackwardationResult> {
    samuelson::analyze_samuelson(close, basis)
}

#[pyfunction]
fn unified_analyze(high: Vec<f64>, low: Vec<f64>, close: Vec<f64>, volume: Vec<f64>, basis: Option<Vec<f64>>) -> PyResult<unified::FuturesUnifiedResult> {
    unified::unified_analyze(high, low, close, volume, basis)
}