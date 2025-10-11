use pyo3::prelude::*;

mod s2f;
mod metcalfe;
mod log_regression;
mod mvrv;
mod sopr;
mod puell;
mod hash_ribbons;
mod hodl_waves;

#[pyfunction]
fn stock_to_flow(stock: Vec<f64>, flow: Vec<f64>, prices: Option<Vec<f64>>) -> PyResult<PyObject> {
    let result = s2f::compute_stock_to_flow(&stock, &flow, prices.as_ref().map(|v| v.as_slice()));
    Python::with_gil(|py| {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("s2f", result.s2f.clone())?;
        if let Some(series) = result.predicted_series.clone() {
            dict.set_item("predicted_series", series)?;
        }
        if let Some(r2) = result.r_squared {
            dict.set_item("r_squared", r2)?;
        }
        if let Some(bands) = result.bands.clone() {
            dict.set_item("upper_band", bands.0)?;
            dict.set_item("lower_band", bands.1)?;
        }
        Ok(dict.into_py(py))
    })
}

#[pyfunction]
fn metcalfe_law(active_addresses: Vec<f64>, prices: Option<Vec<f64>>) -> PyResult<PyObject> {
    let result = metcalfe::compute_metcalfe(&active_addresses, prices.as_ref().map(|v| v.as_slice()));
    Python::with_gil(|py| {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("predicted_series", result.predicted_series)?;
        if let Some(r2) = result.r_squared {
            dict.set_item("r_squared", r2)?;
        }
        dict.set_item("alpha", result.alpha)?;
        dict.set_item("beta", result.beta)?;
        Ok(dict.into_py(py))
    })
}

#[pyfunction]
fn crypto_log_regression(prices: Vec<f64>, timestamps_days: Option<Vec<f64>>) -> PyResult<PyObject> {
    let result = log_regression::compute_log_regression(&prices, timestamps_days.as_ref().map(|v| v.as_slice()));
    Python::with_gil(|py| {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("predicted_series", result.predicted_series)?;
        dict.set_item("upper_band", result.upper_band)?;
        dict.set_item("lower_band", result.lower_band)?;
        if let Some(r2) = result.r_squared { dict.set_item("r_squared", r2)?; }
        dict.set_item("coefficients", result.coefficients)?;
        Ok(dict.into_py(py))
    })
}

#[pyfunction]
fn mvrv_analyze(market_cap: Vec<f64>, realized_cap: Vec<f64>, timestamps: Option<Vec<String>>) -> PyResult<PyObject> {
    let result = mvrv::compute_mvrv(&market_cap, &realized_cap);
    Python::with_gil(|py| {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("current_mvrv", result.current_mvrv)?;
        dict.set_item("mvrv_z_score", result.mvrv_z_score)?;
        dict.set_item("mvrv_percentile", result.mvrv_percentile)?;
        dict.set_item("market_phase", result.market_phase)?;
        dict.set_item("historical_mvrv", result.historical_mvrv)?;
        let bands = pyo3::types::PyDict::new(py);
        bands.set_item("bottom", result.bottom)?;
        bands.set_item("low", result.low)?;
        bands.set_item("fair", result.fair)?;
        bands.set_item("high", result.high)?;
        bands.set_item("top", result.top)?;
        dict.set_item("mvrv_bands", bands)?;
        if let Some(ts) = timestamps { dict.set_item("timestamps", ts)?; }
        Ok(dict.into_py(py))
    })
}

#[pyfunction]
fn sopr_analyze(sopr_series: Vec<f64>, timestamps: Option<Vec<String>>) -> PyResult<PyObject> {
    let result = sopr::compute_sopr(&sopr_series);
    Python::with_gil(|py| {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("current_sopr", result.current_sopr)?;
        dict.set_item("sopr_trend", result.sopr_trend)?;
        dict.set_item("profit_loss_ratio", result.profit_loss_ratio)?;
        dict.set_item("market_sentiment", result.market_sentiment)?;
        dict.set_item("historical_sopr", result.historical_sopr)?;
        dict.set_item("sopr_ma", result.sopr_ma)?;
        if let Some(ts) = timestamps { dict.set_item("timestamps", ts)?; }
        Ok(dict.into_py(py))
    })
}

#[pyfunction]
fn puell_analyze(daily_issuance_usd: Vec<f64>, issuance_ma_365: Vec<f64>, timestamps: Option<Vec<String>>) -> PyResult<PyObject> {
    let result = puell::compute_puell(&daily_issuance_usd, &issuance_ma_365);
    Python::with_gil(|py| {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("current_puell", result.current_puell)?;
        dict.set_item("puell_percentile", result.puell_percentile)?;
        dict.set_item("mining_profitability", result.mining_profitability)?;
        dict.set_item("market_cycle_phase", result.market_cycle_phase)?;
        dict.set_item("historical_puell", result.historical_puell)?;
        let bands = pyo3::types::PyDict::new(py);
        bands.set_item("bottom", result.bottom)?;
        bands.set_item("low", result.low)?;
        bands.set_item("fair", result.fair)?;
        bands.set_item("high", result.high)?;
        bands.set_item("top", result.top)?;
        dict.set_item("puell_bands", bands)?;
        if let Some(ts) = timestamps { dict.set_item("timestamps", ts)?; }
        Ok(dict.into_py(py))
    })
}

#[pyfunction]
fn hash_ribbons_analyze(hash_rate: Vec<f64>, difficulty: Option<Vec<f64>>, timestamps: Option<Vec<String>>) -> PyResult<PyObject> {
    let result = hash_ribbons::compute_hash_ribbons(&hash_rate, difficulty.as_ref().map(|v| v.as_slice()));
    Python::with_gil(|py| {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("hash_ribbon_signal", result.hash_ribbon_signal)?;
        dict.set_item("miner_capitulation", result.miner_capitulation)?;
        dict.set_item("hash_rate_trend", result.hash_rate_trend)?;
        dict.set_item("difficulty_adjustment", result.difficulty_adjustment)?;
        dict.set_item("mining_health", result.mining_health)?;
        dict.set_item("hash_rate_ma_30", result.hash_rate_ma_30)?;
        dict.set_item("hash_rate_ma_60", result.hash_rate_ma_60)?;
        if let Some(ts) = timestamps { dict.set_item("timestamps", ts)?; }
        Ok(dict.into_py(py))
    })
}

#[pyfunction]
fn hodl_waves_analyze(dates: Vec<String>, age_days: Vec<f64>, values: Vec<f64>) -> PyResult<PyObject> {
    let result = hodl_waves::compute_hodl_waves(&dates, &age_days, &values);
    Python::with_gil(|py| {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("age_distribution", result.age_distribution)?;
        dict.set_item("hodl_strength", result.hodl_strength)?;
        dict.set_item("supply_maturity", result.supply_maturity)?;
        dict.set_item("long_term_holder_ratio", result.long_term_holder_ratio)?;
        dict.set_item("recent_activity_ratio", result.recent_activity_ratio)?;
        dict.set_item("hodl_trend", result.hodl_trend)?;
        dict.set_item("timestamps", result.timestamps)?;
        Ok(dict.into_py(py))
    })
}

#[pymodule]
fn crypto_indicators(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(stock_to_flow, m)?)?;
    m.add_function(wrap_pyfunction!(metcalfe_law, m)?)?;
    m.add_function(wrap_pyfunction!(crypto_log_regression, m)?)?;
    m.add_function(wrap_pyfunction!(mvrv_analyze, m)?)?;
    m.add_function(wrap_pyfunction!(sopr_analyze, m)?)?;
    m.add_function(wrap_pyfunction!(puell_analyze, m)?)?;
    m.add_function(wrap_pyfunction!(hash_ribbons_analyze, m)?)?;
    m.add_function(wrap_pyfunction!(hodl_waves_analyze, m)?)?;
    Ok(())
}