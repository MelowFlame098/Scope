use pyo3::prelude::*;

#[pyclass]
#[derive(Clone)]
pub struct SamuelsonBackwardationResult {
    #[pyo3(get)]
    pub samuelson_effect: f64,
    #[pyo3(get)]
    pub backwardation_signal: f64,
}

#[pyfunction]
pub fn analyze_samuelson(close: Vec<f64>, basis: Option<Vec<f64>>) -> PyResult<SamuelsonBackwardationResult> {
    // Simple volatility ratio as proxy for Samuelson effect
    let vol_short = rolling_std(&close, 10).last().cloned().unwrap_or(0.0);
    let vol_long = rolling_std(&close, 30).last().cloned().unwrap_or(0.0);
    let ratio = if vol_long <= 1e-12 { 0.0 } else { vol_short / vol_long };
    let samuelson_effect = (0.5 + (ratio - 1.0) * 0.25).clamp(0.0, 1.0);

    let backwardation_signal = match basis {
        Some(b) if !b.is_empty() => {
            let avg = b.iter().copied().sum::<f64>() / b.len() as f64;
            if avg < -0.02 { 0.7 } else if avg > 0.02 { 0.3 } else { 0.5 }
        }
        _ => 0.5,
    };

    Ok(SamuelsonBackwardationResult { samuelson_effect, backwardation_signal })
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