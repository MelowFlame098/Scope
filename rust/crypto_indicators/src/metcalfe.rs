pub struct MetcalfeResult {
    pub predicted_series: Vec<f64>,
    pub r_squared: Option<f64>,
    pub alpha: f64,
    pub beta: f64,
}

fn linear_regression(x: &[f64], y: &[f64]) -> (f64, f64, f64) {
    // Returns (alpha, beta, r_squared) for y = alpha + beta * x
    let n = x.len() as f64;
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let mean_x = sum_x / n;
    let mean_y = sum_y / n;
    let mut ss_xy = 0.0;
    let mut ss_xx = 0.0;
    let mut ss_tot = 0.0;
    let mut ss_res = 0.0;
    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        ss_xy += dx * dy;
        ss_xx += dx * dx;
    }
    let beta = if ss_xx != 0.0 { ss_xy / ss_xx } else { 0.0 };
    let alpha = mean_y - beta * mean_x;
    for i in 0..x.len() {
        let y_hat = alpha + beta * x[i];
        ss_tot += (y[i] - mean_y).powi(2);
        ss_res += (y[i] - y_hat).powi(2);
    }
    let r_squared = if ss_tot != 0.0 { 1.0 - (ss_res / ss_tot) } else { 0.0 };
    (alpha, beta, r_squared)
}

pub fn compute_metcalfe(active_addresses: &[f64], prices: Option<&[f64]>) -> MetcalfeResult {
    let x: Vec<f64> = active_addresses.iter().map(|a| a * a).collect();
    if let Some(p) = prices {
        let m = x.len().min(p.len());
        let (alpha, beta, r2) = linear_regression(&x[..m], &p[..m]);
        let predicted_series: Vec<f64> = x[..m].iter().map(|xi| (alpha + beta * xi).max(0.0)).collect();
        MetcalfeResult {
            predicted_series,
            r_squared: Some(r2),
            alpha,
            beta,
        }
    } else {
        // Without prices, return the raw network value using beta=1, alpha=0
        let predicted_series: Vec<f64> = x.iter().map(|xi| *xi).collect();
        MetcalfeResult {
            predicted_series,
            r_squared: None,
            alpha: 0.0,
            beta: 1.0,
        }
    }
}