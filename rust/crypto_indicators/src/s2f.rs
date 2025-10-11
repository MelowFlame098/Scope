pub struct S2FResult {
    pub s2f: Vec<f64>,
    pub r_squared: Option<f64>,
    pub predicted_series: Option<Vec<f64>>,
    pub bands: Option<(Vec<f64>, Vec<f64>)>,
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

pub fn compute_stock_to_flow(stock: &[f64], flow: &[f64], prices: Option<&[f64]>) -> S2FResult {
    let len = stock.len().min(flow.len());
    let mut s2f = Vec::with_capacity(len);
    for i in 0..len {
        let f = flow[i];
        let ratio = if f.abs() > f64::EPSILON { stock[i] / f } else { 0.0 };
        s2f.push(ratio.max(0.0));
    }

    if let Some(p) = prices {
        let m = len.min(p.len());
        let x: Vec<f64> = s2f[..m].iter().map(|v| v.ln()).collect();
        let y: Vec<f64> = p[..m].iter().map(|v| v.ln()).collect();

        let (alpha, beta, r2) = linear_regression(&x, &y);
        let mut predicted_series = Vec::with_capacity(m);
        let mut residuals = Vec::with_capacity(m);
        for i in 0..m {
            let y_hat_ln = alpha + beta * x[i];
            let y_hat = y_hat_ln.exp();
            predicted_series.push(y_hat);
            residuals.push(y[i].exp() - y_hat);
        }
        let mean_res: f64 = residuals.iter().sum::<f64>() / (m as f64);
        let std_res: f64 = (residuals.iter().map(|r| (r - mean_res).powi(2)).sum::<f64>() / (m as f64)).sqrt();
        let upper_band: Vec<f64> = predicted_series.iter().map(|v| v + std_res).collect();
        let lower_band: Vec<f64> = predicted_series.iter().map(|v| (v - std_res).max(0.0)).collect();

        S2FResult {
            s2f,
            r_squared: Some(r2),
            predicted_series: Some(predicted_series),
            bands: Some((upper_band, lower_band)),
        }
    } else {
        S2FResult {
            s2f,
            r_squared: None,
            predicted_series: None,
            bands: None,
        }
    }
}