pub struct LogRegressionResult {
    pub predicted_series: Vec<f64>,
    pub upper_band: Vec<f64>,
    pub lower_band: Vec<f64>,
    pub r_squared: Option<f64>,
    pub coefficients: Vec<f64>,
}

// Fit log(price) = a + b*log(t) + c*(t_years)
pub fn compute_log_regression(prices: &[f64], timestamps_days: Option<&[f64]>) -> LogRegressionResult {
    let n = prices.len();
    if n == 0 {
        return LogRegressionResult { predicted_series: vec![], upper_band: vec![], lower_band: vec![], r_squared: None, coefficients: vec![0.0,0.0,0.0] };
    }

    // Build time vector in days since start
    let t_days: Vec<f64> = match timestamps_days {
        Some(ts) => ts.iter().map(|v| *v).collect(),
        None => (0..n).map(|i| i as f64).collect(),
    };
    let t_years: Vec<f64> = t_days.iter().map(|d| d / 365.25).collect();

    // Design matrix X: [1, ln(t+1), t_years]
    let mut x0: Vec<f64> = vec![1.0; n];
    let mut x1: Vec<f64> = t_days.iter().map(|d| (d + 1.0).ln()).collect();
    let mut x2: Vec<f64> = t_years.clone();

    // Response y = ln(price)
    let y: Vec<f64> = prices.iter().map(|p| if *p > 0.0 { p.ln() } else { 0.0 }).collect();

    // Mask valid rows (finite values)
    let mut x_clean: Vec<[f64;3]> = Vec::with_capacity(n);
    let mut y_clean: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        let xi = [x0[i], x1[i], x2[i]];
        if xi.iter().all(|v| v.is_finite()) && y[i].is_finite() {
            x_clean.push(xi);
            y_clean.push(y[i]);
        }
    }

    if x_clean.len() < 10 {
        // Not enough data; return baseline
        let predicted = prices.to_vec();
        return LogRegressionResult { predicted_series: predicted.clone(), upper_band: predicted.clone(), lower_band: predicted, r_squared: None, coefficients: vec![0.0,0.0,0.0] };
    }

    // Compute normal equations: (X^T X) beta = X^T y
    let mut xtx = [[0.0;3];3];
    let mut xty = [0.0;3];
    for i in 0..x_clean.len() {
        let xi = x_clean[i];
        for r in 0..3 {
            for c in 0..3 {
                xtx[r][c] += xi[r] * xi[c];
            }
            xty[r] += xi[r] * y_clean[i];
        }
    }

    // Solve 3x3 using Cramer's rule / Gaussian elimination
    let beta = solve_3x3(xtx, xty);
    let alpha = beta[0];
    let b = beta[1];
    let c = beta[2];

    // Predictions
    let mut log_predicted: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        log_predicted.push(alpha + b * x1[i] + c * x2[i]);
    }
    let predicted_series: Vec<f64> = log_predicted.iter().map(|lp| lp.exp()).collect();

    // Residuals on clean subset
    let mut residuals: Vec<f64> = Vec::with_capacity(x_clean.len());
    for i in 0..x_clean.len() {
        let xi = x_clean[i];
        let yh = alpha + b * xi[1] + c * xi[2];
        residuals.push(y_clean[i] - yh);
    }
    let mean_res = residuals.iter().sum::<f64>() / (residuals.len() as f64);
    let std_res = (residuals.iter().map(|r| (r - mean_res).powi(2)).sum::<f64>() / (residuals.len() as f64)).sqrt();
    let upper_band: Vec<f64> = log_predicted.iter().map(|lp| (lp + 2.0 * std_res).exp()).collect();
    let lower_band: Vec<f64> = log_predicted.iter().map(|lp| (lp - 2.0 * std_res).exp()).collect();

    // R-squared
    let ss_res: f64 = residuals.iter().map(|r| r * r).sum();
    let mean_y: f64 = y_clean.iter().sum::<f64>() / (y_clean.len() as f64);
    let ss_tot: f64 = y_clean.iter().map(|yi| (yi - mean_y).powi(2)).sum();
    let r2 = if ss_tot > 0.0 { 1.0 - (ss_res / ss_tot) } else { 0.0 };

    LogRegressionResult {
        predicted_series,
        upper_band,
        lower_band,
        r_squared: Some(r2),
        coefficients: vec![alpha, b, c],
    }
}

fn solve_3x3(a: [[f64;3];3], b: [f64;3]) -> [f64;3] {
    // Gaussian elimination for 3x3
    let mut m = [[0.0;4];3];
    for i in 0..3 {
        for j in 0..3 { m[i][j] = a[i][j]; }
        m[i][3] = b[i];
    }
    // Forward elimination
    for i in 0..3 {
        // Pivot
        let mut pivot = i;
        for r in i+1..3 {
            if m[r][i].abs() > m[pivot][i].abs() { pivot = r; }
        }
        if pivot != i { m.swap(i, pivot); }
        let div = m[i][i];
        if div.abs() < f64::EPSILON { continue; }
        for j in i..4 { m[i][j] /= div; }
        // Eliminate
        for r in i+1..3 {
            let factor = m[r][i];
            for j in i..4 { m[r][j] -= factor * m[i][j]; }
        }
    }
    // Back substitution
    for i in (0..3).rev() {
        for r in 0..i {
            let factor = m[r][i];
            m[r][3] -= factor * m[i][3];
            m[r][i] = 0.0;
        }
    }
    [m[0][3], m[1][3], m[2][3]]
}