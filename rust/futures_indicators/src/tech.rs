pub fn rsi(prices: &[f64], period: usize) -> Vec<f64> {
    if prices.len() < period + 1 {
        return vec![50.0; prices.len()];
    }
    let mut deltas: Vec<f64> = Vec::with_capacity(prices.len().saturating_sub(1));
    for i in 1..prices.len() {
        deltas.push(prices[i] - prices[i - 1]);
    }
    let gains: Vec<f64> = deltas.iter().map(|d| d.max(0.0)).collect();
    let losses: Vec<f64> = deltas.iter().map(|d| (-d).max(0.0)).collect();

    let mut avg_gain: f64 = gains.iter().take(period).sum::<f64>() / period as f64;
    let mut avg_loss: f64 = losses.iter().take(period).sum::<f64>() / period as f64;

    let mut rsi_vals: Vec<f64> = Vec::with_capacity(prices.len());
    rsi_vals.push(50.0); // first neutral

    for i in period..deltas.len() {
        avg_gain = (avg_gain * (period as f64 - 1.0) + gains[i]) / period as f64;
        avg_loss = (avg_loss * (period as f64 - 1.0) + losses[i]) / period as f64;

        let rsi = if avg_loss == 0.0 {
            100.0
        } else {
            let rs = avg_gain / avg_loss;
            100.0 - (100.0 / (1.0 + rs))
        };
        rsi_vals.push(rsi);
    }

    let pad = prices.len().saturating_sub(rsi_vals.len());
    let mut out = vec![50.0; pad];
    out.extend(rsi_vals);
    out
}

fn ema(values: &[f64], period: usize) -> Vec<f64> {
    if values.is_empty() {
        return vec![];
    }
    let k = 2.0 / (period as f64 + 1.0);
    let mut out = Vec::with_capacity(values.len());
    // seed with SMA of first period
    let seed = values.iter().take(period).sum::<f64>() / period as f64;
    out.push(seed);
    for i in 1..values.len() {
        let prev = *out.last().unwrap();
        let ema_val = values[i] * k + prev * (1.0 - k);
        out.push(ema_val);
    }
    out
}

pub fn macd(prices: &[f64]) -> (Vec<f64>, Vec<f64>) {
    // MACD = EMA(12) - EMA(26); Signal = EMA(9) of MACD
    let ema12 = ema(prices, 12);
    let ema26 = ema(prices, 26);
    let len = ema12.len().min(ema26.len());
    let mut macd_line = Vec::with_capacity(len);
    for i in 0..len {
        macd_line.push(ema12[i] - ema26[i]);
    }
    let signal = ema(&macd_line, 9);
    // align lengths
    let l = macd_line.len().min(signal.len());
    (macd_line[..l].to_vec(), signal[..l].to_vec())
}

pub fn stochastic(high: &[f64], low: &[f64], close: &[f64], period: usize) -> (Vec<f64>, Vec<f64>) {
    let n = close.len();
    let mut k_vals = vec![50.0; n];
    for i in 0..n {
        let start = i.saturating_sub(period - 1);
        let hh = high[start..=i].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let ll = low[start..=i].iter().cloned().fold(f64::INFINITY, f64::min);
        let denom = (hh - ll).max(1e-12);
        k_vals[i] = ((close[i] - ll) / denom) * 100.0;
    }
    // D is SMA of K over 3
    let mut d_vals = vec![50.0; n];
    for i in 0..n {
        let start = i.saturating_sub(2);
        let window = &k_vals[start..=i];
        d_vals[i] = window.iter().sum::<f64>() / window.len() as f64;
    }
    (k_vals, d_vals)
}

pub fn williams_r(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let n = close.len();
    let mut wr = vec![-50.0; n];
    for i in 0..n {
        let start = i.saturating_sub(period - 1);
        let hh = high[start..=i].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let ll = low[start..=i].iter().cloned().fold(f64::INFINITY, f64::min);
        let denom = (hh - ll).max(1e-12);
        wr[i] = -100.0 * ((hh - close[i]) / denom);
    }
    wr
}