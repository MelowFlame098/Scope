use std::collections::BTreeMap;

pub struct HODLWavesResult {
    pub age_distribution: BTreeMap<String, f64>,
    pub hodl_strength: f64,
    pub supply_maturity: String,
    pub long_term_holder_ratio: f64,
    pub recent_activity_ratio: f64,
    pub hodl_trend: String,
    pub timestamps: Vec<String>,
}

pub fn compute_hodl_waves(dates: &Vec<String>, age_days: &Vec<f64>, values: &Vec<f64>) -> HODLWavesResult {
    // Guard against mismatched inputs
    if dates.is_empty() || age_days.is_empty() || values.is_empty() || (dates.len() != age_days.len()) || (age_days.len() != values.len()) {
        let mut dist = BTreeMap::new();
        dist.insert("<1m".to_string(), 0.1);
        dist.insert("1-3m".to_string(), 0.15);
        dist.insert("3-6m".to_string(), 0.2);
        dist.insert("6-12m".to_string(), 0.25);
        dist.insert(">1y".to_string(), 0.3);
        let hodl_strength = calculate_hodl_strength(&dist);
        return finalize_result(dist, hodl_strength, vec!["2024-01-01".to_string()], Vec::new());
    }

    // Build ordered unique date list preserving encounter order
    let mut unique_dates: Vec<String> = Vec::new();
    for d in dates.iter() {
        if !unique_dates.contains(d) {
            unique_dates.push(d.clone());
        }
    }
    // Restrict to last 30 unique dates
    let start = if unique_dates.len() > 30 { unique_dates.len() - 30 } else { 0 };
    let window_dates: Vec<String> = unique_dates[start..].to_vec();

    let mut timestamps: Vec<String> = Vec::new();
    let mut hodl_strength_history: Vec<f64> = Vec::new();
    let mut last_distribution: Option<BTreeMap<String, f64>> = None;

    for d in window_dates.iter() {
        // Indices for this date
        let mut idxs: Vec<usize> = Vec::new();
        for (i, di) in dates.iter().enumerate() {
            if di == d { idxs.push(i); }
        }

        let dist = calculate_age_distribution(&idxs, age_days, values);
        let hs = calculate_hodl_strength(&dist);
        timestamps.push(d.clone());
        hodl_strength_history.push(hs);
        last_distribution = Some(dist);
    }

    let (current_dist, current_hs) = if let Some(dist) = last_distribution {
        (dist, hodl_strength_history.last().cloned().unwrap_or(0.5))
    } else {
        // Fallback
        let mut dist = BTreeMap::new();
        dist.insert("<1m".to_string(), 0.1);
        dist.insert("1-3m".to_string(), 0.15);
        dist.insert("3-6m".to_string(), 0.2);
        dist.insert("6-12m".to_string(), 0.25);
        dist.insert(">1y".to_string(), 0.3);
        (dist, 0.7)
    };

    finalize_result(current_dist, current_hs, timestamps, hodl_strength_history)
}

fn calculate_age_distribution(idxs: &Vec<usize>, age_days: &Vec<f64>, values: &Vec<f64>) -> BTreeMap<String, f64> {
    let mut buckets: BTreeMap<String, f64> = BTreeMap::new();
    buckets.insert("<1m".to_string(), 0.0);
    buckets.insert("1-3m".to_string(), 0.0);
    buckets.insert("3-6m".to_string(), 0.0);
    buckets.insert("6-12m".to_string(), 0.0);
    buckets.insert(">1y".to_string(), 0.0);

    let mut total_value: f64 = 0.0;
    for &i in idxs.iter() {
        let age = age_days[i];
        let val = values[i];
        total_value += val.max(0.0);
        if age < 30.0 {
            *buckets.get_mut("<1m").unwrap() += val;
        } else if age < 90.0 {
            *buckets.get_mut("1-3m").unwrap() += val;
        } else if age < 180.0 {
            *buckets.get_mut("3-6m").unwrap() += val;
        } else if age < 365.0 {
            *buckets.get_mut("6-12m").unwrap() += val;
        } else {
            *buckets.get_mut(">1y").unwrap() += val;
        }
    }

    if total_value <= 0.0 {
        return buckets; // all zeros
    }
    for (_k, v) in buckets.iter_mut() {
        *v = *v / total_value;
    }
    buckets
}

fn calculate_hodl_strength(age_distribution: &BTreeMap<String, f64>) -> f64 {
    let weights = vec![
        ("<1m", 0.1),
        ("1-3m", 0.2),
        ("3-6m", 0.3),
        ("6-12m", 0.4),
        (">1y", 1.0),
    ];
    let mut strength = 0.0;
    for (k, w) in weights.into_iter() {
        if let Some(v) = age_distribution.get(k) {
            strength += v * w;
        }
    }
    strength.min(1.0)
}

fn determine_supply_maturity(lth_ratio: f64) -> String {
    if lth_ratio > 0.7 {
        "Very Mature - Strong HODLing".to_string()
    } else if lth_ratio > 0.5 {
        "Mature - Moderate HODLing".to_string()
    } else if lth_ratio > 0.3 {
        "Developing - Mixed Behavior".to_string()
    } else {
        "Young - High Activity".to_string()
    }
}

fn calculate_hodl_trend(history: &Vec<f64>) -> String {
    if history.len() < 7 {
        return "Insufficient Data".to_string();
    }
    let recent_slice_start = if history.len() >= 7 { history.len() - 7 } else { 0 };
    let recent_avg = mean(&history[recent_slice_start..]);
    let older_avg = if history.len() >= 14 {
        mean(&history[(history.len() - 14)..(history.len() - 7)])
    } else {
        recent_avg
    };
    let change = if older_avg > 0.0 { (recent_avg - older_avg) / older_avg } else { 0.0 };
    if change > 0.05 {
        "Strengthening".to_string()
    } else if change < -0.05 {
        "Weakening".to_string()
    } else {
        "Stable".to_string()
    }
}

fn finalize_result(
    current_age_distribution: BTreeMap<String, f64>,
    current_hodl_strength: f64,
    timestamps: Vec<String>,
    hodl_strength_history: Vec<f64>,
) -> HODLWavesResult {
    let lth_ratio = current_age_distribution.get(">1y").cloned().unwrap_or(0.0)
        + current_age_distribution.get("6-12m").cloned().unwrap_or(0.0);
    let recent_activity_ratio = current_age_distribution.get("<1m").cloned().unwrap_or(0.0)
        + current_age_distribution.get("1-3m").cloned().unwrap_or(0.0);
    let maturity = determine_supply_maturity(lth_ratio);
    let hodl_trend = calculate_hodl_trend(&hodl_strength_history);

    HODLWavesResult {
        age_distribution: current_age_distribution,
        hodl_strength: current_hodl_strength,
        supply_maturity: maturity,
        long_term_holder_ratio: lth_ratio,
        recent_activity_ratio,
        hodl_trend,
        timestamps,
    }
}

fn mean(slice: &[f64]) -> f64 {
    if slice.is_empty() { return 0.0; }
    let sum: f64 = slice.iter().copied().sum();
    sum / (slice.len() as f64)
}
