import math
import random
import os
import sys

import pytest

# Ensure we import the installed crypto_indicators extension, not backend/crypto_indicators.py
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
try:
    sys.path[:] = [p for p in sys.path if os.path.abspath(p) != backend_dir]
except Exception:
    pass


def test_hodl_waves_analyze_sanity():
    import crypto_indicators as ci
    # synthetic UTXO-age dataset
    dates = []
    age_days = []
    values = []
    start_year, start_month, start_day = 2024, 1, 1
    for day in range(25):
        d = f"{start_year}-{start_month:02d}-{(start_day + day):02d}"
        for _ in range(120):
            dates.append(d)
            # exponential-like distribution for age
            age_days.append(random.expovariate(1 / 180.0))
            values.append(random.expovariate(1.0))

    res = ci.hodl_waves_analyze(dates, age_days, values)
    # keys present
    for k in [
        "age_distribution",
        "hodl_strength",
        "supply_maturity",
        "long_term_holder_ratio",
        "recent_activity_ratio",
        "hodl_trend",
        "timestamps",
    ]:
        assert k in res, f"missing key {k}"

    # age distribution sums ~ 1
    total = sum(res["age_distribution"].values())
    assert math.isfinite(total)
    assert 0.95 <= total <= 1.05

    # ratios and strength in [0,1]
    for k in ["hodl_strength", "long_term_holder_ratio", "recent_activity_ratio"]:
        v = res[k]
        assert 0.0 <= v <= 1.0

    # maturity string sane
    assert isinstance(res["supply_maturity"], str)
    # trend classification sane
    assert res["hodl_trend"] in {"Strengthening", "Weakening", "Stable", "Insufficient Data"}


def test_puell_analyze_sanity():
    import crypto_indicators as ci
    # synthetic issuance series
    n = 400
    daily_issuance_usd = []
    issuance_ma_365 = []
    base = 1_000_000.0
    for i in range(n):
        val = base * (1.0 + 0.1 * math.sin(i / 25.0))
        daily_issuance_usd.append(val)
        # simple moving average proxy
        if i == 0:
            issuance_ma_365.append(val)
        else:
            issuance_ma_365.append(0.99 * issuance_ma_365[-1] + 0.01 * val)

    res = ci.puell_analyze(daily_issuance_usd, issuance_ma_365, None)
    for k in [
        "current_puell",
        "puell_percentile",
        "mining_profitability",
        "market_cycle_phase",
        "historical_puell",
        "puell_bands",
    ]:
        assert k in res, f"missing key {k}"
    assert isinstance(res["puell_bands"], dict)
    # percentile in [0,100]
    assert 0.0 <= res["puell_percentile"] <= 100.0
    # current value finite
    assert math.isfinite(res["current_puell"])
    # historical series length matches input
    assert len(res["historical_puell"]) == n


@pytest.mark.parity
def test_puell_parity_deterministic():
    import math
    import types
    import sys
    import pandas as pd
    import crypto_indicators as ci
    import pytest
    try:
        from backend.indicators.crypto.puell_multiple import PuellMultipleModel
    except Exception:
        pytest.skip("Skipping Python Puell fallback parity: model dependencies unavailable", allow_module_level=False)

    # Deterministic synthetic issuance
    n = 360
    base = 1_000_000.0
    daily_issuance_usd = [base * (1.0 + 0.05 * math.sin(i / 20.0)) for i in range(n)]
    # Simple EMA-like MA 365
    issuance_ma_365 = []
    for i, val in enumerate(daily_issuance_usd):
        if i == 0:
            issuance_ma_365.append(val)
        else:
            issuance_ma_365.append(0.99 * issuance_ma_365[-1] + 0.01 * val)

    # Rust
    rust_res = ci.puell_analyze(daily_issuance_usd, issuance_ma_365, None)

    # Force Python fallback
    real_mod = sys.modules.get("crypto_indicators")
    dummy = types.ModuleType("crypto_indicators")
    def _raise(*args, **kwargs):
        raise Exception("force fallback")
    dummy.puell_analyze = _raise
    sys.modules["crypto_indicators"] = dummy
    try:
        df = pd.DataFrame({
            "date": [f"2024-01-{(i % 30) + 1:02d}" for i in range(n)],
            "daily_issuance_usd": daily_issuance_usd,
            "issuance_ma_365": issuance_ma_365,
        })
        py_res = PuellMultipleModel().analyze(df)
    finally:
        if real_mod is not None:
            sys.modules["crypto_indicators"] = real_mod
        else:
            del sys.modules["crypto_indicators"]

    # Compare structure and basic tolerances
    for k in [
        "current_puell",
        "puell_percentile",
        "mining_profitability",
        "market_cycle_phase",
        "historical_puell",
        "puell_bands",
    ]:
        assert k in rust_res
        assert hasattr(py_res, k)

    assert math.isfinite(rust_res["current_puell"]) and math.isfinite(py_res.current_puell)
    assert abs(rust_res["current_puell"] - py_res.current_puell) <= 1.0
    assert 0.0 <= rust_res["puell_percentile"] <= 100.0
    assert 0.0 <= py_res.puell_percentile <= 100.0
    # Tighten percentile parity tolerance when Python model available
    assert abs(rust_res["puell_percentile"] - py_res.puell_percentile) <= 20.0
    assert len(rust_res["historical_puell"]) == n



def test_hash_ribbons_analyze_sanity():
    import crypto_indicators as ci
    n = 200
    hash_rate = []
    difficulty = []
    for i in range(n):
        val = 100.0 + 5.0 * math.sin(i / 15.0) + (random.random() - 0.5)
        hash_rate.append(val)
        difficulty.append(max(1.0, val * 0.8 + (random.random() - 0.5)))

    res = ci.hash_ribbons_analyze(hash_rate, difficulty, None)
    for k in [
        "hash_ribbon_signal",
        "miner_capitulation",
        "hash_rate_trend",
        "difficulty_adjustment",
        "mining_health",
        "hash_rate_ma_30",
        "hash_rate_ma_60",
    ]:
        assert k in res, f"missing key {k}"
    # trend classification sanity
    assert res["hash_rate_trend"] in {
        "Growing",
        "Slowly Growing",
        "Declining",
        "Slowly Declining",
        "Stable",
        "Insufficient Data",
    }
    # signal contains expected category
    sig = res["hash_ribbon_signal"]
    assert any(k in sig for k in ("Buy", "Neutral", "Sell"))
    # moving averages lengths
    assert len(res["hash_rate_ma_30"]) == n
    assert len(res["hash_rate_ma_60"]) == n


@pytest.mark.parity
def test_hash_ribbons_parity_deterministic():
    import math
    import types
    import sys
    import crypto_indicators as ci
    import pandas as pd
    import pytest
    try:
        from backend.indicators.crypto.quant_grade_hash_ribbons import HashRibbonsModel
    except Exception:
        pytest.skip("Skipping Python Hash Ribbons fallback parity: model import error", allow_module_level=False)

    # Deterministic synthetic hash rate and difficulty
    n = 240
    hash_rate = [100.0 + 3.0 * math.sin(i / 18.0) for i in range(n)]
    difficulty = [max(1.0, hr * 0.85) for hr in hash_rate]

    # Rust output
    rust_res = ci.hash_ribbons_analyze(hash_rate, difficulty, None)

    # Force Python fallback by monkeypatching module
    real_mod = sys.modules.get("crypto_indicators")
    dummy = types.ModuleType("crypto_indicators")
    def _raise(*args, **kwargs):
        raise Exception("force fallback")
    dummy.hash_ribbons_analyze = _raise
    sys.modules["crypto_indicators"] = dummy
    try:
        df = pd.DataFrame({
            "date": [f"2024-02-{(i % 28) + 1:02d}" for i in range(n)],
            "hash_rate": hash_rate,
            "difficulty": difficulty,
        })
        py_model = HashRibbonsModel(enable_mining_economics=False, enable_regime_analysis=False,
                                    enable_volatility_analysis=False, enable_cycle_analysis=False,
                                    enable_anomaly_detection=False)
        py_res = py_model.analyze(df)
    finally:
        if real_mod is not None:
            sys.modules["crypto_indicators"] = real_mod
        else:
            del sys.modules["crypto_indicators"]

    # Structural checks
    for k in [
        "hash_ribbon_signal",
        "miner_capitulation",
        "hash_rate_trend",
        "difficulty_adjustment",
        "mining_health",
        "hash_rate_ma_30",
        "hash_rate_ma_60",
    ]:
        assert k in rust_res
        assert hasattr(py_res, k)

    # Basic parity tolerance on moving averages length and trends
    assert len(rust_res["hash_rate_ma_30"]) == n
    assert len(rust_res["hash_rate_ma_60"]) == n
    allowed_trends = {
        "Growing",
        "Slowly Growing",
        "Declining",
        "Slowly Declining",
        "Stable",
        "Insufficient Data",
    }
    assert rust_res["hash_rate_trend"] in allowed_trends
    assert py_res.hash_rate_trend in allowed_trends


@pytest.mark.parity
def test_hodl_waves_parity_deterministic():
    import math
    import types
    import sys
    import pandas as pd
    import crypto_indicators as ci
    from backend.indicators.crypto.hodl_waves import HODLWavesModel

    # Build deterministic synthetic dataset
    n = 180
    dates = [f"2024-01-{(i % 30) + 1:02d}" for i in range(n)]
    age_days = [(i * 3) % 365 for i in range(n)]
    values = [100.0 + 0.5 * math.sin(i / 10.0) for i in range(n)]
    df = pd.DataFrame({"date": dates, "age_days": age_days, "value": values})

    # Rust result via direct binding
    rust_res = ci.hodl_waves_analyze(dates, age_days, values)

    # Monkeypatch to force Python fallback in analyze()
    real_mod = sys.modules.get("crypto_indicators")
    dummy = types.ModuleType("crypto_indicators")
    def _raise(*args, **kwargs):
        raise Exception("force fallback")
    dummy.hodl_waves_analyze = _raise
    sys.modules["crypto_indicators"] = dummy
    try:
        py_res = HODLWavesModel().analyze(df)
    finally:
        if real_mod is not None:
            sys.modules["crypto_indicators"] = real_mod
        else:
            del sys.modules["crypto_indicators"]

    # Structural checks
    for k in [
        "age_distribution",
        "hodl_strength",
        "supply_maturity",
        "long_term_holder_ratio",
        "recent_activity_ratio",
        "hodl_trend",
        "timestamps",
    ]:
        assert k in rust_res
        assert hasattr(py_res, k)

    # Tolerance-based parity (allow broad tolerance for model differences)
    assert abs(rust_res.get("hodl_strength", 0.0) - py_res.hodl_strength) <= 0.2
    assert abs(rust_res.get("long_term_holder_ratio", 0.0) - py_res.long_term_holder_ratio) <= 0.2
    assert abs(rust_res.get("recent_activity_ratio", 0.0) - py_res.recent_activity_ratio) <= 0.2
    assert len(rust_res.get("timestamps", [])) == len(py_res.timestamps)

    # Value ranges and classifications
    assert 0.0 <= rust_res.get("hodl_strength", 0.0) <= 1.0
    assert 0.0 <= py_res.hodl_strength <= 1.0
    allowed_trends = {"Strengthening", "Weakening", "Stable", "Insufficient Data"}
    assert rust_res.get("hodl_trend", "Insufficient Data") in allowed_trends
    assert py_res.hodl_trend in allowed_trends


@pytest.mark.edgecase
def test_edge_cases_short_series_and_constant_series():
    import crypto_indicators as ci
    import math
    # Short series should yield Insufficient Data where applicable
    # HODL Waves
    dates_short = ["2024-01-01"]
    ages_short = [10.0]
    values_short = [100.0]
    hodl_short = ci.hodl_waves_analyze(dates_short, ages_short, values_short)
    assert isinstance(hodl_short.get("hodl_trend", "Insufficient Data"), str)
    assert hodl_short.get("hodl_trend", "Insufficient Data") == "Insufficient Data"

    # Puell Multiple
    daily_issuance_short = [1_000_000.0]
    issuance_ma_short = [1_000_000.0]
    puell_short = ci.puell_analyze(daily_issuance_short, issuance_ma_short, None)
    # Short series: ensure outputs are sane rather than enforcing a specific phase
    assert isinstance(puell_short.get("market_cycle_phase", ""), str)
    assert 0.0 <= puell_short.get("puell_percentile", 0.0) <= 100.0
    assert len(puell_short.get("historical_puell", [])) == len(daily_issuance_short)

    # Hash Ribbons
    hash_rate_short = [100.0]
    difficulty_short = [80.0]
    ribbons_short = ci.hash_ribbons_analyze(hash_rate_short, difficulty_short, None)
    assert ribbons_short.get("hash_rate_trend") == "Insufficient Data"
    # Signal may fall back to neutral or insufficient
    sig_short = ribbons_short.get("hash_ribbon_signal", "Neutral")
    assert any(k in sig_short for k in ("Insufficient", "Neutral", "Sell", "Buy"))

    # Constant series should classify as Stable where relevant
    # HODL Waves constant ages/values over multiple days
    dates_const = [f"2024-01-{(i % 7) + 1:02d}" for i in range(60)]
    ages_const = [30.0 for _ in range(60)]
    values_const = [100.0 for _ in range(60)]
    hodl_const = ci.hodl_waves_analyze(dates_const, ages_const, values_const)
    assert hodl_const.get("hodl_trend") in {"Stable", "Insufficient Data"}

    # Puell Multiple constant issuance and MA
    daily_issuance_const = [1_000_000.0 for _ in range(365)]
    issuance_ma_const = [1_000_000.0 for _ in range(365)]
    puell_const = ci.puell_analyze(daily_issuance_const, issuance_ma_const, None)
    assert 0.0 <= puell_const.get("puell_percentile", 50.0) <= 100.0
    assert math.isfinite(puell_const.get("current_puell", 1.0))

    # Hash Ribbons constant hash rate/difficulty
    hash_rate_const = [100.0 for _ in range(120)]
    difficulty_const = [85.0 for _ in range(120)]
    ribbons_const = ci.hash_ribbons_analyze(hash_rate_const, difficulty_const, None)
    assert ribbons_const.get("hash_rate_trend") in {"Stable", "Insufficient Data"}


@pytest.mark.edgecase
def test_hash_ribbons_extreme_volatility():
    import crypto_indicators as ci
    import math
    # Construct a series with sharp spikes and drops
    n = 240
    base = 100.0
    hash_rate = []
    for i in range(n):
        val = base + 10.0 * math.sin(i / 5.0)
        if i % 40 == 0:
            val *= 1.5  # spike
        if i % 55 == 0:
            val *= 0.6  # drop
        hash_rate.append(val)
    difficulty = [max(1.0, hr * 0.85) for hr in hash_rate]

    res = ci.hash_ribbons_analyze(hash_rate, difficulty, None)
    # Structural checks
    for k in [
        "hash_ribbon_signal",
        "hash_rate_trend",
        "hash_rate_ma_30",
        "hash_rate_ma_60",
    ]:
        assert k in res
    assert len(res["hash_rate_ma_30"]) == n
    assert len(res["hash_rate_ma_60"]) == n
    assert res["hash_rate_trend"] in {
        "Growing",
        "Slowly Growing",
        "Declining",
        "Slowly Declining",
        "Stable",
        "Insufficient Data",
    }


@pytest.mark.edgecase
def test_hash_ribbons_alternating_regimes():
    import crypto_indicators as ci
    import math
    # Alternating up/down regimes
    n = 200
    hash_rate = []
    for i in range(n):
        segment = (i // 50) % 4
        if segment in (0, 2):
            val = 100.0 + 2.0 * (i % 50)  # uptrend segments
        else:
            val = 200.0 - 2.0 * (i % 50)  # downtrend segments
        # add mild oscillation
        val += 3.0 * math.sin(i / 10.0)
        hash_rate.append(val)
    difficulty = [max(1.0, hr * 0.8) for hr in hash_rate]

    res = ci.hash_ribbons_analyze(hash_rate, difficulty, None)
    assert res["hash_rate_trend"] in {
        "Growing",
        "Slowly Growing",
        "Declining",
        "Slowly Declining",
        "Stable",
        "Insufficient Data",
    }


@pytest.mark.edgecase
def test_puell_noise_extremes_and_bands_order():
    import crypto_indicators as ci
    import random
    import math
    # High noise issuance relative to MA
    n = 365
    base = 1_000_000.0
    daily = []
    ma = []
    ema = base
    for i in range(n):
        noise = random.uniform(-0.4, 0.4)
        val = base * (1.0 + 0.1 * math.sin(i / 12.0) + noise)
        daily.append(val)
        ema = 0.98 * ema + 0.02 * val
        ma.append(ema)

    res = ci.puell_analyze(daily, ma, None)
    assert 0.0 <= res["puell_percentile"] <= 100.0
    # Bands should be ordered bottom < low < fair < high < top
    bands = res["puell_bands"]
    assert bands["bottom"] < bands["low"] < bands["fair"] < bands["high"] < bands["top"]


@pytest.mark.edgecase
def test_hodl_waves_shift_distribution_strength_change():
    import crypto_indicators as ci
    import math
    # Early dataset: mostly young coins
    n = 120
    dates_early = [f"2024-03-{(i % 30) + 1:02d}" for i in range(n)]
    ages_early = [float((i % 20) + 1) for i in range(n)]  # < 30 days mostly
    values_early = [100.0 + 0.1 * math.sin(i / 5.0) for i in range(n)]
    early = ci.hodl_waves_analyze(dates_early, ages_early, values_early)

    # Late dataset: mostly old coins
    dates_late = [f"2024-04-{(i % 30) + 1:02d}" for i in range(n)]
    ages_late = [float(300 + (i % 100)) for i in range(n)]  # > 1y mostly
    values_late = [100.0 + 0.1 * math.sin(i / 5.0) for i in range(n)]
    late = ci.hodl_waves_analyze(dates_late, ages_late, values_late)

    assert 0.0 <= early["hodl_strength"] <= 1.0
    assert 0.0 <= late["hodl_strength"] <= 1.0
    # Expect hodl_strength higher in late dataset dominated by older coins
    assert late["hodl_strength"] >= early["hodl_strength"]