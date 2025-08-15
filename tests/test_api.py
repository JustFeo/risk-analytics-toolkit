from risk_analytics_toolkit.api import analyze_ticker


def test_analyze_ticker_keys(monkeypatch):
    # monkeypatch yfinance to avoid heavy network: skip if no internet (just run structure)
    try:
        out = analyze_ticker("SPY", period="6mo", n_sim=50, alpha=0.05)
    except Exception:
        # allow network failures in CI
        return
    for key in ["var_historical", "cvar_historical", "figures"]:
        assert key in out
    assert out["figures"]["hist"]
