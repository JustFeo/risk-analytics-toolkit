from risk_analytics_toolkit.api import analyze_ticker

if __name__ == "__main__":
    out = analyze_ticker("SPY", period="6mo", n_sim=200)
    print({k: v for k, v in out.items() if k != "figures"})
