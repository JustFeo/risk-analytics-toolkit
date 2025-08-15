# risk-analytics-toolkit

this is a small package + streamlit app i made for a class project. it does VaR/CVaR, some sims (gbm + ruin), and plots. it’s probably not perfect, but it works on my machine :)

## quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
# or
pip install -r requirements.txt
```

run the app:

```bash
streamlit run app/streamlit_app.py
```

there’s docs in `docs/` but i’m still writing them and probably missing stuff lol. data is from yfinance (SPY) or csv uploads.

not financial advice. pls double check math if you use this for real. 