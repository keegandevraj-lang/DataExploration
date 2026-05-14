
# ============================================================
# RETIREMENT PROJECTION STREAMLIT APP
# Built from the DAT9X01 Dynamic Fund Optimiser Notebook logic
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf

from itertools import product
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

# ============================================================
# STREAMLIT PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="Retirement Portfolio Projection",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Retirement Portfolio Projection Simulator")
st.caption("Aggressive, Balanced and Conservative projections using the fund modelling logic from the notebook.")

st.markdown(
    """
    This app lets a user enter their current age, expected retirement age, initial investment and monthly investment.
    It then builds three portfolio scenarios using the same core approach as the notebook:
    historical price download, ZAR conversion, return/risk metrics, regression forecasting and portfolio combination analysis.

    **Educational use only. This is not financial advice.**
    """
)

# ============================================================
# 1. USER INPUT PANEL — ADAPTED FROM NOTEBOOK
# ============================================================

START_DATE = "2008-01-01"
END_DATE = None              # None = latest available

RISK_FREE_RATE_ZAR = 0.07
INFLATION_RATE = 0.055
DIVIDEND_TAX_RATE = 0.20
EFFECTIVE_CGT_RATE = 0.18

WEIGHT_STEP = 0.10
MAX_SINGLE_FUND_WEIGHT = 0.70
MIN_ACTIVE_FUNDS = 2

fund_universe = pd.DataFrame([
    {
        "fund": "Satrix Top 40",
        "ticker": "STX40.JO",
        "currency": "ZAR",
        "asset_class": "SA Equity",
        "ter": 0.0040,
        "qual_score": 7,
        "include": True,
        "notes": "SA large-cap equity; verify TER from latest Satrix MDD."
    },
    {
        "fund": "Satrix S&P 500",
        "ticker": "STX500.JO",
        "currency": "ZAR",
        "asset_class": "Global Equity",
        "ter": 0.0030,
        "qual_score": 8,
        "include": True,
        "notes": "Verify Yahoo ticker. Alternative: use VOO in USD and convert to ZAR."
    },
    {
        "fund": "Vanguard S&P 500 ETF",
        "ticker": "VOO",
        "currency": "USD",
        "asset_class": "US Equity",
        "ter": 0.0003,
        "qual_score": 8,
        "include": True,
        "notes": "USD proxy for broad US equity."
    },
    {
        "fund": "Nasdaq 100 ETF Proxy",
        "ticker": "QQQ",
        "currency": "USD",
        "asset_class": "US Growth Equity",
        "ter": 0.0020,
        "qual_score": 7,
        "include": True,
        "notes": "High-growth but concentrated technology exposure."
    },
    {
        "fund": "Gold ETF Proxy",
        "ticker": "GLD",
        "currency": "USD",
        "asset_class": "Gold",
        "ter": 0.0040,
        "qual_score": 6,
        "include": True,
        "notes": "Crisis/inflation hedge; does not generate earnings."
    },
    {
        "fund": "US Bond ETF Proxy",
        "ticker": "BND",
        "currency": "USD",
        "asset_class": "Bonds",
        "ter": 0.0003,
        "qual_score": 6,
        "include": True,
        "notes": "Lower-risk stabiliser, but may reduce long-term return."
    }
])

# ============================================================
# 2. DATA DOWNLOAD FUNCTIONS — FROM NOTEBOOK
# ============================================================

@st.cache_data(show_spinner=False)
def download_adjusted_close(ticker, start=START_DATE, end=END_DATE):
    """Download adjusted close prices from Yahoo Finance."""
    data = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False
    )

    if data.empty:
        raise ValueError(f"No data downloaded for ticker: {ticker}")

    if "Adj Close" in data.columns:
        series = data["Adj Close"].copy()
    else:
        series = data["Close"].copy()

    series = series.dropna()

    # yfinance sometimes returns a DataFrame for a single ticker depending on version.
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]

    series.name = ticker
    return series


def get_fx_series(currency, start=START_DATE, end=END_DATE):
    """Return FX conversion series to ZAR."""
    currency = currency.upper()

    if currency == "ZAR":
        return None

    fx_ticker = f"{currency}ZAR=X"
    fx = download_adjusted_close(fx_ticker, start=start, end=end)
    fx.name = fx_ticker
    return fx


def convert_to_zar(price_series, currency):
    """Convert a price series into ZAR."""
    currency = currency.upper()

    if currency == "ZAR":
        return price_series.copy()

    fx = get_fx_series(currency)
    converted = price_series.mul(fx, axis=0).dropna()
    converted.name = price_series.name
    return converted


@st.cache_data(show_spinner=False)
def download_fund_prices_cached(fund_table_dict):
    """Download and convert all included funds to ZAR."""
    fund_table = pd.DataFrame(fund_table_dict)
    included = fund_table[fund_table["include"] == True].copy()

    price_dict = {}
    failures = []

    for _, row in included.iterrows():
        fund_name = row["fund"]
        ticker = row["ticker"]
        currency = row["currency"]

        try:
            raw = download_adjusted_close(ticker)
            zar = convert_to_zar(raw, currency)
            zar.name = fund_name
            price_dict[fund_name] = zar
        except Exception as e:
            failures.append({
                "fund": fund_name,
                "ticker": ticker,
                "currency": currency,
                "error": str(e)
            })

    if len(price_dict) == 0:
        raise ValueError("No valid fund data downloaded. Check tickers and internet connection.")

    prices = pd.concat(price_dict.values(), axis=1).dropna(how="all")
    prices = prices.ffill().dropna()

    return prices, pd.DataFrame(failures)


# ============================================================
# 4. PERFORMANCE METRICS — FROM NOTEBOOK
# ============================================================

def calculate_cagr(series):
    s = series.dropna()
    years = (s.index[-1] - s.index[0]).days / 365.25
    return (s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1


def calculate_max_drawdown(series):
    s = series.dropna()
    wealth = s / s.iloc[0]
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1
    return drawdown.min()


def regression_forecast(series, years_ahead):
    """Forecast future price using linear regression on log prices."""
    s = series.dropna()

    y = np.log(s.values).reshape(-1, 1)
    X = np.arange(len(s)).reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, y)

    future_days = int(252 * years_ahead)
    future_X = np.array([[len(s) + future_days]])

    predicted_log_price = model.predict(future_X)[0][0]
    predicted_price = float(np.exp(predicted_log_price))

    r2 = r2_score(y, model.predict(X))
    annualised_regression_return = float(np.exp(model.coef_[0][0] * 252) - 1)

    return predicted_price, annualised_regression_return, r2


def after_tax_real_future_value(initial, annual_return, years, cgt_rate, inflation):
    fv_pre_tax = initial * ((1 + annual_return) ** years)
    gain = max(0, fv_pre_tax - initial)
    tax = gain * cgt_rate
    fv_after_tax = fv_pre_tax - tax
    real_fv = fv_after_tax / ((1 + inflation) ** years)
    return fv_pre_tax, tax, fv_after_tax, real_fv


# ============================================================
# 5. FUND SCORING MODEL — FROM NOTEBOOK
# ============================================================

rating_weights = {
    "Return Score": 0.18,
    "Risk Score": 0.14,
    "Drawdown Score": 0.14,
    "Sharpe Score": 0.16,
    "Forecast Score": 0.12,
    "Model Reliability Score": 0.08,
    "Cost Score": 0.08,
    "Qualitative Score Normalised": 0.05,
    "Real Wealth Score": 0.05
}

assert abs(sum(rating_weights.values()) - 1) < 1e-9, "Weights must sum to 1."


def minmax_score(series, higher_is_better=True):
    s = series.astype(float)

    if s.max() == s.min():
        return pd.Series(50, index=s.index)

    score = 100 * (s - s.min()) / (s.max() - s.min())

    if not higher_is_better:
        score = 100 - score

    return score


# ============================================================
# EXTRA APP FUNCTION
# This is the only material extension needed because the original
# notebook projected an initial lump sum. The app needs monthly
# contributions entered by students.
# ============================================================

def project_with_monthly_contributions(initial, monthly, annual_return, years, inflation):
    monthly_return = (1 + annual_return) ** (1 / 12) - 1
    monthly_inflation = (1 + inflation) ** (1 / 12) - 1
    months = int(years * 12)

    value = initial
    total_contributions = initial
    rows = []

    for month in range(months + 1):
        rows.append({
            "Month": month,
            "Year": month / 12,
            "Nominal Value": value,
            "Real Value": value / ((1 + monthly_inflation) ** month),
            "Total Contributions": total_contributions
        })

        if month < months:
            value = value * (1 + monthly_return) + monthly
            total_contributions += monthly

    return pd.DataFrame(rows)


def format_rand(value):
    return f"R{value:,.0f}"


# ============================================================
# SIDEBAR USER INPUTS
# ============================================================

with st.sidebar:
    st.header("Student Inputs")

    age = st.number_input("Current age", min_value=16, max_value=80, value=20, step=1)
    retirement_age = st.number_input("Expected retirement age", min_value=age + 1, max_value=90, value=65, step=1)
    initial_investment = st.number_input("Initial capital investment (R)", min_value=0.0, value=100000.0, step=1000.0)
    monthly_investment = st.number_input("Monthly investment (R)", min_value=0.0, value=2000.0, step=100.0)

    st.divider()

    inflation_rate = st.slider(
        "Inflation assumption",
        min_value=0.00,
        max_value=0.15,
        value=INFLATION_RATE,
        step=0.005,
        format="%.3f"
    )

    cgt_rate = st.slider(
        "Effective CGT rate",
        min_value=0.00,
        max_value=0.30,
        value=EFFECTIVE_CGT_RATE,
        step=0.01,
        format="%.2f"
    )

    run_model = st.button("Run projection", type="primary")

investment_years = int(retirement_age - age)

if investment_years <= 0:
    st.error("Retirement age must be higher than current age.")
    st.stop()

# ============================================================
# RUN MODEL
# ============================================================

if run_model:
    with st.spinner("Downloading market data and running the notebook model logic..."):
        prices_zar, failed_downloads = download_fund_prices_cached(fund_universe.to_dict("records"))
        returns = prices_zar.pct_change().dropna()

        metrics = []

        for fund in prices_zar.columns:
            s = prices_zar[fund].dropna()
            r = returns[fund].dropna()

            cagr = calculate_cagr(s)
            volatility = r.std() * np.sqrt(252)
            sharpe = (cagr - RISK_FREE_RATE_ZAR) / volatility if volatility > 0 else np.nan
            max_dd = calculate_max_drawdown(s)
            predicted_price, reg_return, reg_r2 = regression_forecast(s, investment_years)

            fv_pre_tax, tax, fv_after_tax, real_fv = after_tax_real_future_value(
                initial_investment,
                reg_return,
                investment_years,
                cgt_rate,
                inflation_rate
            )

            meta = fund_universe.set_index("fund").loc[fund]

            metrics.append({
                "Fund": fund,
                "Ticker": meta["ticker"],
                "Asset Class": meta["asset_class"],
                "Currency Source": meta["currency"],
                "CAGR": cagr,
                "Volatility": volatility,
                "Sharpe Ratio": sharpe,
                "Max Drawdown": max_dd,
                "Regression Annual Return": reg_return,
                "Regression R2": reg_r2,
                "TER": meta["ter"],
                "Qualitative Score": meta["qual_score"],
                "Future Value Pre-Tax": fv_pre_tax,
                "Estimated CGT": tax,
                "Future Value After Tax": fv_after_tax,
                "Real Future Value": real_fv
            })

        metrics_df = pd.DataFrame(metrics)

        score_df = metrics_df.copy()

        score_df["Return Score"] = minmax_score(score_df["CAGR"], True)
        score_df["Risk Score"] = minmax_score(score_df["Volatility"], False)
        score_df["Drawdown Score"] = minmax_score(score_df["Max Drawdown"].abs(), False)
        score_df["Sharpe Score"] = minmax_score(score_df["Sharpe Ratio"], True)
        score_df["Forecast Score"] = minmax_score(score_df["Regression Annual Return"], True)
        score_df["Model Reliability Score"] = minmax_score(score_df["Regression R2"], True)
        score_df["Cost Score"] = minmax_score(score_df["TER"], False)
        score_df["Qualitative Score Normalised"] = score_df["Qualitative Score"] * 10
        score_df["Real Wealth Score"] = minmax_score(score_df["Real Future Value"], True)

        score_df["Total Rating"] = sum(
            score_df[col] * weight for col, weight in rating_weights.items()
        )

        score_df["Rating Band"] = pd.cut(
            score_df["Total Rating"],
            bins=[0, 50, 65, 80, 100],
            labels=["Weak", "Acceptable", "Strong", "Excellent"],
            include_lowest=True
        )

        ranked_funds = score_df.sort_values("Total Rating", ascending=False)

        # ============================================================
        # 7. PORTFOLIO COMBINATION ANALYSIS — FROM NOTEBOOK
        # ============================================================

        fund_names = list(prices_zar.columns)

        annual_returns = returns[fund_names].mean() * 252
        cov_matrix = returns[fund_names].cov() * 252

        weight_values = np.round(np.arange(0, 1 + WEIGHT_STEP, WEIGHT_STEP), 10)

        portfolio_results = []

        for weights_tuple in product(weight_values, repeat=len(fund_names)):
            if abs(sum(weights_tuple) - 1) > 1e-9:
                continue

            w = np.array(weights_tuple)

            if w.max() > MAX_SINGLE_FUND_WEIGHT:
                continue

            if np.sum(w > 0) < MIN_ACTIVE_FUNDS:
                continue

            port_return = float(np.dot(w, annual_returns.loc[fund_names]))
            port_vol = float(np.sqrt(np.dot(w.T, np.dot(cov_matrix.loc[fund_names, fund_names], w))))
            port_sharpe = (port_return - RISK_FREE_RATE_ZAR) / port_vol if port_vol > 0 else np.nan

            portfolio_daily_returns = returns[fund_names].dot(w)
            portfolio_wealth = (1 + portfolio_daily_returns).cumprod()
            port_max_dd = calculate_max_drawdown(portfolio_wealth)

            fv_pre_tax, tax, fv_after_tax, real_fv = after_tax_real_future_value(
                initial_investment,
                port_return,
                investment_years,
                cgt_rate,
                inflation_rate
            )

            row = {
                "Expected Return": port_return,
                "Volatility": port_vol,
                "Sharpe Ratio": port_sharpe,
                "Max Drawdown": port_max_dd,
                "Future Value Pre-Tax": fv_pre_tax,
                "Estimated CGT": tax,
                "Future Value After Tax": fv_after_tax,
                "Real Future Value": real_fv
            }

            for fund, weight in zip(fund_names, weights_tuple):
                row[fund] = weight

            portfolio_results.append(row)

        portfolio_df = pd.DataFrame(portfolio_results)

        portfolio_df["Return Score"] = minmax_score(portfolio_df["Expected Return"], True)
        portfolio_df["Risk Score"] = minmax_score(portfolio_df["Volatility"], False)
        portfolio_df["Drawdown Score"] = minmax_score(portfolio_df["Max Drawdown"].abs(), False)
        portfolio_df["Sharpe Score"] = minmax_score(portfolio_df["Sharpe Ratio"], True)
        portfolio_df["Real Wealth Score"] = minmax_score(portfolio_df["Real Future Value"], True)

        portfolio_df["Portfolio Rating"] = (
            0.25 * portfolio_df["Return Score"] +
            0.20 * portfolio_df["Risk Score"] +
            0.20 * portfolio_df["Drawdown Score"] +
            0.25 * portfolio_df["Sharpe Score"] +
            0.10 * portfolio_df["Real Wealth Score"]
        )

        # Three student-facing scenarios derived from optimiser output.
        aggressive = portfolio_df.sort_values("Expected Return", ascending=False).iloc[0].copy()
        balanced = portfolio_df.sort_values("Portfolio Rating", ascending=False).iloc[0].copy()
        conservative = portfolio_df.sort_values(["Volatility", "Max Drawdown"], ascending=[True, False]).iloc[0].copy()

        scenarios = {
            "Aggressive": aggressive,
            "Balanced": balanced,
            "Conservative": conservative
        }

    st.success("Projection complete.")

    if not failed_downloads.empty:
        with st.expander("Failed downloads"):
            st.dataframe(failed_downloads, use_container_width=True)

    # ============================================================
    # SUMMARY CARDS
    # ============================================================

    st.subheader("Scenario Summary")

    summary_rows = []

    projection_frames = []

    for name, scenario in scenarios.items():
        projected = project_with_monthly_contributions(
            initial=initial_investment,
            monthly=monthly_investment,
            annual_return=float(scenario["Expected Return"]),
            years=investment_years,
            inflation=inflation_rate
        )

        projected["Scenario"] = name
        projection_frames.append(projected)

        final = projected.iloc[-1]

        summary_rows.append({
            "Scenario": name,
            "Expected Annual Return": scenario["Expected Return"],
            "Volatility": scenario["Volatility"],
            "Sharpe Ratio": scenario["Sharpe Ratio"],
            "Max Drawdown": scenario["Max Drawdown"],
            "Nominal Value at Retirement": final["Nominal Value"],
            "Real Value at Retirement": final["Real Value"],
            "Total Contributions": final["Total Contributions"]
        })

    summary_df = pd.DataFrame(summary_rows)

    c1, c2, c3 = st.columns(3)
    for col, name in zip([c1, c2, c3], ["Aggressive", "Balanced", "Conservative"]):
        row = summary_df[summary_df["Scenario"] == name].iloc[0]
        with col:
            st.metric(name, format_rand(row["Nominal Value at Retirement"]))
            st.caption(
                f"Real value today: {format_rand(row['Real Value at Retirement'])} | "
                f"Return: {row['Expected Annual Return']:.1%}"
            )

    st.dataframe(
        summary_df.style.format({
            "Expected Annual Return": "{:.2%}",
            "Volatility": "{:.2%}",
            "Sharpe Ratio": "{:.2f}",
            "Max Drawdown": "{:.2%}",
            "Nominal Value at Retirement": "R{:,.0f}",
            "Real Value at Retirement": "R{:,.0f}",
            "Total Contributions": "R{:,.0f}"
        }),
        use_container_width=True
    )

    all_projections = pd.concat(projection_frames, ignore_index=True)

    # ============================================================
    # CHARTS
    # ============================================================

    st.subheader("Charts")

    fig1, ax1 = plt.subplots(figsize=(11, 6))
    for name in scenarios.keys():
        temp = all_projections[all_projections["Scenario"] == name]
        ax1.plot(age + temp["Year"], temp["Nominal Value"], label=name)

    ax1.set_title("Nominal Portfolio Value to Retirement")
    ax1.set_xlabel("Age")
    ax1.set_ylabel("Portfolio Value (R)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(11, 6))
    for name in scenarios.keys():
        temp = all_projections[all_projections["Scenario"] == name]
        ax2.plot(age + temp["Year"], temp["Real Value"], label=name)

    ax2.set_title("Inflation-Adjusted Portfolio Value")
    ax2.set_xlabel("Age")
    ax2.set_ylabel("Value in Today's Rand (R)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    final_values = summary_df.set_index("Scenario")[["Nominal Value at Retirement", "Real Value at Retirement"]]
    final_values.plot(kind="bar", ax=ax3)
    ax3.set_title("Nominal vs Real Value at Retirement")
    ax3.set_ylabel("Rand Value")
    ax3.set_xlabel("")
    ax3.tick_params(axis="x", rotation=0)
    ax3.grid(True, axis="y", alpha=0.3)
    st.pyplot(fig3)

    st.subheader("Scenario Allocations")

    allocation_tabs = st.tabs(["Aggressive", "Balanced", "Conservative"])

    for tab, name in zip(allocation_tabs, ["Aggressive", "Balanced", "Conservative"]):
        with tab:
            scenario = scenarios[name]
            weights = scenario[fund_names].to_frame("Weight")
            weights = weights[weights["Weight"] > 0].sort_values("Weight", ascending=False)
            weights["Rand Allocation of Initial Capital"] = weights["Weight"] * initial_investment

            st.dataframe(
                weights.style.format({
                    "Weight": "{:.0%}",
                    "Rand Allocation of Initial Capital": "R{:,.0f}"
                }),
                use_container_width=True
            )

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(weights.index, weights["Weight"])
            ax.set_title(f"{name} Portfolio Weights")
            ax.set_ylabel("Weight")
            ax.tick_params(axis="x", rotation=45)
            st.pyplot(fig)

    st.subheader("Fund Ratings from Notebook Model")

    display_cols = [
        "Fund", "Asset Class", "CAGR", "Volatility", "Sharpe Ratio", "Max Drawdown",
        "Regression Annual Return", "Regression R2", "TER",
        "Future Value After Tax", "Real Future Value",
        "Total Rating", "Rating Band"
    ]

    st.dataframe(
        ranked_funds[display_cols].style.format({
            "CAGR": "{:.2%}",
            "Volatility": "{:.2%}",
            "Sharpe Ratio": "{:.2f}",
            "Max Drawdown": "{:.2%}",
            "Regression Annual Return": "{:.2%}",
            "Regression R2": "{:.2f}",
            "TER": "{:.2%}",
            "Future Value After Tax": "R{:,.0f}",
            "Real Future Value": "R{:,.0f}",
            "Total Rating": "{:.1f}"
        }),
        use_container_width=True
    )

    csv = all_projections.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download projection results as CSV",
        data=csv,
        file_name="retirement_projection_results.csv",
        mime="text/csv"
    )

else:
    st.info("Enter the inputs in the sidebar and click **Run projection**.")
