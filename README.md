
# Retirement Portfolio Projection Streamlit App

This is a simple Streamlit app built around the same core modelling logic used in the `DAT9X01_Dynamic_Fund_Optimiser_Notebook.ipynb`.

## What the app asks the student to enter

- Current age
- Expected retirement age
- Initial capital investment
- Monthly investment
- Inflation assumption
- Effective capital gains tax assumption

## What the app produces

- Aggressive portfolio projection
- Balanced portfolio projection
- Conservative portfolio projection
- Nominal value at retirement
- Inflation-adjusted value at retirement
- Total contributions
- Portfolio allocation tables
- Fund rating table
- Charts:
  - Nominal portfolio growth
  - Real/inflation-adjusted portfolio growth
  - Nominal vs real retirement value
  - Portfolio allocation weights

## Important model note

The original notebook projected the value of an initial lump-sum investment.  
This app keeps the notebook's core download, ZAR conversion, return, risk, regression, scoring and portfolio optimisation logic, but adds one required function for monthly contributions because the app asks users to enter monthly investments.

## Run locally

Install the requirements:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

## Host online and create a QR code

The easiest route is:

1. Create a free GitHub repository.
2. Upload:
   - `app.py`
   - `requirements.txt`
   - `README.md`
3. Go to Streamlit Community Cloud.
4. Create a new app from your GitHub repository.
5. Copy the deployed app URL.
6. Use any QR code generator to turn the deployed app URL into a QR code.

Students should scan the QR code and open the app in their browser.
