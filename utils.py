import pandas as pd
import joblib

# =========================================================
# LOAD MODEL + DATA
# =========================================================
reg = joblib.load("rf_gqi_model.pkl")
df_ml = pd.read_csv("cleaned_data.csv")

# =========================================================
# STATES EXCLUDED FROM FORECASTING (DATA INSUFFICIENCY)
# =========================================================
excluded_states = {
    "HARYANA": [
        "Only 1 year of available data",
        "Flat conductivity profile",
        "No temporal information for trend learning"
    ],
    "JHARKHAND": [
        "Only 2 years of data",
        "Completely constant plot",
        "Dominated by imputed values"
    ],
    "DELHI": [
        "Only 3 years of data",
        "Artificially linear increasing trend",
        "Pattern influenced by median imputation"
    ],
    "NAGALAND": [
        "Around 4 years of observations",
        "Near-flat trend with minimal variation",
        "Insufficient signal for meaningful modeling"
    ],
    "TELANGANA": [
        "4 years of data",
        "Short and unstable temporal segment",
        "No reliable long-term pattern"
    ]
}

# =========================================================
# FEATURES USED BY MODEL
# =========================================================
features = [
    'year',
    'ph',
    'conductivity',
    'tds',
    'ph_trend',
    'conductivity_trend',
    'tds_trend',
    'ph_3yr_avg',
    'tds_3yr_avg'
]

# =========================================================
# QUALITY + RISK (0–100 GQI SCALE)
# =========================================================
def quality_class(gqi):
    if gqi >= 70:
        return "Good"
    elif gqi >= 40:
        return "Moderate"
    else:
        return "Poor"

def risk_level(gqi):
    if gqi < 40:
        return "High Risk"
    elif gqi < 70:
        return "Moderate Risk"
    else:
        return "Low Risk"

# =========================================================
# RECOMMENDATION ENGINE
# =========================================================
def engineering_recommendation(row):

    ph = row['ph'].iloc[0]
    tds = row['tds'].iloc[0]
    cond = row['conductivity'].iloc[0]
    ph_trend = row['ph_trend'].iloc[0]
    tds_trend = row['tds_trend'].iloc[0]
    tds_avg = row['tds_3yr_avg'].iloc[0]

    actions = []

    if ph < 6.5:
        actions += [
            "Introduce limestone dosing to neutralize acidic groundwater",
            "Promote rainwater harvesting to dilute aquifers"
        ]

    elif ph > 8.5:
        actions += [
            "Implement aquifer dilution through recharge",
            "Promote gypsum treatment in agriculture"
        ]

    if ph_trend > 0.05:
        actions.append("pH rising over time → start long-term alkalinity monitoring")

    if tds > 500:
        actions += [
            "Deploy community RO filtration plants",
            "Restrict untreated industrial discharge"
        ]

    if tds > 1000:
        actions.append("Declare groundwater non-potable and supply alternative drinking water")

    if tds_trend > 20:
        actions.append("Rapid salinity rise → start artificial recharge using low-TDS water")

    if tds_avg > 700:
        actions.append("Introduce managed aquifer recharge & groundwater blending")

    if cond > 750:
        actions.append("High mineralization → enforce industrial effluent treatment")

    if cond > 1500:
        actions.append("Severe contamination → initiate groundwater remediation planning")

    if not actions:
        actions.append("Maintain periodic groundwater monitoring")

    actions.append("Encourage drip irrigation and sustainable agriculture")

    return sorted(set(actions))

# =========================================================
# PREPARE MODEL INPUT
# =========================================================
from sklearn.linear_model import LinearRegression
import numpy as np

def prepare_input(state, year):

    state_df = df_ml[df_ml['state'].str.upper() == state.upper()].copy()

    if len(state_df) < 4:
        raise ValueError(f"Not enough historical data for {state}")

    # ========= TRAIN MINI TIME SERIES MODELS =========
    X_year = state_df[['year']]

    ph_model = LinearRegression().fit(X_year, state_df['ph'])
    tds_model = LinearRegression().fit(X_year, state_df['tds'])
    cond_model = LinearRegression().fit(X_year, state_df['conductivity'])

    # ========= FUTURE CHEMISTRY FORECAST =========
    future_year = np.array([[year]])

    future_ph = ph_model.predict(future_year)[0]
    future_tds = tds_model.predict(future_year)[0]
    future_cond = cond_model.predict(future_year)[0]

    # ========= SCIENTIFIC TREND CALCULATION =========
    ph_trend = ph_model.coef_[0]
    tds_trend = tds_model.coef_[0]
    cond_trend = cond_model.coef_[0]

    # rolling averages (still useful)
    ph_avg = state_df['ph'].tail(3).mean()
    tds_avg = state_df['tds'].tail(3).mean()

    # ========= BUILD FINAL MODEL INPUT =========
    row = {
        "year": year,
        "ph": future_ph,
        "conductivity": future_cond,
        "tds": future_tds,
        "ph_trend": ph_trend,
        "conductivity_trend": cond_trend,
        "tds_trend": tds_trend,
        "ph_3yr_avg": ph_avg,
        "tds_3yr_avg": tds_avg
    }

    return pd.DataFrame([row])



# =========================================================
# MAIN PREDICTION FUNCTION
# =========================================================
def predict_for_state_year(state, year):

    if state.upper() in excluded_states:
        return None, None, None, excluded_states[state.upper()]

    row = prepare_input(state, year)
    gqi = float(reg.predict(row)[0])
    quality = quality_class(gqi)
    risk = risk_level(gqi)
    recommendation = engineering_recommendation(row)

    return gqi, quality, risk, recommendation
