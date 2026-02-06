import streamlit as st
import pandas as pd
import plotly.express as px
from utils import predict_for_state_year, df_ml

st.set_page_config(
    page_title="Groundwater Quality Prediction & Recommendation System",
    layout="wide"
)

# ================= HEADER =================
col_logo, col_title = st.columns([1.2, 8])

with col_logo:
    st.image("https://cdn-icons-png.flaticon.com/512/4148/4148460.png", width=95)

with col_title:
    st.title("Groundwater Quality Prediction & Recommendation System")
    st.write("AI-based forecasting and decision-support tool for sustainable groundwater management.")

# ================= SIDEBAR =================
st.sidebar.image("https://img.icons8.com/fluency/96/filter.png", width=60)
st.sidebar.header("User Input")

states = sorted(df_ml['state'].unique())
state = st.sidebar.selectbox("Select State", states)
year = st.sidebar.slider("Select Future Year", 2022, 2035, 2026)

# ================= PREDICTION =================
if st.sidebar.button("Run Prediction"):

    gqi, quality, risk, recommendation = predict_for_state_year(state, year)

    # ===== EXCLUDED STATES =====
    if gqi is None:
        st.error("Prediction not available for this state")

        st.subheader("Reason for Exclusion from ML Forecasting")
        for reason in recommendation:
            st.write("•", reason)

    # ===== NORMAL PREDICTION =====
    else:
        st.subheader("Prediction Results")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.image("https://cdn-icons-png.flaticon.com/512/2933/2933245.png", width=60)
            st.metric("Predicted GQI", f"{gqi:.2f} / 100")

        with c2:
            st.image("https://cdn-icons-png.flaticon.com/512/4341/4341139.png", width=60)
            st.metric("Quality Class", quality)

        with c3:
            st.image("https://cdn-icons-png.flaticon.com/512/595/595067.png", width=60)
            st.metric("Risk Level", risk)

        st.info("Prediction is based on historical groundwater trends and machine learning forecasting.")

        # =====================================================
        # 📈 GQI TREND GRAPH (NEW FEATURE)
        # =====================================================
        st.subheader("GQI Trend Analysis")

        state_hist = df_ml[df_ml['state'] == state].copy()

        # historical yearly average GQI
        yearly_gqi = state_hist.groupby("year")["GQI"].mean().reset_index()

        # append future predicted GQI
        future_row = pd.DataFrame({
            "year": [year],
            "GQI": [gqi]
        })

        yearly_gqi = pd.concat([yearly_gqi, future_row])

        fig = px.line(
            yearly_gqi,
            x="year",
            y="GQI",
            markers=True,
            title=f"GQI Trend for {state} (2012–{year})"
        )

        fig.update_layout(
            template="plotly_dark",
            title_font_size=22,
            title_x=0.25,
            xaxis_title="Year",
            yaxis_title="Groundwater Quality Index",
            hovermode="x unified"
        )

        fig.update_traces(line=dict(width=4))

        st.plotly_chart(fig, use_container_width=True)

        # =====================================================
        # 🔧 RECOMMENDATIONS
        # =====================================================
        st.subheader("Engineering Recommendations")
        for rec in recommendation:
            st.write("•", rec)

        # =====================================================
        # 📊 DATA TABLE (DROPDOWN)
        # =====================================================
        with st.expander("View Recent State Data"):
            st.dataframe(
                df_ml[df_ml['state'] == state].tail(5),
                use_container_width=True
            )

# ================= FOOTER =================
st.markdown("---")
st.caption("EDUNET Training Project • Groundwater Decision Support System")
