import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os, sys

matplotlib.use("Agg")

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #0d0d1a 0%, #1a1a2e 50%, #16213e 100%);
    color: #e0e0e0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f1e 0%, #1a1a2e 100%);
    border-right: 1px solid #e9456022;
}
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }

/* ── Main header ── */
.hero-title {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(90deg, #e94560, #00b4d8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
    line-height: 1.1;
}
.hero-sub {
    font-size: 1rem;
    color: #9090b0;
    margin-bottom: 2rem;
    font-weight: 400;
}

/* ── Metric cards ── */
.metric-card {
    background: linear-gradient(135deg, #1e1e3a, #252545);
    border: 1px solid #e9456033;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-3px);
    border-color: #e94560aa;
}
.metric-label {
    font-size: 0.75rem;
    color: #9090b0;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-size: 1.9rem;
    font-weight: 700;
    color: #e94560;
}
.metric-unit {
    font-size: 0.75rem;
    color: #9090b0;
    margin-top: 0.1rem;
}

/* ── Prediction result box ── */
.result-box {
    background: linear-gradient(135deg, #1a1a35 0%, #12122a 100%);
    border: 2px solid #e94560;
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    box-shadow: 0 0 40px #e9456033;
    margin: 1.5rem 0;
}
.result-label {
    font-size: 1rem;
    color: #9090b0;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.6rem;
}
.result-price {
    font-size: 3.6rem;
    font-weight: 700;
    background: linear-gradient(90deg, #e94560, #ffd60a);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
}
.result-note {
    font-size: 0.8rem;
    color: #9090b0;
    margin-top: 0.8rem;
}

/* ── Section headers ── */
.section-header {
    font-size: 1.3rem;
    font-weight: 600;
    color: #e94560;
    border-left: 3px solid #e94560;
    padding-left: 0.8rem;
    margin: 1.5rem 0 1rem 0;
}

/* ── Info pill ── */
.info-pill {
    display: inline-block;
    background: #e9456022;
    border: 1px solid #e9456055;
    color: #e94560;
    border-radius: 20px;
    padding: 0.2rem 0.9rem;
    font-size: 0.8rem;
    font-weight: 500;
    margin: 0.2rem;
}

/* ── Predict button ── */
.stButton > button {
    background: linear-gradient(135deg, #e94560, #c0303f) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 20px #e9456044 !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 28px #e9456077 !important;
}

/* ── Inputs ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] > div > div > input {
    background: #1e1e3a !important;
    border: 1px solid #0f3460 !important;
    color: #e0e0e0 !important;
    border-radius: 8px !important;
}
[data-testid="stSlider"] > div { color: #e0e0e0; }
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background-color: #e94560 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    gap: 0.5rem;
}
.stTabs [data-baseweb="tab"] {
    background: #1e1e3a !important;
    border-radius: 8px 8px 0 0 !important;
    color: #9090b0 !important;
    border: 1px solid #0f3460 !important;
    border-bottom: none !important;
}
.stTabs [aria-selected="true"] {
    background: #e9456022 !important;
    color: #e94560 !important;
    border-color: #e9456055 !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: #1a1a2e;
    border: 1px solid #0f3460;
    border-radius: 0 8px 8px 8px;
    padding: 1.5rem;
}

/* ── Dividers ── */
hr { border-color: #0f3460 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d0d1a; }
::-webkit-scrollbar-thumb { background: #e94560; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


MODEL_PATH   = "model_weights.npz"
DATASET_PATH = "data.csv"

FUEL_TYPES = [
    "premium unleaded (required)",
    "premium unleaded (recommended)",
    "regular unleaded",
    "flex-fuel (unleaded/E85)",
    "flex-fuel (premium unleaded required/E85)",
    "flex-fuel (premium unleaded recommended/E85)",
    "diesel",
    "electric",
    "natural gas",
]
TRANSMISSIONS = ["AUTOMATIC", "MANUAL", "AUTOMATED_MANUAL", "DIRECT_DRIVE", "UNKNOWN"]
DRIVE_TYPES   = ["front wheel drive", "rear wheel drive", "all wheel drive", "four wheel drive"]
SIZES         = ["Compact", "Midsize", "Large"]
STYLES        = [
    "Sedan", "Coupe", "Convertible", "Wagon", "4dr Hatchback", "2dr Hatchback",
    "4dr SUV", "2dr SUV", "Crew Cab Pickup", "Extended Cab Pickup", "Regular Cab Pickup",
    "Cargo Van", "Passenger Van", "Cargo Minivan", "Passenger Minivan", "Convertible SUV",
]

REFERENCE_YEAR = 2017
CYLINDER_OPTIONS = [2, 3, 4, 5, 6, 8, 10, 12, 16]


def _clip_int(value, low, high, fallback):
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        return fallback
    return int(min(max(number, low), high))


def _choice(value, options, fallback):
    text = "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value).strip()
    return text if text in options else fallback


def flags_from_row(row) -> dict:
    text = "" if pd.isna(row["Market Category"]) else str(row["Market Category"])
    fuel = "" if pd.isna(row["Engine Fuel Type"]) else str(row["Engine Fuel Type"]).lower()
    return {
        "lux": "Luxury" in text,
        "perf": "Performance" in text,
        "exo": "Exotic" in text,
        "die": ("Diesel" in text) or ("diesel" in fuel),
        "flex": ("Flex Fuel" in text) or ("flex-fuel" in fuel),
    }


def representative_row(df: pd.DataFrame, make: str, model: str, year: int):
    sub = df[(df["Make"] == make) & (df["Model"] == model)]
    if sub.empty:
        sub = df[df["Make"] == make]
    if sub.empty:
        return None
    nearest = sub.assign(_yd=(sub["Year"] - year).abs())
    return nearest.sort_values(["_yd", "MSRP"]).iloc[0]


def apply_listing_defaults(df: pd.DataFrame | None):
    """Fill spec fields from the closest listing for the chosen brand, model, and year."""
    if df is None or "brand" not in st.session_state:
        return
    make = st.session_state["brand"]
    models = sorted(df.loc[df["Make"] == make, "Model"].dropna().unique().tolist())
    model = st.session_state.get("model_name")
    if model not in models:
        model = models[0] if models else ""
        st.session_state["model_name"] = model
    year = int(st.session_state.get("year", 2015))
    row = representative_row(df, make, model, year)
    if row is None:
        return
    st.session_state["hp"] = _clip_int(row["Engine HP"], 50, 1500, 200)
    cyl = _clip_int(row["Engine Cylinders"], 2, 16, 4)
    st.session_state["cyl"] = min(CYLINDER_OPTIONS, key=lambda option: abs(option - cyl))
    doors = _clip_int(row["Number of Doors"], 2, 4, 4)
    st.session_state["doors"] = doors if doors in (2, 3, 4) else 4
    st.session_state["fuel"] = _choice(row["Engine Fuel Type"], FUEL_TYPES, "regular unleaded")
    st.session_state["trans"] = _choice(row["Transmission Type"], TRANSMISSIONS, "AUTOMATIC")
    st.session_state["drive"] = _choice(row["Driven_Wheels"], DRIVE_TYPES, DRIVE_TYPES[0])
    st.session_state["size"] = _choice(row["Vehicle Size"], SIZES, "Midsize")
    st.session_state["style"] = _choice(row["Vehicle Style"], STYLES, "Sedan")
    st.session_state["hwy"] = _clip_int(row["highway MPG"], 8, 100, 28)
    st.session_state["city"] = _clip_int(row["city mpg"], 5, 80, 20)
    st.session_state["pop"] = _clip_int(row["Popularity"], 1, 6000, 1500)
    st.session_state.update(flags_from_row(row))


def on_brand_change():
    df = load_dataset(DATASET_PATH)
    if df is None:
        return
    make = st.session_state.get("brand")
    models = sorted(df.loc[df["Make"] == make, "Model"].dropna().unique().tolist())
    st.session_state["model_name"] = models[0] if models else ""
    apply_listing_defaults(df)


def on_listing_change():
    apply_listing_defaults(load_dataset(DATASET_PATH))


@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=True)
    return {
        "intercept":    float(data["intercept"][0]),
        "coefficients": data["coefficients"],
        "feature_names": list(data["feature_names"]),
        "scaler_mean":  data["scaler_mean"],
        "scaler_std":   data["scaler_std"],
    }

@st.cache_data
def load_dataset(path: str):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df[df["MSRP"] > 500]
    df = df[df["MSRP"] < df["MSRP"].quantile(0.99)]
    return df

def predict(model, input_dict: dict) -> float:
    x = np.zeros(len(model["feature_names"]))
    for i, name in enumerate(model["feature_names"]):
        if name in input_dict:
            x[i] = float(input_dict[name])
    x_sc = (x - model["scaler_mean"]) / model["scaler_std"]
    log_price = model["intercept"] + model["coefficients"] @ x_sc
    return float(max(np.expm1(log_price), 0.0))

def build_input_dict(fuel, transmission, drive, size, style,
                     hp, cylinders, doors, hwy_mpg, city_mpg,
                     popularity, year, is_luxury, is_performance,
                     is_exotic, is_diesel, is_flexfuel) -> dict:
    d = {
        "Engine HP":         hp,
        "Engine Cylinders":  cylinders,
        "Number of Doors":   doors,
        "highway MPG":       hwy_mpg,
        "city mpg":          city_mpg,
        "Popularity":        popularity,
        "Car Age":           REFERENCE_YEAR - year,
        "Is Luxury":         int(is_luxury),
        "Is Performance":    int(is_performance),
        "Is Exotic":         int(is_exotic),
        "Is Diesel":         int(is_diesel),
        "Is FlexFuel":       int(is_flexfuel),
        f"Engine Fuel Type_{fuel}":           1,
        f"Transmission Type_{transmission}":  1,
        f"Driven_Wheels_{drive}":             1,
        f"Vehicle Size_{size}":               1,
        f"Vehicle Style_{style}":             1,
    }
    return d

def fig_to_st(fig):
    return fig

def make_coef_chart(model, top_n=15):
    coefs = model["coefficients"]
    names = model["feature_names"]
    abs_c = np.abs(coefs)
    idx   = np.argsort(abs_c)[::-1][:top_n]
    top_names = [names[i] for i in idx][::-1]
    top_coefs = coefs[idx][::-1]
    colors = ["#e94560" if c > 0 else "#00b4d8" for c in top_coefs]

    fig, ax = plt.subplots(figsize=(9, 6), facecolor="#1a1a2e")
    ax.set_facecolor("#16213e")
    bars = ax.barh(range(len(top_names)), top_coefs, color=colors, edgecolor="none", alpha=0.9)
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names, fontsize=8.5, color="#e0e0e0")
    ax.axvline(0, color="#9090b0", lw=1, ls="--")
    ax.set_xlabel("Coefficient (log-price scale)", color="#9090b0", fontsize=9)
    ax.set_title(f"Top-{top_n} Feature Importances", color="#e94560", fontsize=12, fontweight="bold")
    ax.tick_params(colors="#9090b0")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#0f3460")
    ax.spines["bottom"].set_color("#0f3460")
    ax.grid(True, axis="x", alpha=0.3, color="#0f3460")

    from matplotlib.patches import Patch
    legend = [Patch(facecolor="#e94560", label="Increases price"),
              Patch(facecolor="#00b4d8", label="Decreases price")]
    ax.legend(handles=legend, loc="lower right", facecolor="#1a1a2e",
              edgecolor="#0f3460", labelcolor="#e0e0e0", fontsize=8)
    plt.tight_layout()
    return fig

def make_dist_chart(df, predicted_price):
    msrp = df["MSRP"].dropna()
    fig, ax = plt.subplots(figsize=(9, 4), facecolor="#1a1a2e")
    ax.set_facecolor("#16213e")
    ax.hist(msrp, bins=60, color="#00b4d8", edgecolor="none", alpha=0.5, label="Dataset MSRP")
    ax.axvline(predicted_price, color="#e94560", lw=2.5, ls="--", label=f"Your Car: ${predicted_price:,.0f}")
    ax.axvline(msrp.median(), color="#ffd60a", lw=1.5, ls=":", label=f"Median: ${msrp.median():,.0f}")
    ax.set_xlabel("MSRP (USD)", color="#9090b0")
    ax.set_ylabel("Count", color="#9090b0")
    ax.set_title("Your Prediction vs Dataset Distribution", color="#e94560", fontsize=12, fontweight="bold")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k"))
    ax.tick_params(colors="#9090b0")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#0f3460")
    ax.spines["bottom"].set_color("#0f3460")
    ax.legend(facecolor="#1a1a2e", edgecolor="#0f3460", labelcolor="#e0e0e0", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3, color="#0f3460")
    plt.tight_layout()
    return fig


def main():
    model = load_model(MODEL_PATH)
    df    = load_dataset(DATASET_PATH)

    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 1rem 0 1.5rem 0;'>
            <div style='font-size:3rem;'>🚗</div>
            <div style='font-size:1.2rem; font-weight:700; color:#e94560;'>Car Price Predictor</div>
            <div style='font-size:0.75rem; color:#9090b0; margin-top:0.3rem;'>Multiple Linear Regression</div>
        </div>
        """, unsafe_allow_html=True)
        brands = ["BMW"]
        if df is not None:
            brands = sorted(df["Make"].dropna().unique().tolist())
        if "brand" not in st.session_state:
            st.session_state["brand"] = "Toyota" if "Toyota" in brands else brands[0]
            make = st.session_state["brand"]
            model_seed = []
            if df is not None:
                model_seed = sorted(df.loc[df["Make"] == make, "Model"].dropna().unique().tolist())
            st.session_state["model_name"] = model_seed[0] if model_seed else ""
            st.session_state["year"] = 2015
            st.session_state["mileage"] = 30000
            apply_listing_defaults(df)

        st.markdown('<div class="section-header">Car Details</div>', unsafe_allow_html=True)
        brand = st.selectbox("Brand", brands, key="brand", on_change=on_brand_change)
        model_options = [st.session_state.get("model_name") or ""]
        if df is not None:
            model_options = sorted(df.loc[df["Make"] == brand, "Model"].dropna().unique().tolist()) or model_options
        if st.session_state.get("model_name") not in model_options:
            st.session_state["model_name"] = model_options[0]
            apply_listing_defaults(df)
        model_name = st.selectbox("Model", model_options, key="model_name", on_change=on_listing_change)
        year = st.slider("Model Year", 1990, 2017, key="year", on_change=on_listing_change)
        mileage = st.number_input(
            "Mileage (miles)",
            min_value=0,
            max_value=400000,
            step=1000,
            key="mileage",
            help="Collected as specified in the proposal. This Kaggle set is new-car MSRP and has no odometer column, so mileage is not a trained feature.",
        )
        hp = st.number_input(
            "Engine capacity (HP)",
            min_value=50,
            max_value=1500,
            step=5,
            key="hp",
            help="The dataset records engine output in horsepower, which is the capacity measure used by the model.",
        )
        fuel = st.selectbox("Fuel Type", FUEL_TYPES, key="fuel")
        transmission = st.selectbox("Transmission", TRANSMISSIONS, key="trans")

        with st.expander("More specifications", expanded=False):
            st.caption("Filled from the closest listing for this brand, model, and year. Adjust them if the car differs.")
            cylinders = st.selectbox("Engine Cylinders", CYLINDER_OPTIONS, key="cyl")
            doors = st.selectbox("Number of Doors", [2, 3, 4], key="doors")
            drive = st.selectbox("Drive Type", DRIVE_TYPES, key="drive")
            size = st.selectbox("Vehicle Size", SIZES, key="size")
            style = st.selectbox("Vehicle Style", STYLES, key="style")
            hwy_mpg = st.slider("Highway MPG", 8, 100, key="hwy")
            city_mpg = st.slider("City MPG", 5, 80, key="city")
            popularity = st.slider("Brand Popularity Score", 1, 6000, key="pop")
            col1, col2 = st.columns(2)
            with col1:
                is_luxury = st.checkbox("Luxury", key="lux")
                is_exotic = st.checkbox("Exotic", key="exo")
                is_diesel = st.checkbox("Diesel", key="die")
            with col2:
                is_performance = st.checkbox("Performance", key="perf")
                is_flexfuel = st.checkbox("Flex Fuel", key="flex")

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("Predict Price", key="predict_btn")

        st.markdown("""
        <hr>
        <div style='font-size:0.72rem; color:#9090b0; text-align:center; line-height:1.6;'>
            FAST-NUCES Karachi &nbsp;|&nbsp; BAI-3B<br>
            Hunain · Hassan · Vinod<br>
            Programming For AI
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="hero-title">Car Price Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Multiple Linear Regression &nbsp;|&nbsp; OLS Normal Equation &nbsp;|&nbsp; 44 Features &nbsp;|&nbsp; 10,990 samples</div>', unsafe_allow_html=True)

    if model is None:
        st.error("Model weights not found. Please run `main.py` first to train the model.", icon="⚠️")
        st.stop()

    tab_predict, tab_insights, tab_about = st.tabs(["🎯  Predict", "📊  Model Insights", "ℹ️  About"])

    with tab_predict:
        if predict_btn or "last_price" in st.session_state:
            if predict_btn:
                input_dict = build_input_dict(
                    fuel, transmission, drive, size, style,
                    hp, cylinders, doors, hwy_mpg, city_mpg,
                    popularity, year, is_luxury, is_performance,
                    is_exotic, is_diesel, is_flexfuel
                )
                price = predict(model, input_dict)
                st.session_state["last_price"] = price
                st.session_state["last_input"] = input_dict

            price = st.session_state["last_price"]
            median_msrp = df["MSRP"].median() if df is not None else 27000
            diff  = price - median_msrp
            diff_pct = (diff / median_msrp) * 100

            st.markdown(f"""
            <div class="result-box">
                <div class="result-label">Estimated Market Price</div>
                <div class="result-price">${price:,.0f}</div>
                <div class="result-note">
                    {'▲' if diff >= 0 else '▼'} ${abs(diff):,.0f} ({abs(diff_pct):.1f}%) {'above' if diff >= 0 else 'below'} dataset median
                </div>
            </div>
            """, unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            cards = [
                ("Brand", brand, model_name),
                ("Model Year", str(year), f"{REFERENCE_YEAR - year} yrs old"),
                ("Mileage", f"{mileage:,.0f}", "miles"),
                ("Engine", f"{hp}", "hp"),
            ]
            for col, (label, val, unit) in zip([c1, c2, c3, c4], cards):
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{val}</div>
                        <div class="metric-unit">{unit}</div>
                    </div>
                    """, unsafe_allow_html=True)

            if df is not None:
                st.markdown('<div class="section-header">Price vs Dataset Distribution</div>', unsafe_allow_html=True)
                fig_dist = make_dist_chart(df, price)
                st.pyplot(fig_dist, use_container_width=True)
                plt.close(fig_dist)

            pill_items = [fuel, transmission.replace("_", " ").title()]
            if is_luxury:      pill_items.append("Luxury")
            if is_performance: pill_items.append("Performance")
            if is_exotic:      pill_items.append("Exotic")
            if is_diesel:      pill_items.append("Diesel")
            if is_flexfuel:    pill_items.append("Flex Fuel")
            pills_html = " ".join(f'<span class="info-pill">{p}</span>' for p in pill_items)
            st.markdown(f"**Entered specs:** {pills_html}", unsafe_allow_html=True)
            st.caption(
                f"{brand} {model_name}, {year}, {mileage:,.0f} miles, {hp} hp, {fuel}, {transmission.replace('_', ' ').title()}. "
                "Mileage is stored with the request. The trained model uses brand popularity and the vehicle specifications, because the listings have no odometer column."
            )

            st.markdown('<div class="section-header">What Drives This Price</div>', unsafe_allow_html=True)
            st.markdown("Coefficients from the fitted linear model. Red raises the estimated price, blue lowers it.")
            fig_coef_live = make_coef_chart(model, top_n=12)
            st.pyplot(fig_coef_live, use_container_width=True)
            plt.close(fig_coef_live)

        else:
            st.markdown("""
            <div style='text-align:center; padding:5rem 2rem; color:#9090b0;'>
                <div style='font-size:4rem; margin-bottom:1rem;'>🚗</div>
                <div style='font-size:1.2rem; font-weight:600; margin-bottom:0.5rem; color:#e0e0e0;'>Configure your vehicle</div>
                <div style='font-size:0.9rem;'>Choose brand, year, mileage, engine capacity, fuel, and transmission, then click <strong style='color:#e94560;'>Predict Price</strong></div>
            </div>
            """, unsafe_allow_html=True)

    with tab_insights:
        st.markdown('<div class="section-header">Feature Importances (OLS Coefficients)</div>', unsafe_allow_html=True)
        st.markdown("The bars below show how much each feature shifts the predicted log-price. Red = raises price, Blue = lowers price.")
        fig_coef = make_coef_chart(model, top_n=15)
        st.pyplot(fig_coef, use_container_width=True)
        plt.close(fig_coef)

        st.markdown('<div class="section-header">Model Performance Metrics</div>', unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        metric_data = [
            ("MAE", "$9,957", "Mean Abs Error"),
            ("RMSE", "$18,425", "Root Mean Sq Err"),
            ("R²", "0.715", "Variance Explained"),
            ("Adj-R²", "0.709", "Penalised R²"),
            ("CV R²", "0.699 ± 0.024", "5-Fold Cross-Val"),
        ]
        for col, (label, val, sub) in zip([c1, c2, c3, c4, c5], metric_data):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value" style="font-size:1.3rem;">{val}</div>
                    <div class="metric-unit">{sub}</div>
                </div>
                """, unsafe_allow_html=True)

        if os.path.exists("plots/5_model_comparison.png"):
            st.markdown('<div class="section-header">Model Comparison (OLS vs Ridge vs Lasso)</div>', unsafe_allow_html=True)
            st.image("plots/5_model_comparison.png", use_container_width=True)

        if os.path.exists("plots/3_actual_vs_predicted_OLS.png"):
            st.markdown('<div class="section-header">Actual vs Predicted & Residuals</div>', unsafe_allow_html=True)
            st.image("plots/3_actual_vs_predicted_OLS.png", use_container_width=True)

        if os.path.exists("plots/2_correlation_heatmap.png"):
            st.markdown('<div class="section-header">Correlation Heatmap</div>', unsafe_allow_html=True)
            st.image("plots/2_correlation_heatmap.png", use_container_width=True)

        if os.path.exists("plots/7_eda_price_vs_features.png"):
            st.markdown('<div class="section-header">EDA: MSRP vs Key Features</div>', unsafe_allow_html=True)
            st.image("plots/7_eda_price_vs_features.png", use_container_width=True)

    with tab_about:
        col_a, col_b = st.columns([1.2, 1])
        with col_a:
            st.markdown("""
            <div class="section-header">Project Overview</div>

            <p style="color:#c0c0d0; line-height:1.8;">
            This Car Price Predictor uses <strong style="color:#e94560;">Multiple Linear Regression</strong>
            with the Ordinary Least Squares (OLS) Normal Equation to estimate a vehicle's market price from
            its specifications. It was built as a full end-to-end ML pipeline — from raw data to predictions.
            </p>

            <div class="section-header">OLS Normal Equation</div>
            <div style="background:#0f0f1e; border:1px solid #0f3460; border-radius:10px; padding:1rem 1.5rem; font-family:monospace; color:#ffd60a; font-size:1rem; text-align:center; margin-bottom:1rem;">
                &beta; = (X<sup>T</sup>X)<sup>-1</sup> X<sup>T</sup>y
            </div>
            <p style="color:#9090b0; font-size:0.85rem; line-height:1.7;">
                &beta; = coefficient vector &nbsp;|&nbsp; X = feature matrix &nbsp;|&nbsp; y = log(MSRP).<br>
                Implemented from scratch using NumPy — no sklearn for regression.
            </p>

            <div class="section-header">Pipeline Steps</div>
            """, unsafe_allow_html=True)

            steps = [
                ("1", "Data Loading", "11,914 raw rows from data.csv"),
                ("2", "Cleaning", "715 duplicates removed, 99th pct outlier cap"),
                ("3", "Feature Engineering", "Car Age, market flags, one-hot encoding"),
                ("4", "Train/Test Split", "80% train / 20% test via sklearn"),
                ("5", "OLS Fit", "Normal Equation β=(XᵀX)⁻¹Xᵀy"),
                ("6", "Evaluation", "MAE, RMSE, R², 5-fold CV"),
                ("7", "Visualizations", "7 plots saved to ./plots/"),
            ]
            for num, title, desc in steps:
                st.markdown(f"""
                <div style="display:flex; align-items:flex-start; margin-bottom:0.6rem;">
                    <div style="background:#e94560; color:white; border-radius:50%; width:24px; height:24px; display:flex; align-items:center; justify-content:center; font-size:0.75rem; font-weight:700; flex-shrink:0; margin-right:0.75rem; margin-top:0.1rem;">{num}</div>
                    <div>
                        <div style="color:#e0e0e0; font-weight:600; font-size:0.9rem;">{title}</div>
                        <div style="color:#9090b0; font-size:0.8rem;">{desc}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with col_b:
            st.markdown("""
            <div class="section-header">Tech Stack</div>
            """, unsafe_allow_html=True)
            tech = [("🐍", "Python 3"), ("🐼", "pandas"), ("🔢", "NumPy"),
                    ("📊", "Matplotlib / Seaborn"), ("🤖", "scikit-learn (split only)"),
                    ("🌐", "Streamlit")]
            for icon, name in tech:
                st.markdown(f"""
                <div style="display:flex; align-items:center; background:#1e1e3a; border:1px solid #0f3460; border-radius:8px; padding:0.5rem 1rem; margin-bottom:0.4rem;">
                    <span style="font-size:1.3rem; margin-right:0.75rem;">{icon}</span>
                    <span style="color:#e0e0e0; font-size:0.9rem;">{name}</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<div class="section-header">Dataset</div>', unsafe_allow_html=True)
            st.markdown("""
            <div style="background:#1e1e3a; border:1px solid #0f3460; border-radius:10px; padding:1rem 1.2rem; color:#c0c0d0; font-size:0.85rem; line-height:1.9;">
                <b style="color:#e94560;">Source:</b> Kaggle Car Features & MSRP<br>
                <b style="color:#e94560;">Rows:</b> 10,990 (after cleaning)<br>
                <b style="color:#e94560;">Features:</b> 44 (after engineering)<br>
                <b style="color:#e94560;">Target:</b> MSRP (log-transformed)<br>
                <b style="color:#e94560;">Years:</b> 1990 – 2017
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="section-header">Team</div>', unsafe_allow_html=True)
            team = [("Mohammed Hunain", "25k-0068"), ("Hassan Ali Khan", "25k-0061"), ("Vinod Kumar", "25k-0114")]
            for name, roll in team:
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between; align-items:center; background:#1e1e3a; border:1px solid #0f3460; border-radius:8px; padding:0.5rem 1rem; margin-bottom:0.4rem;">
                    <span style="color:#e0e0e0; font-size:0.9rem;">👤 {name}</span>
                    <span style="color:#9090b0; font-size:0.8rem; font-family:monospace;">{roll}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("""
            <div style="text-align:center; color:#9090b0; font-size:0.78rem; margin-top:1rem;">
                FAST-NUCES Karachi &nbsp;|&nbsp; Section BAI-3B<br>Programming For AI
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
