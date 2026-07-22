"""
ShopKart Profit Prediction — Streamlit Web App (Premium Cinematic Edition)
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Paths to models
MODEL_PATH = Path("models/best_profit_model.joblib")
SCALER_PATH = Path("models/scaler.joblib")
SCALE_COLS_PATH = Path("models/scale_columns.joblib")
FEATURE_COLS_PATH = Path("models/feature_columns.joblib")

CITIES = ["Bengaluru", "Chennai", "Delhi", "Hyderabad", "Jaipur", "Lucknow", "Mumbai", "Pune"]
CATEGORIES = ["Beauty", "Electronics", "Fashion", "Furniture", "Grocery", "Sports"]
GENDERS = ["Male", "Female"]

CATEGORY_META = {
    "Beauty":      {"icon": "💄", "color": "#FF64B4"},
    "Electronics": {"icon": "🔌", "color": "#00D2FF"},
    "Fashion":     {"icon": "👗", "color": "#9B51E0"},
    "Furniture":   {"icon": "🛋️", "color": "#F2994A"},
    "Grocery":     {"icon": "🛒", "color": "#27AE60"},
    "Sports":      {"icon": "🏸", "color": "#E2B93B"},
}

def inject_premium_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

        /* Global overrides for dark cinematic mode */
        html, body, [class*="css"] { 
            font-family: 'Plus Jakarta Sans', sans-serif; 
        }
        
        .stApp { 
            background: linear-gradient(135deg, #0F0C20 0%, #15102A 50%, #060211 100%);
            color: #E2E8F0;
        }

        /* Smooth UI entry animations */
        @keyframes slideUp {
            from { opacity: 0; transform: translateY(24px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulseGlow {
            0% { box-shadow: 0 0 15px rgba(0, 210, 255, 0.2); }
            50% { box-shadow: 0 0 30px rgba(0, 210, 255, 0.4); }
            100% { box-shadow: 0 0 15px rgba(0, 210, 255, 0.2); }
        }

        /* Header design styling */
        .premium-header {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.07);
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 2rem;
            animation: slideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) both;
        }
        .premium-header .tagline {
            font-family: 'JetBrains Mono', monospace;
            color: #00D2FF;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.2em;
        }
        .premium-header h1 {
            margin: 0.4rem 0;
            font-weight: 800;
            font-size: 2.6rem;
            background: linear-gradient(90deg, #FFFFFF 0%, #A5A1FF 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .premium-header p {
            margin: 0;
            color: #94A3B8;
            font-size: 1.05rem;
        }

        /* Premium Dashboard Containers */
        .glass-card {
            background: rgba(255, 255, 255, 0.02);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 20px;
            padding: 1.8rem;
            margin-bottom: 1.5rem;
            animation: slideUp 0.7s cubic-bezier(0.16, 1, 0.3, 1) both;
        }
        .glass-card h3 {
            margin-top: 0;
            font-weight: 700;
            color: #FFFFFF;
            font-size: 1.2rem;
            letter-spacing: -0.02em;
        }

        /* Sidebar Styling Customizations */
        section[data-testid="stSidebar"] {
            background-color: #0B0816 !important;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }
        section[data-testid="stSidebar"] hr {
            border-color: rgba(255, 255, 255, 0.08);
        }

        /* Action Buttons Overhaul */
        div.stButton > button {
            border-radius: 12px;
            font-weight: 700;
            padding: 0.8rem 1.5rem;
            background: linear-gradient(90deg, #00D2FF 0%, #7928CA 100%) !important;
            border: none !important;
            color: #FFFFFF !important;
            box-shadow: 0 4px 20px rgba(0, 210, 255, 0.25);
            transition: all 0.3s ease;
        }
        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 25px rgba(0, 210, 255, 0.45);
        }

        /* Dynamic Status Indicators */
        .status-pill {
            display: inline-block;
            padding: 0.6rem 1.4rem;
            border-radius: 30px;
            font-weight: 700;
            font-size: 1.4rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            text-align: center;
            width: 100%;
            margin: 1rem 0;
        }
        .status-high {
            background: rgba(39, 174, 96, 0.15);
            color: #2ECC71;
            border: 2px solid #27AE60;
            box-shadow: 0 0 20px rgba(39, 174, 96, 0.3);
        }
        .status-low {
            background: rgba(235, 87, 87, 0.15);
            color: #FF6B6B;
            border: 2px solid #EB5757;
            box-shadow: 0 0 20px rgba(235, 87, 87, 0.3);
        }

        /* Layout List items styling */
        .metric-row {
            display: flex;
            justify-content: space-between;
            padding: 0.75rem 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
            font-size: 0.95rem;
        }
        .metric-row span:first-child { color: #94A3B8; }
        .metric-row span:last-child { color: #F8FAFC; font-weight: 600; font-family: 'JetBrains Mono', monospace; }

        /* Custom Modern Micro Progress Bars */
        .progress-container { margin: 1.5rem 0; }
        .progress-label {
            display: flex; justify-content: space-between;
            font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #94A3B8; margin-bottom: 0.5rem;
        }
        .progress-bar-bg { height: 8px; background: rgba(255, 255, 255, 0.05); border-radius: 4px; overflow: hidden; }
        .progress-bar-fill { height: 100%; border-radius: 4px; transition: width 1s ease-in-out; }
        </style>
        """,
        unsafe_allow_html=True,
    )

def render_interactive_bar(prob_high: float):
    pct = round(prob_high * 100, 1)
    fill_color = "linear-gradient(90deg, #11998e, #38ef7d)" if prob_high >= 0.5 else "linear-gradient(90deg, #ff416c, #ff4b2b)"
    st.markdown(
        f"""
        <div class="progress-container">
            <div class="progress-label">
                <span>Confidence Metric</span>
                <span>{pct}% High Profit Margin Vector</span>
            </div>
            <div class="progress-bar-bg">
                <div class="progress-bar-fill" style="width: {pct}%; background: {fill_color};"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        st.error("Model assets not found. Verify deployment model tracking folder locations.")
        st.stop()
    return (
        joblib.load(MODEL_PATH),
        joblib.load(SCALER_PATH),
        joblib.load(SCALE_COLS_PATH),
        joblib.load(FEATURE_COLS_PATH),
    )

def preprocess_input(record: dict, scaler, scale_columns: list, feature_columns: list) -> pd.DataFrame:
    df = pd.DataFrame([record])
    df["Order_Date"] = pd.to_datetime(df["Order_Date"], format="%d-%m-%y", errors="coerce")
    
    df.loc[(df["Customer_Age"] < 18) | (df["Customer_Age"] > 80), "Customer_Age"] = df["Customer_Age"].median()
    df.loc[df["Qty"] <= 0, "Qty"] = 1
    df.loc[(df["Discount"] < 0) | (df["Discount"] > 100), "Discount"] = 0
    df.loc[df["Delivery"] < 0, "Delivery"] = 1

    df["Month"] = df["Order_Date"].dt.month
    df["Year"] = df["Order_Date"].dt.year
    df["Day_of_Week"] = df["Order_Date"].dt.dayofweek
    df["Weekend"] = df["Day_of_Week"].isin([5, 6]).astype(int)
    df["Profit_Margin"] = (df["Profit"] / df["Sales"]) * 100
    df["Revenue_per_Item"] = df["Sales"] / df["Qty"]

    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
    df = pd.get_dummies(df, columns=["City", "Category"], drop_first=True)
    df[scale_columns] = scaler.transform(df[scale_columns])

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    return df[feature_columns]

def main():
    st.set_page_config(page_title="ShopKart Core Engine", page_icon="⚡", layout="wide")
    inject_premium_css()

    st.markdown(
        """
        <div class="premium-header">
            <div class="tagline">Predictive Analytics Platform</div>
            <h1>ShopKart Revenue & Profit Classifier</h1>
            <p>Leverage operational features to determine margin optimization parameters before fulfillment cycles hit staging.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model, scaler, scale_columns, feature_columns = load_artifacts()

    # --- Sidebar Inputs Design ---
    with st.sidebar:
        st.markdown("### 🛠️ Config Configuration Parameters")
        
        order_date = st.date_input("Order Timestamp Valuation")
        customer_age = st.number_input("Demographic Target Age", min_value=18, max_value=80, value=32)
        gender = st.selectbox("Target Demographics Segment Group", GENDERS)
        city = st.selectbox("Geographic Node Distribution Center", CITIES)
        
        category = st.selectbox(
            "Product Core Inventory Classification Group", CATEGORIES,
            format_func=lambda c: f"{CATEGORY_META.get(c, {}).get('icon', '')} {c}",
        )
        qty = st.number_input("Transactional Inventory Count (Quantity)", min_value=1, value=2)
        unit_price = st.number_input("Unit Value (Base Multiplier in ₹)", min_value=1, value=3500)
        discount = st.slider("Applied Markdown Index Strategy (%)", min_value=0, max_value=100, value=15)
        
        shipping = st.number_input("Logistics Freight Allocation Fee (₹)", min_value=0, value=200)
        delivery = st.number_input("Target Dispatch Timeline (Days Allocation)", min_value=1, max_value=15, value=4)
        rating = st.slider("User Perception Rating Matrix Score", min_value=1, max_value=5, value=4)

        sales = qty * unit_price * (1 - discount / 100)
        profit_estimate = sales * 0.14
        
        st.markdown("---")
        predict_clicked = st.button("RUN PREDICTIVE INFERENCE ENGINE", use_container_width=True)

    # --- Main Application Frame Matrix Display ---
    if not predict_clicked:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                """
                <div class="glass-card">
                    <h3>⚡ Awaiting Evaluation Vectors</h3>
                    <p>Modify parameter data models inside the configuration window sidebar. Trigger inference compilation loops natively to calculate live margins instantly.</p>
                </div>
                """, unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                """
                <div class="glass-card">
                    <h3>🧠 Pipeline Technical Standard</h3>
                    <p>Engine features evaluate dynamic categorical variables using real-time array scaling matrices ($X_{scaled} = \\frac{x - \\mu}{\\sigma}$) to determine high/low operational margins.</p>
                </div>
                """, unsafe_allow_html=True
            )
        return

    # Processing Real-time Inference Engine Data Execution Layer
    record = {
        "Order_Date": order_date.strftime("%d-%m-%y"),
        "Customer_Age": customer_age,
        "Gender": gender,
        "City": city,
        "Category": category,
        "Qty": qty,
        "Unit Price": unit_price,
        "Discount": discount,
        "Shipping": shipping,
        "Delivery": delivery,
        "Sales": sales,
        "Profit": profit_estimate,
        "Rating": rating,
    }

    x_new = preprocess_input(record, scaler, scale_columns, feature_columns)
    prediction = model.predict(x_new)[0]
    proba = model.predict_proba(x_new)[0]
    is_high = prediction == 1

    status_class = "status-high" if is_high else "status-low"
    status_text = "Optimized High Profit Model Output" if is_high else "Risk Warning: Sub-Optimal Low Margin Yield"

    # Layout: Split Analysis Showcase Viewport Workspace 
    panel_left, panel_right = st.columns([1.2, 1])

    with panel_left:
        st.markdown(f'<div class="glass-card"><h3>📈 Production Engine Result Model Output</h3>', unsafe_allow_html=True)
        st.markdown(f'<div class="status-pill {status_class}">{status_text}</div>', unsafe_allow_html=True)
        render_interactive_bar(proba[1])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="glass-card">
                <h3>💼 Evaluated Core Transaction Matrix</h3>
                <div class="metric-row"><span>Gross Computed Run-Time Revenue Valuation</span><span>₹{sales:,.2f}</span></div>
                <div class="metric-row"><span>Assumed Pipeline Scale Base Cost Model (Calculated)</span><span>₹{profit_estimate:,.2f}</span></div>
                <div class="metric-row"><span>Fulfillment Window Index Parameters</span><span>{delivery} Logistics Days Set</span></div>
            </div>
            """, unsafe_allow_html=True
        )

    with panel_right:
        st.markdown(
            f"""
            <div class="glass-card">
                <h3>📋 Encoded Tracking Metadata Elements</h3>
                <div class="metric-row"><span>Distribution Area Domain Node</span><span>{city}</span></div>
                <div class="metric-row"><span>SKU Categorization Variant Group</span><span>{category}</span></div>
                <div class="metric-row"><span>Customer Lifecycle Track Matrix Profile</span><span>{gender} Class Segment, {customer_age} Years</span></div>
                <div class="metric-row"><span>Fulfillment Overhead Target Allocation</span><span>₹{shipping:,.2f}</span></div>
                <div class="metric-row"><span>Promotional Strategy Markdown Index</span><span>{discount}% Base Discount</span></div>
                <div class="metric-row"><span>Current Customer Service Rating Score</span><span>{rating} / 5.0 Index</span></div>
            </div>
            """, unsafe_allow_html=True
        )

    st.markdown("### 📊 Internal Feature Layer Array Mapping Insights")
    t1, t2 = st.tabs(["📊 Live Unstructured Array Sourcing Input", "⚙️ Scaled Processed Matrix Weights Layer ($X_{input}$)"])
    with t1:
        st.dataframe(pd.DataFrame([record]), use_container_width=True)
    with t2:
        st.dataframe(x_new, use_container_width=True)

if __name__ == "__main__":
    main()