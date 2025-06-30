
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score

# ==== CONFIGURACIÓN GLOBAL ====
st.set_page_config(page_title="Olistic Delivery & Review Risk", layout="wide")

# ==== ESTILO GLOBAL Y NAV ====
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f0f4ff, #fcfcfc);
    background-attachment: fixed;
}
[data-testid="column"] {
    background: transparent;
}
body {
    background: transparent;
}
html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
    color: #2c3e50;
}
h1, h2, h3 {
    font-family: 'Raleway', sans-serif;
}
.card {
    background: #ffffff;
    padding: 1.2rem;
    border-radius: 12px;
    box-shadow: 0 6px 16px rgba(0,0,0,0.08);
    text-align: center;
    transition: transform 0.2s ease;
}
.card:hover {
    transform: scale(1.03);
    box-shadow: 0 10px 24px rgba(0,0,0,0.1);
}
.navbar {
    position: sticky;
    top: 0;
    background-color: #ffffffee;
    z-index: 999;
    padding: 0.8rem 1rem;
    border-bottom: 1px solid #ddd;
    backdrop-filter: blur(8px);
}
.navbar a {
    margin-right: 1.5rem;
    text-decoration: none;
    font-weight: bold;
    color: #2c3e50;
    font-family: 'Raleway', sans-serif;
}
.navbar a:hover {
    color: #6c63ff;
    text-shadow: 0 0 4px rgba(108,99,255,0.3);
}
</style>

<div class='navbar'>
    <a href="#home">Home</a>
    <a href="#model">Model</a>
    <a href="#risk">Risk Explorer</a>
    <a href="#simulator">Simulator</a>
    <a href="#strategy">Strategy</a>
    <a href="#contact">Contact</a>
</div>
""", unsafe_allow_html=True)

# ==== CARGA DE DATOS ====
@st.cache_data
def load_data():
    orders = pd.read_csv("olist_orders_dataset.csv", parse_dates=[
        "order_purchase_timestamp", "order_approved_at",
        "order_delivered_customer_date", "order_estimated_delivery_date"
    ])
    reviews = pd.read_csv("olist_order_reviews_dataset.csv")
    customers = pd.read_csv("olist_customers_dataset.csv")
    sellers = pd.read_csv("olist_sellers_dataset.csv")
    order_items = pd.read_csv("olist_order_items_dataset.csv")
    return orders, reviews, customers, sellers, order_items

orders, reviews, customers, sellers, order_items = load_data()

# === SECCIÓN: HOME / PORTADA ===
st.markdown("<div id='home'></div>", unsafe_allow_html=True)

st.markdown("""
<h1 style='font-size: 2.8rem; font-weight: 600; font-family: "Raleway", sans-serif; color: #1e3a8a; margin-bottom: 0.3rem;'>Unpacking E-Commerce Efficiency</h1>
<p style='font-size: 1.25rem; color: #444; font-family: "Raleway", sans-serif; margin-top: 0;'>
Welcome to a data-driven platform designed to detect delivery risks and anticipate bad reviews in the Brazilian marketplace.
</p>
""", unsafe_allow_html=True)

st.markdown("""
<a href="#model">
    <button style="background-color: #6c63ff; color: white; padding: 0.9rem 2rem; border-radius: 8px; border: none; font-size: 1.05rem; font-family: 'Raleway', sans-serif; cursor: pointer; transition: all 0.3s ease;">
        🚀 Start Exploring the Risk Model
    </button>
</a>
""", unsafe_allow_html=True)

# === SECCIÓN: DATASET FOUNDATION ===
st.markdown("<div id='dataset-foundation'></div>", unsafe_allow_html=True)
st.markdown("## 📦 Dataset Foundation")
st.markdown("Overview of the marketplace dataset used to power our predictive analytics.")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 🌍 Brazilian Marketplace Map")
    try:
        with open("brazil-states.geojson", "r") as f:
            geojson = json.load(f)

        orders_geo = orders.merge(customers, on="customer_id", how="left")
        state_counts = orders_geo["customer_state"].value_counts().reset_index()
        state_counts.columns = ["estado", "valor"]

        fig = px.choropleth(
            state_counts,
            geojson=geojson,
            locations="estado",
            featureidkey="properties.sigla",
            color="valor",
            color_continuous_scale="Blues",
            title="Total Orders per State"
        )
        fig.update_geos(fitbounds="geojson", visible=False)
        fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)

    except:
        st.warning("⚠️ File 'brazil-states.geojson' not found. Please make sure it's in the same folder.")

with col2:
    st.markdown("### 📊 Key Metrics")

    total_orders = orders.shape[0]
    unique_customers = customers["customer_unique_id"].nunique()
    total_sellers = sellers["seller_id"].nunique()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"<div class='card'><h2>{total_orders:,}</h2><p>Total Orders</p></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card'><h2>2016–2018</h2><p>Time Range</p></div>", unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown(f"<div class='card'><h2>{unique_customers:,}</h2><p>Unique Customers</p></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='card'><h2>{total_sellers:,}</h2><p>Total Sellers</p></div>", unsafe_allow_html=True)
        
        
# === FORECAST DE PEDIDOS DIARIOS ===
from prophet import Prophet
from prophet.plot import plot_plotly

st.markdown("## 📈 Daily Orders Forecast")
st.markdown("This chart anticipates future daily order volumes to help plan operational capacity.")

# Preparar datos por día
orders_by_day = orders.copy()
orders_by_day['order_purchase_date'] = orders_by_day['order_purchase_timestamp'].dt.date
daily_counts = orders_by_day.groupby('order_purchase_date').size().reset_index(name='y')
daily_counts.columns = ['ds', 'y']

# Entrenar modelo Prophet
forecast_model = Prophet()
forecast_model.fit(daily_counts)

# Crear futuro y predecir
future = forecast_model.make_future_dataframe(periods=30)
forecast = forecast_model.predict(future)

# Gráfico interactivo
fig_forecast = plot_plotly(forecast_model, forecast)
fig_forecast.update_layout(
    title="📦 Forecast of Daily Orders (Next 30 Days)",
    xaxis_title="Date",
    yaxis_title="Number of Orders",
    height=450
)
st.plotly_chart(fig_forecast, use_container_width=True)



# === Crear dataset para modelado ===
df = orders.merge(reviews[["order_id", "review_score"]], on="order_id", how="inner")
df["delivery_delay_days"] = (df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]).dt.days
df["delivery_time_days"] = (df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.days
df["bad_review"] = (df["review_score"] <= 2).astype(int)
df = df[df["delivery_delay_days"].notnull()]

# === Modelo 1: REGRESIÓN (entrega tardía)
X_reg = df[["delivery_time_days"]]
y_reg = df["delivery_delay_days"]
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_train_r, y_train_r)
y_pred_r = reg_model.predict(X_test_r)
rmse = mean_squared_error(y_test_r, y_pred_r) ** 0.5

# === Modelo 2: CLASIFICACIÓN (review mala)
X_class = df[["delivery_time_days", "delivery_delay_days"]]
y_class = df["bad_review"]
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_class, y_class, test_size=0.2, random_state=42)
clf_model = LogisticRegression()
clf_model.fit(X_train_c, y_train_c)
y_pred_proba = clf_model.predict_proba(X_test_c)[:, 1]
auc_score = roc_auc_score(y_test_c, y_pred_proba)
 # === MODELO 3: CLASIFICADOR XGBOOST (Pedido llegará tarde) ===
from xgboost import XGBClassifier

# Crear nueva columna binaria de retraso
df["is_late"] = (df["delivery_delay_days"] > 0).astype(int)

# Variables conocidas al momento de compra
# Añade más si las has preprocesado antes en tus notebooks
xgb_data = df.copy()
xgb_data["order_hour"] = df["order_purchase_timestamp"].dt.hour
xgb_data["order_dayofweek"] = df["order_purchase_timestamp"].dt.dayofweek

X_xgb = xgb_data[["delivery_time_days", "order_hour", "order_dayofweek"]]
y_xgb = xgb_data["is_late"]

X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(X_xgb, y_xgb, test_size=0.2, random_state=42)

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
xgb_model.fit(X_train_xgb, y_train_xgb)

xgb_late_proba = xgb_model.predict_proba(X_test_xgb)[:, 1]
xgb_data["xgb_late_prob"] = xgb_model.predict_proba(X_xgb)[:, 1]       
# === SECCIÓN: RISK DASHBOARD ===
st.markdown("<div id='risk'></div>", unsafe_allow_html=True)
st.markdown("## 📊 Risk Explorer Dashboard")
st.markdown("Explore delivery patterns and bad review risk using our trained models.")

tab1 = st.tabs(["📦 Delivery Delay Accuracy"])[0]

# === TAB 1: Predicción vs realidad del retraso ===
with tab1:
    st.markdown("### 📦 Delivery Delay: Predicted vs Actual")

    delay_pred_df = pd.DataFrame({
        "Actual Delay (days)": y_test_r,
        "Predicted Delay (days)": y_pred_r
    })

    fig1 = px.scatter(
        delay_pred_df,
        x="Actual Delay (days)",
        y="Predicted Delay (days)",
        trendline="ols",
        title="Predicted vs Actual Delivery Delay",
        color_discrete_sequence=["#0984e3"]
    )
    st.plotly_chart(fig1, use_container_width=True)


# === SECCIÓN: SIMULADOR DE ESCENARIOS ===
st.markdown("<div id='simulator'></div>", unsafe_allow_html=True)
st.markdown("## 🧪 What-If Risk Simulator")
st.markdown("Adjust delivery timing below to simulate the predicted risk of a bad review.")

# Sliders interactivos
sim_col1, sim_col2 = st.columns(2)
with sim_col1:
    sim_delivery_time = st.slider("📦 Delivery Time (days)", min_value=1, max_value=30, value=10)
with sim_col2:
    sim_delay = st.slider("⏱️ Delivery Delay (Estimated - Real)", min_value=-15, max_value=15, value=0)

# Predicción
sim_input = np.array([[sim_delivery_time, sim_delay]])
sim_pred = clf_model.predict_proba(sim_input)[0][1]

# Resultado visual
st.markdown(f"<div class='card'><h2>{sim_pred:.2%}</h2><p>Predicted Risk of Bad Review</p></div>", unsafe_allow_html=True)

# Gráfico interactivo
fig = px.bar(
    x=["Good Review", "Bad Review"],
    y=[1 - sim_pred, sim_pred],
    color=["Good Review", "Bad Review"],
    text=[f"{(1 - sim_pred):.1%}", f"{sim_pred:.1%}"],
    color_discrete_map={"Good Review": "#2ecc71", "Bad Review": "#e74c3c"}
)
fig.update_layout(
    title="Simulated Review Outcome",
    xaxis_title="Predicted Outcome",
    yaxis_title="Probability",
    showlegend=False
)
st.plotly_chart(fig, use_container_width=True)

# === SECCIÓN: SELLER RISK EXPLORER ===
st.markdown("## 🛍️ Seller Risk Explorer")
st.markdown("Select a seller to analyze their delivery behavior and predicted risk of bad reviews.")

# Combinar datasets para vendedor
seller_df = order_items.merge(df[["order_id", "delivery_time_days", "delivery_delay_days", "bad_review"]], on="order_id", how="inner")
seller_df = seller_df.merge(sellers, on="seller_id", how="left")
seller_df["predicted_risk"] = clf_model.predict_proba(seller_df[["delivery_time_days", "delivery_delay_days"]])[:, 1]

# Selector interactivo
unique_sellers = sorted(seller_df["seller_id"].unique())
selected_seller = st.selectbox("🔍 Select a Seller ID", unique_sellers)

# Filtrar datos del vendedor
filtered = seller_df[seller_df["seller_id"] == selected_seller]
avg_risk = filtered["predicted_risk"].mean()
high_risk_pct = (filtered["predicted_risk"] > 0.7).mean() * 100
total_orders = len(filtered)

k1, k2, k3 = st.columns(3)
k1.markdown(f"<div class='card'><h2>{avg_risk:.2f}</h2><p>Avg Risk Score</p></div>", unsafe_allow_html=True)
k2.markdown(f"<div class='card'><h2>{high_risk_pct:.1f}%</h2><p>Orders > 70% Risk</p></div>", unsafe_allow_html=True)
k3.markdown(f"<div class='card'><h2>{total_orders}</h2><p>Orders Analyzed</p></div>", unsafe_allow_html=True)

# Histograma de riesgo
fig = px.histogram(
    filtered,
    x="predicted_risk",
    nbins=30,
    color_discrete_sequence=["#e74c3c"],
    title=f"Predicted Risk Distribution for Seller {selected_seller}"
)
fig.update_layout(xaxis_title="Predicted Probability", yaxis_title="Orders")
st.plotly_chart(fig, use_container_width=True)

# === SECCIÓN: PEDIDOS DE ALTO RIESGO ===
st.markdown("## 🚨 High-Risk Orders")
st.markdown("This table shows orders with a high predicted probability of delay or bad review.")

# Combinar pedidos, clientes y sellers
merged_orders = orders.merge(
    df[["order_id", "delivery_time_days", "delivery_delay_days", "bad_review"]],
    on="order_id", how="left"
).merge(
    xgb_data[["order_id", "xgb_late_prob"]],
    on="order_id", how="left"
).merge(
    order_items[["order_id", "seller_id"]],
    on="order_id", how="left"
).merge(
    customers[["customer_id", "customer_city"]],
    on="customer_id", how="left"
)

# Filtrar por riesgo > 0.7
high_risk_orders = merged_orders[(merged_orders["xgb_late_prob"] > 0.7)]

# Selección de columnas
high_risk_display = high_risk_orders[[
    "order_id", "customer_city", "seller_id", "xgb_late_prob", "delivery_time_days"
]].drop_duplicates().sort_values(by="xgb_late_prob", ascending=False)

high_risk_display.rename(columns={
    "xgb_late_prob": "Predicted Delay Risk",
    "delivery_time_days": "Delivery Time (days)"
}, inplace=True)

# Mostrar tabla
st.dataframe(high_risk_display.reset_index(drop=True), use_container_width=True)


# === SECCIÓN: MAPA DE RIESGO PRECISO ===


# === COMBINAR CUSTOMERS + COORDENADAS POR CIUDAD ===
geo = pd.read_csv("olist_geolocation_dataset.csv")
customers = pd.read_csv("olist_customers_dataset.csv")

# Calcular lat/lon promedio por ciudad + estado
geo_avg = geo.groupby(["geolocation_city", "geolocation_state"]).agg({
    "geolocation_lat": "mean",
    "geolocation_lng": "mean"
}).reset_index()

geo_avg.columns = ["city", "state", "lat", "lon"]

# Unir coordenadas a los clientes
customers_geo = customers.merge(
    geo_avg,
    left_on=["customer_city", "customer_state"],
    right_on=["city", "state"],
    how="left"
)


st.markdown("## 🗺️ Risk Map by City (Geo-Based)")
st.markdown("This map shows average predicted delivery risk by customer city based on geolocation coordinates.")

try:
    # Combinar xgb_data con customers con coordenadas
    risk_geo = orders.merge(
        xgb_data[["order_id", "xgb_late_prob"]],
        on="order_id", how="left"
    ).merge(
        customers_geo[["customer_id", "customer_city", "lat", "lon"]],
        on="customer_id", how="left"
    )

    # Eliminar ciudades sin datos de riesgo o coordenadas
    city_risk_geo = risk_geo.dropna(subset=["lat", "lon", "xgb_late_prob"])
    risk_by_city = city_risk_geo.groupby(
        ["customer_city", "lat", "lon"]
    )["xgb_late_prob"].mean().reset_index()

    # Crear mapa scatter_geo
    fig = px.scatter_geo(
        risk_by_city,
        lat="lat", lon="lon",
        hover_name="customer_city",
        size="xgb_late_prob",
        color="xgb_late_prob",
        color_continuous_scale="OrRd",
        title="📍 Avg Late Delivery Risk by City (Geolocated)",
        projection="natural earth",
        scope="south america"
    )
    fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.warning(f"⚠️ Could not generate geo risk map: {e}")



# === SECCIÓN FINAL: CONTACTO ===
st.markdown("<div id='contact'></div>", unsafe_allow_html=True)
st.markdown("## 📬 Contact")
st.markdown("For questions, collaboration or feedback:")

st.markdown("""
<style>
.contact-box {
    background: linear-gradient(140deg, #e6f0ff, #ffffff);
    border-radius: 12px;
    padding: 2rem;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.08);
    transition: all 0.3s ease;
}
.contact-box:hover {
    transform: scale(1.01);
    box-shadow: 0px 12px 28px rgba(0,0,0,0.1);
}
.contact-box h3 {
    color: #1e3a8a;
}
.contact-box ul {
    font-size: 1.05rem;
    line-height: 2rem;
    list-style: none;
    padding-left: 0;
}
.contact-box li::before {
    content: "•";
    color: #6c63ff;
    margin-right: 0.6rem;
}
</style>

<div class='contact-box'>
    <h3>👥 Group 1</h3>
    <p style='margin:0;'>Strategic Business Analytics · IE University</p>
    <ul>
        <li>Email: –</li>
        <li>LinkedIn: –</li>
        <li>App: Olistic Delivery Risk Predictor</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<p style='text-align:center; color: #7f8c8d; font-size: 0.95rem;'>
© 2025 · Olistic Predictive Dashboard · Created by Group 1 – IE Strategic Analytics Lab
</p>
""", unsafe_allow_html=True)

