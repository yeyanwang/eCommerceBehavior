#!/usr/bin/env python
# coding: utf-8
"""
Fall 2025 Term Project
E-commerce Overview Dashboard
Author: Niha Gupta, Yeyan Wang, Sebastian Henao
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import dash

# --------------------------------------------------
# 1. Load all parquet files
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

files = {
    "daily": "daily_events.parquet",
    "hourly": "hourly_events.parquet",
    "dow": "dow_events.parquet",
    "price_summary": "price_summary.parquet",
    "funnel": "funnel_summary.parquet",
    "brands": "top_brands.parquet",
    "categories": "top_categories.parquet",
    "sessions": "session_histogram.parquet",
    "stats": "session_stats_sample.parquet",
    "price": "price_sample.parquet",
    # Predictive analytics files
    "heatmap": "heatmap_data.parquet",
    "feature_importance": "feature_importance_data.parquet",
    # Clustering analytics
    "pca": "pca.parquet",
    "cluster_features": "cluster_features.parquet"
}

dfs = {}
for key, fname in files.items():
    path = BASE_DIR / fname
    try:
        dfs[key] = pd.read_parquet(path)
        print(f"Loaded {fname} ({dfs[key].shape[0]} rows)")
    except Exception as e:
        print(f"Could not load {fname}: {e}")

# --------------------------------------------------
# 2. Dropdown Options
# --------------------------------------------------
def make_opts(series):
    return [{"label": str(v), "value": str(v)} for v in sorted(series.dropna().unique())]

category_opts = make_opts(dfs["categories"]["category_code"]) if "categories" in dfs else []
brand_opts = make_opts(dfs["brands"]["brand"]) if "brands" in dfs else []

# --------------------------------------------------
# 3. App Initialization
# --------------------------------------------------
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# --------------------------------------------------
# DESCRIPTIVE TAB
# --------------------------------------------------
descriptive_layout = html.Div([
    html.H4("Descriptive Analytics", className="section-subtitle"),
    html.H5("Key Metrics and Trends", className="section-header"),

    html.Div(
        [
            html.Div([html.Div("Total Sessions", className="kpi-title"), html.H3(id="kpi-total", className="kpi-value")], className="kpi-card"),
            html.Div([html.Div("Conversion Rate", className="kpi-title"), html.H3(id="kpi-conv", className="kpi-value")], className="kpi-card"),
            html.Div([html.Div("Avg Events / Session", className="kpi-title"), html.H3(id="kpi-events", className="kpi-value")], className="kpi-card"),
            html.Div([html.Div("Avg Price (All Events)", className="kpi-title"), html.H3(id="kpi-price", className="kpi-value")], className="kpi-card"),
        ],
        className="kpi-row",
    ),

    html.Div([dcc.Graph(id="g-funnel")], className="chart-card static-card"),

    html.Div([
        html.Div([dcc.Graph(id="g-daily")], className="chart-half static-card"),
        html.Div([dcc.Graph(id="g-hourly")], className="chart-half static-card"),
    ], className="charts-row"),

    html.Div([dcc.Graph(id="g-dow")], className="chart-card static-card"),

    html.H4("Interactive Analysis", className="section-subtitle"),

    html.Div([
        html.Div([html.Label("Category"), dcc.Dropdown(id="f-category", options=category_opts, multi=True, placeholder="All")], className="filter-item"),
        html.Div([html.Label("Brand"), dcc.Dropdown(id="f-brand", options=brand_opts, multi=True, placeholder="All")], className="filter-item"),
    ], className="filters-row"),

    html.Div([
        html.Div([dcc.Graph(id="g-brands")], className="chart-half"),
        html.Div([dcc.Graph(id="g-categories")], className="chart-half"),
    ], className="charts-row"),

    html.Div([
        html.Label("Price Range Filter", className="slider-label"),
        dcc.RangeSlider(
            id="price-slider",
            min=0,
            max=2600,
            step=50,
            value=[0, 2600],
            marks=None,
            tooltip={"placement": "bottom", "always_visible": False},
        ),
        html.Div([dcc.Graph(id="g-price-summary")], className="chart-card"),
    ], className="chart-card"),

    html.Div([dcc.Graph(id="g-hist")], className="chart-card"),
])

# --------------------------------------------------
# PREDICTIVE TAB
# --------------------------------------------------
predictive_layout = html.Div([
    html.H4("Predictive Analytics", className="section-subtitle"),
    html.P("This section visualizes model outputs."),

    html.Div([
        html.H5("Predicted Conversion Probability by Day and Hour"),
        dcc.Graph(id="g-heatmap"),
    ], className="chart-card"),

    html.Div([
        html.H5("Feature Influence on Purchase Likelihood"),
        dcc.Graph(id="g-feature-importance"),
    ], className="chart-card"),
])

# --------------------------------------------------
# CLUSTERING TAB
# --------------------------------------------------
clustering_layout = html.Div([
    html.H3("Clustering Analytics"),

    html.Div([dcc.Graph(id="pca-graph")]),

    html.Hr(),

    html.Div(
        id="boxplots-container",
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(2, 1fr)",
            "gap": "20px",
            "marginTop": "20px",
        },
    ),
])

# --------------------------------------------------
# APP LAYOUT
# --------------------------------------------------
app.layout = html.Div([
    html.H2("E-Commerce Behavior Dashboard"),
    dcc.Tabs(
        id="tabs",
        value="tab-descriptive",
        children=[
            dcc.Tab(label="Descriptive Analytics", value="tab-descriptive"),
            dcc.Tab(label="Predictive Analytics", value="tab-predictive"),
            dcc.Tab(label="Clustering Analytics", value="tab-clustering"),
        ],
    ),
    html.Div(id="tabs-content"),
])

# --------------------------------------------------
# TAB SWITCHING
# --------------------------------------------------
@app.callback(Output("tabs-content", "children"), Input("tabs", "value"))
def render_tab_content(tab):
    if tab == "tab-predictive":
        return predictive_layout
    elif tab == "tab-clustering":
        return clustering_layout
    return descriptive_layout

# --------------------------------------------------
# DESCRIPTIVE CALLBACK (with local filtering)
# --------------------------------------------------
@app.callback(
    Output("kpi-total", "children"),
    Output("kpi-conv", "children"),
    Output("kpi-events", "children"),
    Output("kpi-price", "children"),
    Output("g-funnel", "figure"),
    Output("g-daily", "figure"),
    Output("g-hourly", "figure"),
    Output("g-dow", "figure"),
    Output("g-hist", "figure"),
    Output("g-brands", "figure"),
    Output("g-categories", "figure"),
    Output("g-price-summary", "figure"),
    Input("f-category", "value"),
    Input("f-brand", "value"),
    Input("price-slider", "value"),
)
def update_dashboard(categories, brands, price_range):
    daily = dfs["daily"]
    hourly = dfs["hourly"]
    dow = dfs["dow"]
    funnel = dfs["funnel"]
    stats = dfs["stats"]
    brands_df = dfs["brands"]
    cat_df = dfs["categories"]
    price_df = dfs["price"]
    price_summary = dfs["price_summary"]

    total_sessions = int(funnel["unique_sessions"].sum())
    purchases = funnel.loc[funnel["event_type"] == "purchase", "unique_sessions"].values[0]
    views = funnel.loc[funnel["event_type"] == "view", "unique_sessions"].values[0]
    conv_rate = (purchases / views) * 100 if views else 0

    kpi_conv = f"{conv_rate:.2f}%"
    kpi_events = f"{stats['event_count'].mean():.2f}"
    kpi_price = f"${price_summary['mean'].mean():.2f}"

    # LOCAL FILTERING
    categories = categories or []
    brands = brands or []

    cat_df_f = cat_df[cat_df["category_code"].isin(categories)] if categories else cat_df
    brands_df_f = brands_df[brands_df["brand"].isin(brands)] if brands else brands_df

    low, high = price_range
    price_df_f = price_df[(price_df["price"] >= low) & (price_df["price"] <= high)]

    # STATIC CHARTS
    fig_funnel = px.funnel(funnel, y="event_type", x="unique_sessions")
    fig_daily = px.line(daily, x="date", y="count", color="event_type")
    fig_hour = px.bar(hourly, x="hour", y="count", color="event_type")
    fig_dow = px.bar(dow, x="day_of_week", y="count", color="event_type")
    fig_hist = px.histogram(stats, x="event_count", nbins=30)

    # FILTERED CHARTS
    fig_brands = px.bar(brands_df_f, x="purchase_count", y="brand", orientation="h")
    fig_cats = px.bar(cat_df_f, x="purchase_count", y="category_code", orientation="h")
    fig_price = px.histogram(price_df_f, x="price", color="event_type", nbins=50)

    return (
        f"{total_sessions:,}",
        kpi_conv,
        kpi_events,
        kpi_price,
        fig_funnel,
        fig_daily,
        fig_hour,
        fig_dow,
        fig_hist,
        fig_brands,
        fig_cats,
        fig_price,
    )

# --------------------------------------------------
# PREDICTIVE CALLBACK
# --------------------------------------------------
@app.callback(
    Output("g-heatmap", "figure"),
    Output("g-feature-importance", "figure"),
    Input("tabs", "value")
)
def update_predictive_tab(tab):
    if tab != "tab-predictive":
        raise dash.exceptions.PreventUpdate

    heatmap_df = dfs["heatmap"]
    fi_df = dfs["feature_importance"]

    pivot = heatmap_df.pivot(index="dom_dow", columns="dom_hour", values="pred_prob")

    fig_heatmap = px.imshow(
        pivot,
        color_continuous_scale="RdYlGn",
        labels=dict(x="Hour", y="Day", color="Probability"),
    )
    fig_heatmap.update_layout(
        yaxis=dict(categoryorder="array",
                   categoryarray=["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"])
    )

    friendly = {
        "num__evt_view": "Number of Product Views",
        "num__evt_cart": "Number of Cart Additions",
        "num__price_mean": "Average Viewed Price",
        "num__price_max": "Maximum Viewed Price",
        "num__price_min": "Minimum Viewed Price",
        "num__n_products": "Distinct Products Viewed",
        "num__n_brands": "Distinct Brands Viewed",
        "num__n_categories": "Distinct Categories Viewed",
        "num__sess_duration_min": "Session Duration (Minutes)",
        "num__dom_hour": "Hour of Day",
    }

    fi_df = fi_df.copy()
    fi_df["Feature_Label"] = fi_df["Feature"].map(friendly).fillna(fi_df["Feature"])
    fi_df = fi_df[~fi_df["Feature"].str.contains("cat__dom_dow")]

    fi_df["Odds_Ratio"] = np.exp(fi_df["Coefficient"])
    fi_df["Impact_%"] = (fi_df["Odds_Ratio"] - 1) * 100
    fi_df["Effect"] = fi_df["Coefficient"].apply(lambda x: "↑ Increases likelihood" if x > 0 else "↓ Decreases likelihood")
    fi_df = fi_df.sort_values("Coefficient")

    fig_feature = px.bar(
        fi_df,
        x="Coefficient",
        y="Feature_Label",
        color="Effect",
        orientation="h",
        color_discrete_map={
            "↑ Increases likelihood": "#58D68D",
            "↓ Decreases likelihood": "#EC7063",
        },
    )

    return fig_heatmap, fig_feature

# --------------------------------------------------
# CLUSTERING CALLBACK
# --------------------------------------------------
@app.callback(
    Output("pca-graph", "figure"),
    Output("boxplots-container", "children"),
    Input("tabs", "value")
)
def update_clustering_tab(tab):
    if tab != "tab-clustering":
        raise dash.exceptions.PreventUpdate

    pca_df = dfs["pca"]
    cluster_df = dfs["cluster_features"]

    fig_pca = px.scatter_3d(
        pca_df,
        x="PCA1",
        y="PCA2",
        z="PCA3",
        color="cluster",
        opacity=0.7
    )

    numeric_cols = ["purchase_freq", "view_freq", "recency_days", "avg_price_viewed"]
    boxplots = []

    for col in [c for c in numeric_cols if c in cluster_df.columns]:
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(
            y=cluster_df[col],
            x=cluster_df["cluster"],
            name=col,
            boxmean="sd"
        ))
        fig_box.update_layout(
            title=f"{col.replace('_', ' ').title()} by Cluster",
            height=300,
        )
        boxplots.append(dcc.Graph(figure=fig_box))

    return fig_pca, boxplots

# --------------------------------------------------
# RUN APP
# --------------------------------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, debug=True)
