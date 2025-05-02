# import libraries *****************************************************************
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import seaborn as sns
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pickle

import sys

sys.path.append("../src")
from transformers.transformers_l import (
    DropColumnsTransformer,
    CityBasedImputer,
    CityMapTransformer,
    RollingAverageTransformer,
)

# page configuration ***************************************************************
st.set_page_config(
    page_title="DengAI: Predicting Disease Spread",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded",
)

alt.themes.enable("dark")

st.title("🦟 DengAI: Predicting Disease Spread - a Challenge by DrivenData")

with st.sidebar:
    st.markdown("**Team members:** Mine, Lucas & Vicky")  # Add this line at the top

    # Optional: Add some spacing
    st.markdown("---")

# read the data ********************************************************************
df = pd.read_csv("full_data_frame.csv")


# bubble chart *********************************************************************


def plot_bubble_chart(df):
    st.subheader("Yearly Average Dengue Cases by City")

    # City label mapping
    city_label_map = {"sj": "San Juan", "iq": "Iquitos"}
    # Create label list for sidebar
    display_city_names = [
        city_label_map.get(city, city) for city in df["city"].unique()
    ]
    city_map_reverse = {v: k for k, v in city_label_map.items()}

    # Sidebar: City selection with clear label
    selected_display_cities = st.sidebar.multiselect(
        "Select City (for Bubble Chart by Year)",
        display_city_names,
        default=display_city_names,
        key="bubble_chart_city_selector",
    )

    # Convert display names back to city codes
    selected_cities = [
        city_map_reverse.get(city, city) for city in selected_display_cities
    ]

    # Filter the dataframe
    df_filtered = df[df["city"].isin(selected_cities)].copy()

    # Map city names for plotting
    df_filtered["city_label"] = df_filtered["city"].map(city_label_map)

    # Group by year and city (aggregate weekly data to yearly)
    yearly_df = (
        df_filtered.groupby(["year", "city_label"])
        .agg(
            {
                "total_cases": "sum",
                "reanalysis_avg_temp_k": "mean",
                "precipitation_amt_mm": "mean",
            }
        )
        .reset_index()
    )

    # Plotly bubble chart
    fig = px.scatter(
        yearly_df,
        x="year",
        y="total_cases",
        size="precipitation_amt_mm",
        color="reanalysis_avg_temp_k",
        hover_name="city_label",
        facet_col="city_label" if len(selected_cities) > 1 else None,
        color_continuous_scale="RdBu_r",
        size_max=40,
    )

    fig.update_layout(
        yaxis_title="Total Dengue Cases",
        coloraxis_colorbar=dict(title="Avg Temp (K)"),
    )

    # Update all x-axes to have "Year" as the title
    for axis in fig.layout:
        if axis.startswith("xaxis"):
            fig.layout[axis].title = dict(text="Year")

    # Remove "city_label=" from subplot titles
    if "annotations" in fig.layout:
        for annotation in fig.layout.annotations:
            if "city_label=" in annotation.text:
                annotation.text = annotation.text.replace("city_label=", "")

    st.plotly_chart(fig)


plot_bubble_chart(df)

# weekly bubbly chart *************************************************************************


def plot_weekly_bubble_chart(df):
    st.subheader("Weekly Dengue Cases by City")

    # Year selector with unique key
    available_years = sorted(df["year"].unique())
    available_years = [round(year) for year in available_years]  # Round available years
    selected_year = st.sidebar.selectbox(
        "Select Year (for Bubble Chart by Week)",
        available_years,
        key="weekly_chart_year_selector",
    )

    # Optionally, round the selected year to a specific value
    selected_year = round(selected_year, -1)  # Example: rounding to the nearest decade

    # City selector with explicit names in the dropdown but use short names internally
    city_mapping = {"San Juan": "sj", "Iquitos": "iq"}
    cities = df["city"].unique()
    selected_cities = st.sidebar.multiselect(
        "Select City (for Bubble Chart by Week)",
        options=["San Juan", "Iquitos"],  # Showing full names in the dropdown
        default=["San Juan", "Iquitos"],  # Default values as full names
        key="weekly_chart_city_selector",
    )

    # Map selected full city names to their short versions
    selected_city_codes = [city_mapping[city] for city in selected_cities]

    # Filter for selected year and cities (based on short city names)
    df_filtered = df[
        (df["year"] == selected_year) & (df["city"].isin(selected_city_codes))
    ].copy()

    # Plotly bubble chart
    fig = px.scatter(
        df_filtered,
        x="weekofyear",
        y="total_cases",
        size="precipitation_amt_mm",
        color="reanalysis_avg_temp_k",
        hover_name="city",
        facet_col="city" if len(selected_cities) > 1 else None,
        color_continuous_scale="RdBu_r",
        size_max=30,
    )

    # Custom facet titles
    facet_titles = {"sj": "San Juan", "iq": "Iquitos"}

    for facet in fig.select_annotations():
        city_code = facet.text.split("=")[-1].strip()
        facet.text = facet_titles.get(
            city_code, facet.text
        )  # Replace with explicit city name

    fig.update_layout(
        xaxis_title="Week of Year",
        yaxis_title="Total Dengue Cases",
        coloraxis_colorbar=dict(title="Avg Temp (K)"),
    )

    # Set the x-axis title for all subplots (if faceting is used)
    if len(selected_cities) > 1:
        for i in range(len(selected_cities)):
            fig.update_layout({f"xaxis{i + 1}": {"title": "Week of Year"}})

    st.plotly_chart(fig)


plot_weekly_bubble_chart(df)

# histogram of target variable *************************************************************


def plot_total_cases_distribution(df):
    st.subheader("Distribution of Target Variable (Total Dengue Cases)")

    # Sidebar dropdown with friendly labels
    option = st.sidebar.selectbox(
        "Select City (for Distribution Chart)",
        ["Total (All Cities)", "San Juan", "Iquitos"],
    )

    # Map selection to data
    if option == "Total (All Cities)":
        data_raw = df["total_cases"]
        label = "Total"
    elif option == "San Juan":
        data_raw = df[df["city"] == "sj"]["total_cases"]
        label = "San Juan"
    elif option == "Iquitos":
        data_raw = df[df["city"] == "iq"]["total_cases"]
        label = "Iquitos"

    # Drop non-positive values for log-transform
    data_nonzero = data_raw[data_raw > 0]
    data_log = np.log(data_nonzero)

    tabs = st.tabs(["Raw Total Cases", "Log-Transformed Total Cases"])

    with tabs[0]:
        fig_raw = make_dist_and_boxplot(data_raw, f"Total Cases ({label})")
        st.plotly_chart(fig_raw)

    with tabs[1]:
        fig_log = make_dist_and_boxplot(data_log, f"Log(Total Cases) ({label})")
        st.plotly_chart(fig_log)


def make_dist_and_boxplot(data_series, title):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05
    )

    # Histogram
    fig.add_trace(
        go.Histogram(
            x=data_series, nbinsx=50, marker_color="steelblue", name="Histogram"
        ),
        row=1,
        col=1,
    )

    # Boxplot
    fig.add_trace(
        go.Box(
            x=data_series,
            boxpoints="outliers",
            marker_color="indianred",
            name="Boxplot",
            orientation="h",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=500,
        title_text=f"{title} Distribution and Boxplot",
        showlegend=False,
    )
    return fig


plot_total_cases_distribution(df)


# heatmap **************************************************************


def plot_heatmap(df):
    st.subheader("Correlation Heatmap")  # Smaller than st.title()

    # Select only numerical columns
    numerical_df = df.select_dtypes(include=["number"])

    # Calculate the correlation matrix
    corr_matrix = numerical_df.corr()

    # Set up the matplotlib figure with a larger size for better readability
    fig, ax = plt.subplots(figsize=(14, 12))  # Larger figure size for better spacing

    # Generate the heatmap with smaller font size for annotations
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",  # Color palette
        ax=ax,
        fmt=".2f",  # Format numbers to 2 decimal places
        annot_kws={"size": 10},  # Smaller font size for the correlation numbers
        vmin=-1,
        vmax=1,  # Correlation range
        cbar_kws={"shrink": 0.8},  # Adjust color bar size
        linewidths=0.5,  # Add separation lines
        linecolor="gray",  # Color for the grid lines
        cbar=True,  # Show color bar
        square=True,  # Make squares instead of rectangular cells
    )

    # Rotate the x and y axis labels for better readability
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(rotation=0, ha="right", fontsize=12)

    # Display the heatmap
    st.pyplot(fig)


# Call the function to display the heatmap
plot_heatmap(df)

# NDVI ****************************************************************


def plot_ndvi_line_chart(df):
    st.subheader("Vegetation Trends Over Time by Region")  # Smaller than st.title()

    # City label mapping
    city_label_map = {"sj": "San Juan", "iq": "Iquitos"}
    # Create label list for sidebar
    display_city_names = [
        city_label_map.get(city, city) for city in df["city"].unique()
    ]
    city_map_reverse = {v: k for k, v in city_label_map.items()}

    # Sidebar: City selection
    selected_display_cities = st.sidebar.multiselect(
        "Select City (for Vegetation Index Chart)",
        display_city_names,
        default=display_city_names,
    )

    # Sidebar: Show trend lines only
    show_trend_only = st.sidebar.checkbox("Show only trends", value=False)

    # Convert display names back to codes
    selected_cities = [
        city_map_reverse.get(city, city) for city in selected_display_cities
    ]

    # Filter the dataframe
    df_filtered = df[df["city"].isin(selected_cities)].copy()

    # Group by year and compute mean NDVI
    df_filtered = (
        df_filtered.groupby(["year", "city"])[
            ["ndvi_ne", "ndvi_nw", "ndvi_se", "ndvi_sw"]
        ]
        .mean()
        .reset_index()
    )

    # Reshape to long format
    df_long = df_filtered.melt(
        id_vars=["year", "city"],
        value_vars=["ndvi_ne", "ndvi_nw", "ndvi_se", "ndvi_sw"],
        var_name="Region",
        value_name="NDVI",
    )

    # Rename regions
    region_mapping = {
        "ndvi_ne": "Northeast",
        "ndvi_nw": "Northwest",
        "ndvi_se": "Southeast",
        "ndvi_sw": "Southwest",
    }
    df_long["Region"] = df_long["Region"].replace(region_mapping)

    # Create base figure
    fig = go.Figure()

    if not show_trend_only:
        # Add actual NDVI lines
        for region in df_long["Region"].unique():
            df_region = df_long[df_long["Region"] == region]
            fig.add_trace(
                go.Scatter(
                    x=df_region["year"],
                    y=df_region["NDVI"],
                    mode="lines+markers",
                    name=region,
                )
            )

    # Add trend lines
    for region in df_long["Region"].unique():
        df_region = df_long[df_long["Region"] == region]
        z = np.polyfit(df_region["year"], df_region["NDVI"], 1)
        p = np.poly1d(z)
        trendline = p(df_region["year"])
        fig.add_trace(
            go.Scatter(
                x=df_region["year"],
                y=trendline,
                mode="lines",
                name=f"{region} Trend",
                line=dict(dash="dash"),
            )
        )

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Vegetation index",
        legend_title_text=None,
    )

    # Show plot
    st.plotly_chart(fig)


# Call the function to plot the NDVI line chart
plot_ndvi_line_chart(df)


# model prediction feature **************************************************************


def load_model_predictor():
    st.header("🧠 Predict Dengue Cases")

    import logging

    # Load the model
    try:
        import joblib

        model = joblib.load("baseline_model.pkl")
    except FileNotFoundError:
        st.error("Model file 'baseline_model.pkl' not found.")
        return

    st.markdown("Enter input data to get a predicted number of dengue cases.")

    # Input fields
    city_map = {"San Juan": "sj", "Iquitos": "iq"}
    city_reverse_map = {"sj": 0, "iq": 1}  # example mapping for encoding

    selected_city = st.selectbox("Select City", list(city_map.keys()))
    year = st.number_input("Enter Year", min_value=1990, max_value=2100, value=2005)
    weekofyear = st.number_input(
        "Enter Week of Year", min_value=1, max_value=53, value=1
    )

    # Optional: trigger prediction with a button
    if st.button("Predict Cases"):
        # Prepare input for the model
        # You might need to match the input features the model expects
        city_encoded = city_reverse_map[city_map[selected_city]]
        input_data = np.array([[city_encoded, year, weekofyear]])

        logging.info("Predicting now...")
        try:
            prediction = model.predict(input_data)
            st.success(f"✅ Predicted number of dengue cases: **{int(prediction[0])}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")


load_model_predictor()
