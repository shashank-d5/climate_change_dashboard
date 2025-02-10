import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go

# Load the dataset
df = pd.read_csv(r'C:\Users\home\Downloads\climate_change_dataset.csv')

# Convert columns to numeric where applicable
columns_to_convert = [
    'Year', 'Month', 'Avg_Temp (°C)', 'Max_Temp (°C)', 'Min_Temp (°C)',
    'Precipitation (mm)', 'Humidity (%)', 'Wind_Speed (m/s)',
    'Solar_Irradiance (W/m²)', 'Cloud_Cover (%)', 'CO2_Concentration (ppm)',
    'Latitude', 'Longitude', 'Altitude (m)', 'Proximity_to_Water (km)',
    'Urbanization_Index', 'Vegetation_Index', 'ENSO_Index',
    'Particulate_Matter (µg/m³)', 'Sea_Surface_Temp (°C)'
]

for col in columns_to_convert:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# Remove outliers using IQR
def remove_outliers(df, column):
    Q1, Q3 = df[column].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    df[column] = np.clip(df[column], lower, upper)  # Replace outliers with bounds

for col in df.select_dtypes(include=[np.number]).columns:
    remove_outliers(df, col)

# Streamlit App UI
st.title("Climate Change Data Dashboard")

# Select Year Range
year_range = st.slider(
    'Select Year Range',
    min_value=int(df['Year'].min()),
    max_value=int(df['Year'].max()),
    value=(int(df['Year'].min()), int(df['Year'].max()))
)

# Filter dataset based on selected year range
filtered_df = df[df['Year'].between(year_range[0], year_range[1])]

# Display basic dataset details
st.write("### Filtered Dataset Overview")
st.write("Shape:", filtered_df.shape)
st.write("Missing values:", filtered_df.isnull().sum())

# Train Model and Visualizations
if st.button("Train Model and Generate Predictions"):
    with st.spinner("Training the model and making predictions..."):
        if not filtered_df.empty:
            # Plot 1: Average Temperature Over Years
            fig_temp = px.line(filtered_df, x='Year', y='Avg_Temp (°C)', title="Average Temperature Over Years")
            st.plotly_chart(fig_temp)

            # Plot 2: CO2 Concentration Over Years
            fig_co2 = px.line(filtered_df, x='Year', y='CO2_Concentration (ppm)', title="CO2 Concentration Over Years")
            st.plotly_chart(fig_co2)

            # Plot 3: CO2 Concentration vs. Sea Surface Temperature
            fig_scatter = px.scatter(
                filtered_df,
                x='CO2_Concentration (ppm)',
                y='Sea_Surface_Temp (°C)',
                color='Year',
                title="CO2 Concentration vs. Sea Surface Temperature"
            )
            st.plotly_chart(fig_scatter)

            # Train Model: Predicting Sea Surface Temperature
            X = filtered_df[['CO2_Concentration (ppm)', 'Solar_Irradiance (W/m²)', 'ENSO_Index', 'Precipitation (mm)', 'Humidity (%)']]
            y = filtered_df['Sea_Surface_Temp (°C)']

            if not X.empty and not y.empty:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Plot 4: Actual vs. Predicted Sea Surface Temperature
                fig_actual_vs_pred = go.Figure()
                fig_actual_vs_pred.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predicted'))
                fig_actual_vs_pred.add_trace(go.Scatter(x=y_test, y=y_test, mode='lines', name='Actual', line=dict(color='red')))
                fig_actual_vs_pred.update_layout(title="Actual vs Predicted Sea Surface Temperature",
                                                 xaxis_title="Actual Temperature (°C)",
                                                 yaxis_title="Predicted Temperature (°C)")
                st.plotly_chart(fig_actual_vs_pred)

                # Model Performance Metrics
                mse, r2 = mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred)
                st.write(f"**Mean Squared Error:** {mse:.2f}")
                st.write(f"**R² Score:** {r2:.2f}")

    st.success("Model training and predictions completed!")

