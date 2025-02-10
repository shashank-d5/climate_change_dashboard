import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go

# Load the dataset
df=pd.read_csv('https://drive.google.com/uc?id=1ouidvn7aisQhRT_Wmf-xlgixMPAGK7J3')

# Data preprocessing
df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
df['Month'] = pd.to_numeric(df['Month'], errors='coerce').astype('Int64')
df['Avg_Temp (°C)'] = pd.to_numeric(df['Avg_Temp (°C)'], errors='coerce').astype(float)
df['Max_Temp (°C)'] = pd.to_numeric(df['Max_Temp (°C)'], errors='coerce').astype(float)
df['Min_Temp (°C)'] = pd.to_numeric(df['Min_Temp (°C)'], errors='coerce').astype(float)
df['Precipitation (mm)'] = pd.to_numeric(df['Precipitation (mm)'], errors='coerce').astype(float)
df['Humidity (%)'] = pd.to_numeric(df['Humidity (%)'], errors='coerce').astype(float)
df['Wind_Speed (m/s)'] = pd.to_numeric(df['Wind_Speed (m/s)'], errors='coerce').astype(float)
df['Solar_Irradiance (W/m²)'] = pd.to_numeric(df['Solar_Irradiance (W/m²)'], errors='coerce').astype(float)
df['Cloud_Cover (%)'] = pd.to_numeric(df['Cloud_Cover (%)'], errors='coerce').astype(float)
df['CO2_Concentration (ppm)'] = pd.to_numeric(df['CO2_Concentration (ppm)'], errors='coerce').astype(float)
df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce').astype(float)
df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce').astype(float)
df['Altitude (m)'] = pd.to_numeric(df['Altitude (m)'], errors='coerce').astype(float)
df['Proximity_to_Water (km)'] = pd.to_numeric(df['Proximity_to_Water (km)'], errors='coerce').astype(float)
df['Urbanization_Index'] = pd.to_numeric(df['Urbanization_Index'], errors='coerce').astype(float)
df['Vegetation_Index'] = pd.to_numeric(df['Vegetation_Index'], errors='coerce').astype(float)
df['ENSO_Index'] = pd.to_numeric(df['ENSO_Index'], errors='coerce').astype(float)
df['Particulate_Matter (µg/m³)'] = pd.to_numeric(df['Particulate_Matter (µg/m³)'], errors='coerce').astype(float)
df['Sea_Surface_Temp (°C)'] = pd.to_numeric(df['Sea_Surface_Temp (°C)'], errors='coerce').astype(float)

# Fill missing values
for column in df.select_dtypes(include=[np.float64, 'Int64']).columns:
    if column in ['Year', 'Month']:  # Special handling for integer columns
        df[column].fillna(int(df[column].median()), inplace=True)
    else:
        df[column].fillna(df[column].mean(), inplace=True)

# Remove outliers
def remove_outliers(df, column):
    Q1 = df[column].dropna().quantile(0.25)
    Q3 = df[column].dropna().quantile(0.75)
    IQR = Q3 - Q1
    lbound = Q1 - 1.5 * IQR
    ubound = Q3 + 1.5 * IQR
    if df[column].dtype == 'Int64':
        lbound = int(np.floor(lbound))
        ubound = int(np.ceil(ubound))
    df.loc[df[column] < lbound, column] = lbound
    df.loc[df[column] > ubound, column] = ubound
    return df

for column in df.select_dtypes(include=[np.float64, 'Int64']).columns:
    df = remove_outliers(df, column)

st.title("Climate Change Data Dashboard")

# Year range selection
year_range = st.slider(
    'Select Year Range',
    min_value=int(df['Year'].min()),
    max_value=int(df['Year'].max()),
    value=(int(df['Year'].min()), int(df['Year'].max()))
)
filtered_df = df[df['Year'].between(year_range[0], year_range[1])]

# Button to trigger model training and predictions
if st.button("Train Model and Generate Predictions"):
    with st.spinner("Training the model and making predictions. Please wait..."):
        # Generate plots
        if not filtered_df.empty:
            # Average Temperature Over Years Plot
            fig_avg_temp = px.line(filtered_df, x='Year', y='Avg_Temp (°C)', title="Average Temperature Over Years")
            fig_avg_temp.update_xaxes(range=[filtered_df['Year'].min(), filtered_df['Year'].max()])  # Adjust x-axis range
            st.plotly_chart(fig_avg_temp)

            # CO2 Concentration Over Years Plot
            fig_co2 = px.line(filtered_df, x='Year', y='CO2_Concentration (ppm)', title="CO2 Concentration Over Years")
            fig_co2.update_xaxes(range=[filtered_df['Year'].min(), filtered_df['Year'].max()])  # Adjust x-axis range
            st.plotly_chart(fig_co2)

            # CO2 Concentration vs. Sea Surface Temperature Plot
            fig_co2_vs_sea_surface = px.scatter(
                filtered_df,
                x='CO2_Concentration (ppm)',
                y='Sea_Surface_Temp (°C)',
                color='Year',
                title="CO2 Concentration vs. Sea Surface Temperature"
            )
            fig_co2_vs_sea_surface.update_xaxes(range=[filtered_df['CO2_Concentration (ppm)'].min(), filtered_df['CO2_Concentration (ppm)'].max()])  # Adjust x-axis range
            fig_co2_vs_sea_surface.update_yaxes(range=[filtered_df['Sea_Surface_Temp (°C)'].min(), filtered_df['Sea_Surface_Temp (°C)'].max()])  # Adjust y-axis range
            st.plotly_chart(fig_co2_vs_sea_surface)

            # Model training and predictions
            X = filtered_df[['CO2_Concentration (ppm)', 'Solar_Irradiance (W/m²)', 'ENSO_Index', 'Precipitation (mm)', 'Humidity (%)']]
            y = filtered_df['Sea_Surface_Temp (°C)']

            if not X.empty and not y.empty:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                regressor = LinearRegression()
                regressor.fit(X_train, y_train)
                y_pred = regressor.predict(X_test)

                # Actual vs Predicted Plot
                fig_actual_vs_predicted = go.Figure()
                fig_actual_vs_predicted.add_trace(go.Scatter(
                    x=y_test,
                    y=y_pred,
                    mode='markers',
                    name='Predicted'
                ))
                fig_actual_vs_predicted.add_trace(go.Scatter(
                    x=y_test,
                    y=y_test,
                    mode='lines',
                    name='Actual',
                    line=dict(color='red')
                ))
                fig_actual_vs_predicted.update_layout(
                    title="Actual vs Predicted Sea Surface Temperature",
                    xaxis_title="Actual Sea Surface Temperature (°C)",
                    yaxis_title="Predicted Sea Surface Temperature (°C)",
                    template="plotly_dark"
                )
                st.plotly_chart(fig_actual_vs_predicted)

                # Show model performance metrics
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                st.write(f"Mean Squared Error: {mse:.2f}")
                st.write(f"R² Score: {r2:.2f}")

    st.success("Model training and predictions completed!")
