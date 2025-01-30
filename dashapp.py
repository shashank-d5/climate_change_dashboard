#Interactive Dashboard using Dash
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df=pd.read_csv(r'C:\Users\home\Downloads\climate_change_dataset.csv')

df['Year']=pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
df['Month']=pd.to_numeric(df['Month'], errors='coerce').astype('Int64')
df['Avg_Temp (°C)']=pd.to_numeric(df['Avg_Temp (°C)'], errors='coerce').astype(float)
df['Max_Temp (°C)']=pd.to_numeric(df['Max_Temp (°C)'], errors='coerce').astype(float)
df['Min_Temp (°C)']=pd.to_numeric(df['Min_Temp (°C)'], errors='coerce').astype(float)
df['Precipitation (mm)']=pd.to_numeric(df['Precipitation (mm)'], errors='coerce').astype(float)
df['Humidity (%)']=pd.to_numeric(df['Humidity (%)'], errors='coerce').astype(float)
df['Wind_Speed (m/s)']=pd.to_numeric(df['Wind_Speed (m/s)'], errors='coerce').astype(float)
df['Solar_Irradiance (W/m²)']=pd.to_numeric(df['Solar_Irradiance (W/m²)'], errors='coerce').astype(float)
df['Cloud_Cover (%)']=pd.to_numeric(df['Cloud_Cover (%)'], errors='coerce').astype(float)
df['CO2_Concentration (ppm)']=pd.to_numeric(df['CO2_Concentration (ppm)'], errors='coerce').astype(float)
df['Latitude']=pd.to_numeric(df['Latitude'], errors='coerce').astype(float)
df['Longitude']=pd.to_numeric(df['Longitude'], errors='coerce').astype(float)
df['Altitude (m)']=pd.to_numeric(df['Altitude (m)'], errors='coerce').astype(float)
df['Proximity_to_Water (km)']=pd.to_numeric(df['Proximity_to_Water (km)'], errors='coerce').astype(float)
df['Urbanization_Index']=pd.to_numeric(df['Urbanization_Index'], errors='coerce').astype(float)
df['Vegetation_Index']=pd.to_numeric(df['Vegetation_Index'], errors='coerce').astype(float)
df['ENSO_Index']=pd.to_numeric(df['ENSO_Index'], errors='coerce').astype(float)
df['Particulate_Matter (µg/m³)']=pd.to_numeric(df['Particulate_Matter (µg/m³)'], errors='coerce').astype(float)
df['Sea_Surface_Temp (°C)']=pd.to_numeric(df['Sea_Surface_Temp (°C)'], errors='coerce').astype(float)
df['Avg_Temp (°C)'].fillna(df['Avg_Temp (°C)'].mean(),inplace=True)
df['Max_Temp (°C)'].fillna(df['Max_Temp (°C)'].mean(),inplace=True)
df['Min_Temp (°C)'].fillna(df['Min_Temp (°C)'].mean(),inplace=True)
df['Precipitation (mm)'].fillna(df['Precipitation (mm)'].mean(),inplace=True)
df['Humidity (%)'].replace('Unknown',np.nan,inplace=True)
df['Humidity (%)'].fillna(df['Humidity (%)'].mean(),inplace=True)
df['Wind_Speed (m/s)'].fillna(df['Wind_Speed (m/s)'].mean(),inplace=True)
df['Solar_Irradiance (W/m²)'].fillna(df['Solar_Irradiance (W/m²)'].mean(),inplace=True)
df['Cloud_Cover (%)'].fillna(df['Cloud_Cover (%)'].mean(),inplace=True)
df['CO2_Concentration (ppm)'].fillna(df['CO2_Concentration (ppm)'].mean(),inplace=True)
df['Latitude'].fillna(df['Latitude'].mean(),inplace=True)
df['Longitude'].fillna(df['Longitude'].mean(),inplace=True)
df['Altitude (m)'].fillna(df['Altitude (m)'].mean(),inplace=True)
df['Proximity_to_Water (km)'].fillna(df['Proximity_to_Water (km)'].mean(),inplace=True)
df['Urbanization_Index'].fillna(df['Urbanization_Index'].mean(),inplace=True)
df['Vegetation_Index'].fillna(df['Vegetation_Index'].mean(),inplace=True)
df['ENSO_Index'].fillna(df['ENSO_Index'].mean(),inplace=True)
df['Particulate_Matter (µg/m³)'].fillna(df['Particulate_Matter (µg/m³)'].mean(),inplace=True)
df['Sea_Surface_Temp (°C)'].fillna(df['Sea_Surface_Temp (°C)'].mean(),inplace=True)

def remove_outliers(df, column):
    Q1=df[column].dropna().quantile(0.25)
    Q3=df[column].dropna().quantile(0.75)
    IQR=Q3-Q1
    lbound=Q1-1.5*IQR
    ubound=Q3+1.5*IQR
    df.loc[df[column]<lbound,column]=lbound
    df.loc[df[column]>ubound,column]=ubound
    return df
for column in df.select_dtypes(include=[np.float64, np.int64]).columns:
    df=remove_outliers(df,column)

app=dash.Dash(__name__)

def generate_figures(filtered_df):
    fig_avg_temp=px.line(filtered_df, x='Year', y='Avg_Temp (°C)', title="Average Temperature Over Years")
    fig_co2=px.line(filtered_df, x='Year', y='CO2_Concentration (ppm)', title="CO2 Concentration Over Years")
    fig_sea_surface_temp=px.line(filtered_df, x='Year', y='Sea_Surface_Temp (°C)', title="Sea Surface Temperature Over Years")
    fig_co2_bar=px.bar(filtered_df, x='Year', y='CO2_Concentration (ppm)', title="CO2 Concentration Over Years")
    fig_co2_vs_sea_surface=px.scatter(filtered_df, x='CO2_Concentration (ppm)', y='Sea_Surface_Temp (°C)', color='Year', title="CO2 Concentration vs. Sea Surface Temperature")
    fig_correlation_heatmap=px.imshow(filtered_df.corr(), text_auto=True, color_continuous_scale='Blues', title="Feature Correlation Heatmap")
    fig_correlation_heatmap.update_layout(width=1000, height=800)
    
    X=filtered_df[['CO2_Concentration (ppm)', 'Solar_Irradiance (W/m²)', 'ENSO_Index', 
                     'Precipitation (mm)', 'Humidity (%)']]
    y=filtered_df['Sea_Surface_Temp (°C)']
    
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)
    regressor=LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred=regressor.predict(X_test)
    
    fig_actual_vs_predicted=px.scatter(x=y_test, y=y_pred, labels={'x': "Actual Sea Surface Temperature (°C)", 'y': "Predicted Sea Surface Temperature (°C)"}, title="Actual vs. Predicted Sea Surface Temperature")
    fig_actual_vs_predicted.add_scatter(x=y_test, y=y_test, mode='lines', line=dict(color='red'))
    
    return fig_avg_temp, fig_co2, fig_sea_surface_temp, fig_co2_bar, fig_co2_vs_sea_surface, fig_correlation_heatmap, fig_actual_vs_predicted

app.layout=html.Div([
    html.H1("Climate Change Data Dashboard"),
    dcc.Dropdown(
        id='year-dropdown',
        options=[{'label': str(year), 'value': year} for year in df['Year'].dropna().unique()],
        multi=True,
        value=[df['Year'].min(), df['Year'].max()],
        placeholder="Select Year Range"
    ),
    dcc.Graph(id='avg-temp-graph'),
    dcc.Graph(id='co2-graph'),
    dcc.Graph(id='sea-surface-temp'),
    dcc.Graph(id='co2-bar-chart'),
    dcc.Graph(id='co2-vs-sea-surface'),
    dcc.Graph(id='correlation-heatmap'),
    dcc.Graph(id='actual-vs-predicted'),
])

@app.callback(
    [Output('avg-temp-graph', 'figure'),
     Output('co2-graph', 'figure'),
     Output('sea-surface-temp', 'figure'),
     Output('co2-bar-chart', 'figure'),
     Output('co2-vs-sea-surface', 'figure'),
     Output('correlation-heatmap', 'figure'),
     Output('actual-vs-predicted', 'figure')],
    [Input('year-dropdown', 'value')]
)
def update_graphs(selected_years):
    filtered_df = df[df['Year'].isin(selected_years)]
    return generate_figures(filtered_df)

if __name__ == '__main__':
    app.run_server(debug=True)
