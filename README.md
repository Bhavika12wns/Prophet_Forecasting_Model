Sales Forecast Webpage
A streamlit webpage to upload monthly sales data, clean outliers, forecast future sales using Prophet, and download Excel reports with graphs.

## Features
Upload '.xlsx' files with 'month' and 'sales'
Cleans spikes and outliers
Forecasts future sales using Prophet
Shows MAPE, R2 Score, Accuracy
Interactive Plotly graphs
Excel export with chart embedded

## To Run Locally
''''bash
pip install -r requirements.txt
streamlit run app.py
