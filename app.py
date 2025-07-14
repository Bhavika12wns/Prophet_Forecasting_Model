#!/usr/bin/env python
# coding: utf-8

# In[17]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from prophet_forecasting_model import load_and_preprocess, remove_spikes, prophet_forecast_model
import io
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.drawing.image import Image as XLImage


# In[18]:


st.set_page_config(page_title="Sales Forecast", layout="wide")
st.title("Sales Forecast App")
st.markdown("Upload your Excel File (.'xlsx') with **'month'** and **'sales'** columns.")


# In[19]:


uploaded_file = st.file_uploader("Upload your Excel File", type=["xlsx"])


# In[21]:


if uploaded_file:
    st.success("File uploaded successfully")
    
    df=load_and_preprocess(uploaded_file)
    st.subheader("Raw Data")
    st.dataframe(df)

    st.subheader("Cleaned Data after Spikes Removal")
    cleaned_df = remove_spikes(df)
    st.dataframe(cleaned_df)

    forecast_months= st.number_input("Enter number of Future Months to Forecast", min_value=1, max_value=36, value=12, step=1)

    if st.button("Run Forecast"):
        final_df, r2, mape, accuracy = prophet_forecast_model(cleaned_df, forecast_months)

        final_df['ds_label']=final_df['ds'].dt.strftime('%b %Y')
        final_df=final_df[['ds', 'Actual_Sales', 'Predicted_Sales', 'Type']]
        
        st.subheader("Forecasted Results")
        st.dataframe(final_df)

        st.subheader("Model Accuracy Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("R2Score", f"{r2:.4f}")
        col2.metric("MAPE (%)", f"{mape*100:.2f}")
        col3.metric("Accuracy (%)", f"{accuracy:.2f}")

        st.subheader("Interactive Sales Forecast Plot")
        final_df['Type'] = final_df['Type'].replace({
            'Actual':'Prediction on Actual Sales',
            'Forecast':'Forecasted Sales'})
        fig_plotly=px.line(
            final_df,
            x='ds',
            y='Predicted_Sales',
            color='Type',
            labels={'ds':'Date', 'Predicted_Sales':'Sales'},
            title="Forecasted vs Actual Sales"
        )
        fig_plotly.add_scatter(
            x=final_df['ds'],
            y=final_df['Actual_Sales'],
            mode='markers',
            name='Actual Sales',
            marker=dict(size=6, color='black')
        )
        y_max=final_df[['Actual_Sales', 'Predicted_Sales']].max().max()
        fig_plotly.update_layout(
            xaxis=dict(
                tickvals=final_df['ds'],
                ticktext=final_df['ds'].dt.strftime('%Y-%m'),
                tickangle=-90), 
            yaxis_range=[0, y_max*1.1]
        )
        st.plotly_chart(fig_plotly, use_container_width=True)
        
        fig, ax = plt.subplots(figsize=(14,6))
        final_df_sorted = final_df.sort_values("ds")
        for label, grp in final_df_sorted.groupby('Type'):
            ax.plot(grp['ds'], grp['Predicted_Sales'], linestyle='--', label=f'{label}')
            if label != 'Forecast':
                ax.plot(grp['ds'], grp['Actual_Sales'], marker ='o', label=f'{label}')

        # final_df_sorted = final_df.sort_values("ds")
        # actual= final_df_sorted[final_df_sorted['Type']=='Actual']
        # forecast= final_df_sorted[final_df_sorted['Type']=='Forecast']
        # ax.plot(actual['ds'], actual['Actual_Sales'], linestyle='--', color='red', label='Actual Sales')
        # ax.plot(actual['ds'], actual['Forecast_Sales'], linestyle='--', color='blue', label='Prediction on Actual Sales')
        # ax.plot(forecast['ds'], forecast['Forecast_Sales'], linestyle='--', color='green', label='Forecasted Sales')
        
        ax.set_title("Forecasted Sales using Prophet")
        ax.set_xlabel("Month")
        ax.set_ylabel("Sales")
        ax.set_ylim(0, y_max*1.1)
        ax.set_xticks(ticks=final_df_sorted['ds'], labels=final_df_sorted['ds'].dt.strftime('%Y-%m'), rotation=90)
        ax.legend()
        ax.grid(True)
        plt.tight_layout()

        img_data=io.BytesIO()
        plt.savefig(img_data, format='png')
        plt.close()
        img_data.seek(0)
        
        output=io.BytesIO()
        wb=Workbook()
        ws=wb.active
        ws.title="Forecast Results"
        for r in dataframe_to_rows(final_df[['ds', 'Actual_Sales', 'Predicted_Sales', 'Type']], index=False, header=True):
            ws.append(r)
        
        img=XLImage(img_data)
        ws.add_image(img, "G2")
        
        wb.save(output)
        output.seek(0)

        st.download_button(
            label="Download Excel File of the Forecasted Sales",
            data=output,
            file_name="Forecast_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
        
