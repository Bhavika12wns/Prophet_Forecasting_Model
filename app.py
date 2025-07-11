#!/usr/bin/env python
# coding: utf-8

# In[17]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet_forecasting_model import load_and_preprocess, remove_spikes, prophet_forecast_model
import io
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows


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

    forecast_months= st.slider("Select number of Future Months to Forecast", 1,36,12)

    if st.button("Run Forecast"):
        final_df, r2, mape, accuracy = prophet_forecast_model(cleaned_df, forecast_months)
        
        st.subheader("Forecasted Results")
        st.dataframe(final_df)

        st.subheader("Model Accuracy Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("R2Score", f"{r2:.4f}")
        col2.metric("MAPE (%)", f"{mape*100:.2f}")
        col3.metric("Accuracy (%)", f"{accuracy:.2f}")

        st.subheader("FOrecasted Sales Plot")
        fig, ax = plt.subplots(figsize=(14,6))
        for label, grp in final_df.groupby('Type'):
            if label != 'Forecast':
                ax.plot(grp['ds'], grp['Actual_Sales'], label=f'{label} Actual')
            ax.plot(grp['ds'], grp['Predicted_Sales'], linestyle='--', label=f'{label} Predicted')
        ax.set_title("Forecasted Sales using Prophet")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        output=io.BytesIO()
        wb=Workbook()
        ws=wb.active
        ws.title="Forecast Results"
        for r in dataframe_to_rows(final_df, index=False, header=True):
            ws.append(r)
        wb.save(output)
        output.seek(0)

        st.download_button(
            label="Download Excel File of the Forecasted Sales",
            data=output,
            file_name="Forecast_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
        


# In[ ]:




