#!/usr/bin/env python
# coding: utf-8

# ### Installing the Libraries

# In[1]:


# !pip install fbprophet
# !pip install pandas
# !pip install scikit-learn
# !pip install matplotlib
# !pip install openpyxl


# In[2]:


import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.model_selection import ParameterGrid
import optuna

# ### Converting the Data into required form

# In[3]:


# Load and preprocess data
def load_and_preprocess(file_path):
    df = pd.read_excel(file_path)
    df['month'] = pd.to_datetime(df['month'], format='%b%y')
    df = df[['month', 'sales']]
    df = df.sort_values('month')
    df['sales'] = df['sales'].apply(int)
    df = df.groupby('month').agg({'sales': 'sum'}).reset_index()
    df.rename(columns={'month': 'ds', 'sales': 'y'}, inplace=True)
    df['y'] = df['y'].astype(float)
    return df


# ### Handle if there is any sudden Rise or Fall in the Data

# In[4]:


def remove_spikes(df, threshold=5):
    df_cleaned = df.copy()
    spikes_replaced = 0
    max_iterations = 100
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        mean_full = df_cleaned['y'].mean()
        max_val = df_cleaned['y'].max()
        min_val = df_cleaned['y'].min()
        mean_wo_max = df_cleaned[df_cleaned['y'] != max_val]['y'].mean()
        mean_wo_min = df_cleaned[df_cleaned['y'] != min_val]['y'].mean()
        diff_max = abs(mean_full - mean_wo_max) / mean_full * 100
        diff_min = abs(mean_full - mean_wo_min) / mean_full * 100

        if diff_max <= threshold and diff_min <= threshold:
            break

        if diff_max > diff_min:
            idx_to_replace = df_cleaned[df_cleaned['y'] == max_val].index[0]
            df_cleaned.at[idx_to_replace, 'y'] = mean_wo_max
        else:
            idx_to_replace = df_cleaned[df_cleaned['y'] == min_val].index[0]
            df_cleaned.at[idx_to_replace, 'y'] = mean_wo_min

        spikes_replaced += 1
    return df_cleaned
    
# ### Fit the model

# In[5]:


def grid_search_optimization(df_cleaned):
    param_grid = {
        'changepoint_prior_scale': [0.01, 0.05, 0.1, 0.5],
        'seasonality_mode': ['additive', 'multiplicative'],
        'yearly_seasonality': [True, False]
    }

    best_score = float('inf')
    best_params = None

    for params in ParameterGrid(param_grid):
        model = Prophet(
            yearly_seasonality=params['yearly_seasonality'],
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode=params['seasonality_mode'],
            changepoint_prior_scale=params['changepoint_prior_scale'],
            changepoint_range=0.8,
            growth='logistic'
        )
        model.add_seasonality(name='quarterly', period=91.25, fourier_order=8)
        model.fit(df_cleaned)

        future = model.make_future_dataframe(periods=12, freq='MS')
        future['cap'] = df_cleaned['cap'].iloc[-1]
        future['floor'] = df_cleaned['floor'].iloc[-1]

        forecast = model.predict(future)
        forecast_df = forecast[['ds', 'yhat']].copy()
        forecast_df.columns = ['ds', 'Predicted_Sales']
        merged = pd.merge(df_cleaned, forecast_df, on='ds', how='left')
        test_eval = merged.dropna()
        mape_test = mean_absolute_percentage_error(test_eval['y'], test_eval['Predicted_Sales'])

        if mape_test < best_score:
            best_score = mape_test
            best_params = params

    return best_params

def optuna_optimization(df_cleaned):
    def objective(trial):
        changepoint_prior_scale = trial.suggest_loguniform('changepoint_prior_scale', 0.01, 0.05, 0.1, 0.5)
        seasonality_mode = trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative'])
        yearly_seasonality = trial.suggest_categorical('yearly_seasonality', [True, False])

        model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=changepoint_prior_scale,
            changepoint_range=0.8,
            growth='logistic'
        )
        model.add_seasonality(name='quarterly', period=91.25, fourier_order=8)
        model.fit(df_cleaned)

        future = model.make_future_dataframe(periods=12, freq='MS')
        future['cap'] = df_cleaned['cap'].iloc[-1]
        future['floor'] = df_cleaned['floor'].iloc[-1]
        forecast = model.predict(future)
        forecast_df = forecast[['ds', 'yhat']].copy()
        forecast_df.columns = ['ds', 'Predicted_Sales']
        merged = pd.merge(df_cleaned, forecast_df, on='ds', how='left')
        test_eval = merged.dropna()
        mape_test = mean_absolute_percentage_error(test_eval['y'], test_eval['Predicted_Sales'])

        return mape_test

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    return study.best_params

def choose_optimizer(df_cleaned):
    # Example decision logic based on dataset size
    if len(df_cleaned) < 50:
        return 'grid_search'
    else:
        return 'optuna'

def prophet_forecast_model(df_cleaned, forecast_months):
    df_cleaned['cap'] = df_cleaned['y'].quantile(0.95)
    df_cleaned['floor'] = df_cleaned['y'].quantile(0.05)

    if len(df_cleaned) < 36:
        model = Prophet(yearly_seasonality = True,
                        weekly_seasonality = False,
                       daily_seasonality = False,
                       seasonality_mode='additive',
                       growth='logistic')
        model.fit(df_cleaned)

    else:
        optimizer = choose_optimizer(df_cleaned)

        if optimizer == 'grid_search':
            best_params = grid_search_optimization(df_cleaned)
        else:
            best_params = optuna_optimization(df_cleaned)

        model = Prophet(
            yearly_seasonality=best_params['yearly_seasonality'],
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode=best_params['seasonality_mode'],
            changepoint_prior_scale=best_params['changepoint_prior_scale'],
            changepoint_range=0.9,
            growth='logistic'
        )
        model.add_seasonality(name='quarterly', period=91.25, fourier_order=8)
        model.fit(df_cleaned)

    future = model.make_future_dataframe(periods=forecast_months, freq='MS')
    future['cap'] = df_cleaned['cap'].iloc[-1]
    future['floor'] = df_cleaned['floor'].iloc[-1]
    forecast = model.predict(future)

    forecast_df = forecast[['ds', 'yhat']].copy()
    forecast_df.columns = ['ds', 'Predicted_Sales']

    merged = pd.merge(df_cleaned, forecast_df, on='ds', how='left')
    merged['Type'] = 'Actual'

    future_forecast = forecast_df[forecast_df['ds'] > df_cleaned['ds'].max()].copy()
    future_forecast['Actual_Sales'] = None
    future_forecast['Type'] = 'Forecast'
    future_forecast = future_forecast[['ds', 'Actual_Sales', 'Predicted_Sales', 'Type']]

    merged.rename(columns={'y': 'Actual_Sales'}, inplace=True)
    merged = merged[['ds', 'Actual_Sales', 'Predicted_Sales', 'Type']]

    final_df = pd.concat([merged, future_forecast], ignore_index=True)

    final_df['MAPE'] = final_df.apply(
        lambda row: 100 * abs(row['Actual_Sales'] - row['Predicted_Sales']) / row['Actual_Sales']
        if pd.notnull(row['Actual_Sales']) and pd.notnull(row['Predicted_Sales']) else None, axis=1)
    test_eval = merged.dropna()
    mape_test = mean_absolute_percentage_error(test_eval['Actual_Sales'], test_eval['Predicted_Sales'])
    r2 = r2_score(test_eval['Actual_Sales'], test_eval['Predicted_Sales'])
    accuracy = 100 - (mape_test * 100)

    return final_df, r2, mape_test, accuracy
