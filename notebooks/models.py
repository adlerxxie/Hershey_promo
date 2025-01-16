import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
import os
import datetime

def load_and_preprocess_data(filepath):
    """Load data from a CSV file and perform initial preprocessing."""
    df = pd.read_csv(filepath)
    df['week_start_date'] = pd.to_datetime(df['week_start_date'])
    return df

def aggregate_data(df, groupby_cols, sum_columns, first_non_missing_columns):
    """Aggregate data by specified columns with sum and first non-missing value rules."""
    agg_dict = {col: 'sum' for col in sum_columns}
    agg_dict.update({col: 'first' for col in first_non_missing_columns})
    aggregated_df = df.groupby(groupby_cols).agg(agg_dict).reset_index()

    # Replace promotion_type with NA if on_promotion is False
    aggregated_df.loc[aggregated_df['on_promotion'] == False, 'promotion_type'] = np.nan

    # Create dummy variables for promotion_type
    aggregated_df = pd.get_dummies(aggregated_df, columns=['promotion_type'], dummy_na=True)
    return aggregated_df

def generate_features(df, qty_col='qty'):
    """Generate lag, lead, and log-transformed features."""
    lags = [8, 4, 2, 1, 0]
    leads = [1, 2, 4, 8]

    for lag in lags:
        df[f'lag{lag}'] = df.groupby(level='unit_id')[qty_col].shift(lag)

    for lead in leads:
        df[f'lead{lead}'] = df.groupby(level='unit_id')[qty_col].shift(-lead)

    # Create log-transformed variables
    df['log_qty'] = np.log(df[qty_col] + 1)
    for lag in lags:
        df[f'log_lag{lag}'] = df.groupby(level='unit_id')['log_qty'].shift(lag)
    for lead in leads:
        df[f'log_lead{lead}'] = df.groupby(level='unit_id')['log_qty'].shift(-lead)
    return df

def prepare_panel_data(df, groupby_cols):
    """Prepare data for panel analysis by setting a multi-index."""
    df['unit_id'] = df['sales_org'].astype(str) + '_' + df['dp_cust'].astype(str) + '_' + df['dmd_item_10'].astype(str)
    df = df.set_index(['unit_id', 'week_start_date'])
    return df

def run_panel_regression(df, x_columns, y_column, lags, leads):
    """Run panel regression for lagged and lead variables."""
    results_summary = []
    
    for y_var in [f'{y_column}_lag{lag}' for lag in lags] + [f'{y_column}_lead{lead}' for lead in leads]:
        y = df[y_var].dropna()
        X_subset = df[x_columns].loc[y.index]  # Align X with the available y data

        # Fit the model
        model = PanelOLS(y, X_subset, entity_effects=True, time_effects=True)
        results = model.fit(cov_type='clustered', cluster_entity=True)

        # Store results
        summary_dict = {
            'Dependent Variable': y_var,
            'R-squared': results.rsquared,
            'Number of Observations': results.nobs
        }
        for var in x_columns:
            summary_dict[f'{var}_coef'] = results.params.get(var, None)
            summary_dict[f'{var}_pval'] = results.pvalues.get(var, None)
        results_summary.append(summary_dict)

    return pd.DataFrame(results_summary)

def save_results(df, output_path):
    """Save results to a file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_excel(output_path, index=False)

# Example usage
if __name__ == "__main__":
    filepath = "C:\\Data\\hershey_promo_impact\\shipment_hist_weekly_with_promo_features.csv"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"C:\\Users\\AdlerXie\\Documents\\GitHub\\Hershey_promo\\output\\log_panel_model_results_{timestamp}.xlsx"

    groupby_cols = ['sales_org', 'dp_cust', 'dmd_item_10', 'week_start_date']
    sum_columns = ['qty', 'po_qty_week_1', 'po_qty_week_2', 'po_qty_week_3', 'po_qty_week_4', 'po_qty_week_8']
    first_non_missing_columns = [
        'descr', 'division', 'franchise', 'subbrand', 'brand', 'packtype',
        'ppg_family', 'ppg', 'season', 'sales_status', 'on_promotion',
        'promotion_type', 'total_expense_trade', 'forecast_cot_spend', 'price_sch3'
    ]

    df = load_and_preprocess_data(filepath)
    aggregated_df = aggregate_data(df, groupby_cols, sum_columns, first_non_missing_columns)
    aggregated_df = prepare_panel_data(aggregated_df, groupby_cols)
    aggregated_df = generate_features(aggregated_df)

    x_columns = [
        'promotion_type_Corporate Promotions',
        'promotion_type_Correction',
        'promotion_type_EDLC',
        'promotion_type_EDLP',
        'promotion_type_Hi Lo',
        'promotion_type_Miscellaneous'
    ]
    
    y_column = 'log_qty'  # Can be 'log_qty', 'qty', or any other column
    results_df = run_panel_regression(aggregated_df, x_columns, y_column, lags=[8, 4, 2, 1, 0], leads=[1, 2, 4, 8])
    save_results(results_df, output_path)
