import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pre_analysis import panel_analysis_visualizations, descriptive_statistics, panel_structure_analysis

def create_output_folder(base_path):
    """Create a timestamped output folder."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join(base_path, f"output_{timestamp}")
    os.makedirs(output_folder, exist_ok=True)
    return output_folder


def load_and_preprocess_data(filepath):
    """Load data from a CSV file and perform initial preprocessing."""
    df = pd.read_csv(filepath)
    df['week_start_date'] = pd.to_datetime(df['week_start_date'])
    # replace all missing value with 0 if data type is numeric
    df.fillna(0, inplace=True)
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

def generate_Y_variables(df, qty_col, lags, leads):
    """Generate lag, lead, and log-transformed features based on a specified column."""
    # generate Y variables
    for lag in lags:
        df[f'{qty_col}_lag{lag}'] = df.groupby(level='unit_id')[qty_col].shift(lag)

    for lead in leads:
        df[f'{qty_col}_lead{lead}'] = df.groupby(level='unit_id')[qty_col].shift(-lead)
    return df

def generate_X_variables(df, x_columns, lags, leads):
    """Generate lag, lead, and log-transformed features based on a specified column.store in a list"""
    x_columns_lag = []
    # generate X variables
    for lag in lags:
        for x_col in x_columns:
            df[f'{x_col}_lag{lag}'] = df.groupby(level='unit_id')[x_col].shift(lag)
            x_columns_lag.append(f'{x_col}_lag{lag}')

    x_columns_lead = []
    for lead in leads:
        for x_col in x_columns:
            df[f'{x_col}_lead{lead}'] = df.groupby(level='unit_id')[x_col].shift(-lead)
            x_columns_lead.append(f'{x_col}_lead{lead}')

    return df, x_columns_lag, x_columns_lead

def prepare_panel_data(df, groupby_cols):
    """Prepare data for panel analysis by setting a multi-index."""
    df['unit_id'] = df['sales_org'].astype(str) + '_' + df['dp_cust'].astype(str) + '_' + df['dmd_item_10'].astype(str)
    df = df.set_index(['unit_id', 'week_start_date'])
    return df

def panel_reg(df, x_columns, y_column, lead_lags):
    """Run panel regression for lagged and lead variables.
    Args:
        df: panel dataframe
        x_columns: list of x variables
        y_column: dependent variable
        lead_lags: list of strings like ['lag1', 'lead1']
    """
    results_summary = []
    
    # Loop through each lag/lead combination
    for time_shift in lead_lags:
        # Filter X columns based on the current lag/lead
        x_columns_filtered = []
        column_mapping = {}  # Dictionary to store original:new column names that is standard for all the lead lag models, for easy results comparison
        
        for col in x_columns:
            # Include if column starts with 'promotion' and ends with current lag/lead
            if col.startswith('promotion_type_'):
                if col.endswith(time_shift):
                    new_col = col.replace(time_shift, 'TimeShift')
                    column_mapping[col] = new_col
                    x_columns_filtered.append(col)
            # Include all non-promotion columns that contains the word lag
            elif 'lag' in col:
                x_columns_filtered.append(col)
        
        # Prepare data for regression
        y = df[y_column]
        X_subset = df[x_columns_filtered].copy()  # Get data with original column names
        X_subset = X_subset.rename(columns=column_mapping)  # Rename columns after selection
        
        # Fit the model
        model = PanelOLS(y, X_subset, entity_effects=True, time_effects=True)
        results = model.fit(cov_type='clustered', cluster_entity=True)

        # Store results
        summary_dict = {
            'Dependent Variable': y_column,
            'Time_Shift': time_shift,
            'R-squared': results.rsquared,
            'Number of Observations': results.nobs
        }
        for var in X_subset.columns:
            summary_dict[f'{var}_coef'] = results.params.get(var, None)
            summary_dict[f'{var}_pval'] = results.pvalues.get(var, None)
        results_summary.append(summary_dict)

    return pd.DataFrame(results_summary)



def save_results(df, output_folder, params):
    """Save results to a file and include the parameters used for the run."""
    # Use model name in file paths if available
    model_name = params.get('model_name', 'model')  # Default to 'model' if not specified
    model_output_path = os.path.join(output_folder, f"{model_name}_results.xlsx")
    params_path = os.path.join(output_folder, f"{model_name}_params.txt")

    # Save the results dataframe
    df.to_excel(model_output_path, index=False)

    # Save the parameters to a text file
    with open(params_path, 'w') as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")

# Example usage
if __name__ == "__main__":
    # Update the input file location to use Linux-style path
    filepath = "/mnt/c/Data/hershey_promo_impact/shipment_hist_weekly_with_promo_features.csv"
    
    # Update the output file location to use Linux-style path
    base_output_path = "/mnt/c/Users/AdlerXie/Documents/GitHub/Hershey_promo/output"
    output_folder = create_output_folder(base_output_path)
    print(f"All outputs will be saved to {output_folder}")

    groupby_cols = ['sales_org', 'dp_cust', 'dmd_item_10', 'week_start_date']
    sum_columns = ['qty', 'total_expense_trade', 'forecast_cot_spend',  'po_qty_week_1', 'po_qty_week_2', 'po_qty_week_3', 'po_qty_week_4', 'po_qty_week_8']
    first_non_missing_columns = [
        'descr', 'division', 'franchise', 'subbrand', 'brand', 'packtype',
        'ppg_family', 'ppg', 'season', 'sales_status', 'on_promotion',
        'promotion_type', 'total_expense_trade', 'forecast_cot_spend', 'price_sch3'
    ]

    df = load_and_preprocess_data(filepath)
    print('finished loading data')

    aggregated_df = aggregate_data(df, groupby_cols, sum_columns, first_non_missing_columns)
    print('finished aggregating data')

    aggregated_df = prepare_panel_data(aggregated_df, groupby_cols)
    print('finished preparing panel data')

    '''# Perform panel structure analysis
    panel_structure_analysis(aggregated_df, output_folder)

    #   Perform descriptive statistics
    descriptive_statistics(aggregated_df, output_folder)  # Pass the output path

    # Generate visualizations to assess fixed effects suitability
    panel_analysis_visualizations(aggregated_df, 'qty', output_folder)
    print("Visualizations for panel analysis have been generated.")
'''
    # First, generate log transformations
    aggregated_df['log_qty'] = np.log1p(aggregated_df['qty']+1)
    aggregated_df['log_forecast_cot_spend'] = np.log1p(aggregated_df['forecast_cot_spend']+1)
    aggregated_df['log_total_expense_trade'] = np.log1p(aggregated_df['total_expense_trade']+1)

    # Then define x_base_columns with the newly created columns
    x_base_columns = [
        'promotion_type_Corporate Promotions',
        'promotion_type_Correction',
        'promotion_type_EDLC',
        'promotion_type_EDLP',
        'promotion_type_Hi Lo',
        'promotion_type_Miscellaneous',
        # 'total_expense_trade',
        #'forecast_cot_spend',
        # 'log_forecast_cot_spend',
        # 'log_total_expense_trade',
        'qty',
        'log_qty'
    ]
    
    # Now generate X variables
    aggregated_df, x_columns_lag, x_columns_lead = generate_X_variables(aggregated_df, x_base_columns, lags=[1,2,4,8,12,26,52], leads=[0,1,2,4,8,12])

    # add promotion type dummies to x_columns_lag
    promotion_dummies = [
        'promotion_type_Corporate Promotions',
        'promotion_type_Correction',
        'promotion_type_EDLC',
        'promotion_type_EDLP',
        'promotion_type_Hi Lo',
        'promotion_type_Miscellaneous'
    ]

    x_columns = promotion_dummies + x_columns_lag + x_columns_lead
    print('finished generating X variables')

   # Specify whether to use qty or log_qty for feature generation
    # generate log_qty, log_forecast_cot_spend, log_total_expense_trade
    qty_column = 'log_qty'  # Change to 'log_qty' or another column as needed for params doc
    y_column = qty_column  # Align with the chosen qty_column

    model = "panel_reg" # for params doc
    results_df = panel_reg(aggregated_df, x_columns, y_column, ['lag12', 'lag8', 'lag4', 'lag2', 'lag1', 'lead0', 'lead1', 'lead2', 'lead4', 'lead8', 'lead12'])

    # Define parameters for this run
    params = {
        'model_name': model,  # Add model name to params
        'filepath': filepath,
        'qty_column': qty_column,
        'x_columns': x_columns,
        'y_column': y_column,
        'output_folder': output_folder
    }

    # Save results to the output folder
    save_results(results_df, output_folder, params)

