import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from notebooks.pre_analysis import panel_analysis_visualizations
##add comments

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

def generate_features(df, qty_col):
    """Generate lag, lead, and log-transformed features based on a specified column."""

    for lag in lags:
        df[f'{qty_col}_lag{lag}'] = df.groupby(level='unit_id')[qty_col].shift(lag)

    for lead in leads:
        df[f'{qty_col}_lead{lead}'] = df.groupby(level='unit_id')[qty_col].shift(-lead)
    return df

def prepare_panel_data(df, groupby_cols):
    """Prepare data for panel analysis by setting a multi-index."""
    df['unit_id'] = df['sales_org'].astype(str) + '_' + df['dp_cust'].astype(str) + '_' + df['dmd_item_10'].astype(str)
    df = df.set_index(['unit_id', 'week_start_date'])
    return df

def descriptive_statistics(df, output_folder):
    """Perform descriptive statistics and save results."""
    # Generate the descriptive output file path
    descriptive_output_path = os.path.join(output_folder, "descriptive_statistics.xlsx")

    with pd.ExcelWriter(descriptive_output_path) as writer:
        # Descriptive statistics for numeric columns
        numeric_stats = df.describe()
        numeric_stats.to_excel(writer, sheet_name="Numeric_Stats")

        # Count of unique values for non-numeric columns
        non_numeric_cols = df.select_dtypes(exclude=['number']).columns
        for col in non_numeric_cols:
            unique_counts = df[col].value_counts()
            unique_counts.to_excel(writer, sheet_name=f"Unique_Counts_{col}")

        # Variability analysis
        numeric_cols = df.select_dtypes(include=['number']).columns
        if not numeric_cols.empty:
            variability = df[numeric_cols].groupby(level='unit_id').var().mean()
            variability_df = pd.DataFrame(variability, columns=["Variability"])
            variability_df.to_excel(writer, sheet_name="Variability_Across_Entities")

            # Correlation matrix
            corr_matrix = df[numeric_cols].corr()
            corr_matrix.to_excel(writer, sheet_name="Correlation_Matrix")

            # Heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
            plt.title("Correlation Matrix Heatmap")
            heatmap_path = os.path.join(output_folder, "correlation_heatmap.png")
            plt.savefig(heatmap_path)
            plt.close()

    print(f"Descriptive statistics saved to {descriptive_output_path}")



def panel_structure_analysis(df, output_folder):
    """Check panel structure and balance, and save results."""
    results = []

    # Check number of observations per entity
    obs_per_entity = df.groupby(level='unit_id').size()

    # Determine if the panel is balanced
    min_obs = obs_per_entity.min()
    max_obs = obs_per_entity.max()
    is_balanced = min_obs == max_obs

    results.append("Panel Structure Analysis:")
    if is_balanced:
        results.append("The panel is balanced.")
    else:
        results.append("The panel is unbalanced.")
        results.append(f"Minimum observations per entity: {min_obs}")
        results.append(f"Maximum observations per entity: {max_obs}")

    # Add a breakdown of panel balance
    panel_balance = obs_per_entity.value_counts()
    results.append("\nBreakdown of Observations per Entity:")
    results.append(str(panel_balance))

    # Panel Dimensions
    num_entities = len(obs_per_entity)
    num_time_periods = len(df.index.get_level_values('week_start_date').unique())
    results.append(f"\nNumber of Entities: {num_entities}")
    results.append(f"Number of Time Periods: {num_time_periods}")

    # Check Key Identifiers
    duplicates = df.reset_index().duplicated(subset=['unit_id', 'week_start_date']).sum()
    if duplicates == 0:
        results.append("\nEach unit_id and week_start_date combination is unique.")
    else:
        results.append(f"\nWarning: {duplicates} duplicate entries found.")

    # Save results to a text file
    panel_analysis_path = os.path.join(output_folder, "panel_structure_analysis.txt")
    with open(panel_analysis_path, 'w') as file:
        file.write("\n".join(results))
    
    print(f"Panel structure analysis saved to {panel_analysis_path}")



def run_panel_regression(df, x_columns, y_column, lags, leads):
    """Run panel regression for lagged and lead variables."""
    results_summary = []
    
    for y_var in [f'{y_column}_lag{lag}' for lag in lags] + [f'{y_column}_lead{lead}' for lead in leads]:
        if y_var not in df.columns:
            print(f"Warning: {y_var} not found in DataFrame. Skipping...")
            continue
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



def save_results(df, output_folder, params):
    """Save results to a file and include the parameters used for the run."""
    model_output_path = os.path.join(output_folder, "model_results.xlsx")
    params_path = os.path.join(output_folder, "model_params.txt")

    # Save the results dataframe
    df.to_excel(model_output_path, index=False)

    # Save the parameters to a text file
    
    with open(params_path, 'w') as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")

# Example usage
if __name__ == "__main__":
    #input file location
    filepath = "C:\\Data\\hershey_promo_impact\\shipment_hist_weekly_with_promo_features.csv"
    
    #output file location
    base_output_path = "C:\\Users\\AdlerXie\\Documents\\GitHub\\Hershey_promo\\output"
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

    # Perform panel structure analysis
    panel_structure_analysis(aggregated_df, output_folder)

    #   Perform descriptive statistics
    descriptive_statistics(aggregated_df, output_folder)  # Pass the output path

     # Specify whether to use qty or log_qty for feature generation
    # generate log_qty, log_forecast_cot_spend, log_total_expense_trade
    aggregated_df['log_qty'] = np.log1p(aggregated_df['qty']+1)
   

    qty_column = 'qty'  # Change to 'log_qty' or another column as needed

    # Generate visualizations to assess fixed effects suitability
    panel_analysis_visualizations(aggregated_df, 'qty', output_folder)
    print("Visualizations for panel analysis have been generated.")

    # define how many lead and lags to use
    lags = [ 0]
    leads = [1]
    aggregated_df = generate_features(aggregated_df, qty_col=qty_column)
    print('finished generating features')

    # x variables
    aggregated_df['log_forecast_cot_spend'] = np.log1p(aggregated_df['forecast_cot_spend']+1)
    aggregated_df['log_total_expense_trade'] = np.log1p(aggregated_df['total_expense_trade']+1)

    x_columns = [
        'total_expense_trade'
    ]

   
    y_column = qty_column  # Align with the chosen qty_column
    results_df = run_panel_regression(aggregated_df, x_columns, y_column, lags, leads)

    # Define parameters for this run
    params = {
    'filepath': filepath,
    'qty_column': qty_column,
    'x_columns': x_columns,
    'y_column': y_column,
    'lags': lags,
    'leads': leads,
    'output_folder': output_folder
    }

    # Save results to the output folder
    save_results(results_df, output_folder, params)

