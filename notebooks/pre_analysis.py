import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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



def panel_analysis_visualizations(df, dependent_var, output_folder):
    """Generate visualizations to assess two-way fixed effects suitability and save results."""

    # Create a subfolder for visualizations
    visuals_folder = os.path.join(output_folder, "visuals")
    os.makedirs(visuals_folder, exist_ok=True)

    # 1. Variation in the dependent variable across entities
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=df.index.get_level_values('unit_id'), y=df[dependent_var])
    plt.xticks([], [])  # Hide x-axis labels if there are too many entities
    plt.title('Variation in Dependent Variable Across Entities')
    plt.ylabel(dependent_var)
    plt.xlabel('Entities')
    entity_variation_path = os.path.join(visuals_folder, 'entity_variation.png')
    plt.savefig(entity_variation_path)
    plt.close()

    # 2. Variation in the dependent variable across time periods
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=df.index.get_level_values('week_start_date'), y=df[dependent_var])
    plt.xticks(rotation=45)
    plt.title('Variation in Dependent Variable Over Time')
    plt.ylabel(dependent_var)
    plt.xlabel('Time Periods')
    time_variation_path = os.path.join(visuals_folder, 'time_variation.png')
    plt.savefig(time_variation_path)
    plt.close()

    # 3. Within-group deviation histogram
    df['within_deviation'] = df[dependent_var] - df.groupby(level='unit_id')[dependent_var].transform('mean')
    plt.figure(figsize=(10, 6))
    sns.histplot(df['within_deviation'], kde=True, bins=30)
    plt.title('Within-Entity Deviation of Dependent Variable')
    plt.xlabel('Deviation')
    plt.ylabel('Frequency')
    within_deviation_path = os.path.join(visuals_folder, 'within_deviation.png')
    plt.savefig(within_deviation_path)
    plt.close()

    # 4. Average dependent variable over time
    time_avg = df.groupby(level='week_start_date')[dependent_var].mean()
    plt.figure(figsize=(12, 6))
    time_avg.plot()
    plt.title('Average Dependent Variable Over Time')
    plt.ylabel(dependent_var)
    plt.xlabel('Time Periods')
    time_avg_path = os.path.join(visuals_folder, 'time_avg.png')
    plt.savefig(time_avg_path)
    plt.close()

    # 5. Correlation heatmap
    # Filter numeric columns for correlation matrix
    numeric_cols = df.select_dtypes(include=['number'])

    if not numeric_cols.empty:
        corr_matrix = numeric_cols.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix of Numeric Variables')
        corr_heatmap_path = os.path.join(visuals_folder, 'correlation_heatmap.png')
        plt.savefig(corr_heatmap_path)
        plt.close()
    else:
        print("No numeric columns available for correlation analysis.")


    # 6. Entity-specific trends
    # Entity-specific trends
    plt.figure(figsize=(12, 6))
    entities_to_plot = df.index.get_level_values('unit_id').unique()[:5]  # Select 5 entities

    for entity in entities_to_plot:
        # Filter data for the specific entity
        entity_data = df[df.index.get_level_values('unit_id') == entity]
        
        # Plot data
        plt.plot(
            entity_data.index.get_level_values('week_start_date'), 
            entity_data[dependent_var], 
            label=str(entity)
        )

    plt.title('Entity-Specific Trends Over Time')
    plt.xlabel('Time Period')
    plt.ylabel(dependent_var)
    plt.legend(title='Entities', loc='best')
    entity_trends_path = os.path.join(visuals_folder, 'entity_trends.png')
    plt.savefig(entity_trends_path)
    plt.close()



    # 7. Intraclass Correlation Coefficient (ICC)
    entity_means = df.groupby(level='unit_id')[dependent_var].mean()
    overall_mean = df[dependent_var].mean()
    between_variance = ((entity_means - overall_mean) ** 2).mean()
    within_variance = df.groupby(level='unit_id')['within_deviation'].var().mean()
    icc = between_variance / (between_variance + within_variance)

    # Bar plot of ICC
    plt.figure(figsize=(6, 4))
    plt.bar(['Between Entities', 'Within Entities'], [icc, 1 - icc], color=['blue', 'orange'])
    plt.title('Intraclass Correlation Coefficient')
    plt.ylabel('Proportion of Variance')
    icc_path = os.path.join(visuals_folder, 'icc.png')
    plt.savefig(icc_path)
    plt.close()

    print(f"All visualizations have been saved to: {visuals_folder}")

# Example Usage
# panel_analysis_visualizations(aggregated_df, 'qty', output_folder)
