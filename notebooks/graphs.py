import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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
