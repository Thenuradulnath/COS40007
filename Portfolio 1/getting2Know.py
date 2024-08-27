import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the output directory for saving plots and reports
output_dir = r'Portfolio 1\Outs'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Read data
data = pd.read_csv(r'Portfolio 1\dataSet\converted_data.csv')

# Check DataFrame info
print(data.info())

# Remove duplicates
data = data.drop_duplicates()

# Check for missing values
print("Missing values in each column:")
print(data.isnull().sum())

# Remove missing values if necessary (optional)
# data = data.dropna()

# Plotting initial boxplots to visualize outliers
plt.figure(figsize=(10, 8))
sns.boxplot(data=data.select_dtypes(include='number'))
plt.title('Boxplot of Numerical Features Before Outlier Removal')
plt.savefig(os.path.join(output_dir, 'boxplot_before_outlier_removal.png'))
plt.show()

# Remove outliers based on IQR method
def remove_outliers(df, column):
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1  # Interquartile range
    # Define bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Remove outliers
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Remove outliers for each numerical column
for col in data.select_dtypes(include='number').columns:
    data = remove_outliers(data, col)

# Check for duplicates again
print(f"Number of duplicate rows after cleaning: {data.duplicated().sum()}")

# Plotting boxplots after outlier removal
plt.figure(figsize=(10, 8))
sns.boxplot(data=data.select_dtypes(include='number'))
plt.title('Boxplot of Numerical Features After Outlier Removal')
plt.savefig(os.path.join(output_dir, 'boxplot_after_outlier_removal.png'))
plt.show()

# Save cleaned data
cleaned_data_path = os.path.join(output_dir, 'cleaned_data.csv')
data.to_csv(cleaned_data_path, index=False)
print(f"Data cleaning completed and saved to '{cleaned_data_path}'.")

# EDA: Identify target and predictors
# For example, assume 'target_column' is your target variable
target_column = 'target_variable_name'  # Replace with your actual target column name
predictors = data.columns.drop(target_column)

# Univariate Analysis
for col in predictors:
    plt.figure(figsize=(8, 4))
    sns.histplot(data[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.savefig(os.path.join(output_dir, f'distribution_{col}.png'))
    plt.show()

# Summary Statistics
summary_stats = data.describe()
print(summary_stats)

# Save summary statistics to a text file
summary_stats_path = os.path.join(output_dir, 'summary_statistics.txt')
with open(summary_stats_path, 'w') as f:
    f.write('Summary Statistics:\n')
    f.write(summary_stats.to_string())
print(f"Summary statistics saved to '{summary_stats_path}'.")

# Multivariate Analysis
sns.pairplot(data[predictors])
pairplot_path = os.path.join(output_dir, 'pairwise_relationships.png')
plt.savefig(pairplot_path)
plt.show()
print(f"Pairplot saved to '{pairplot_path}'.")

# Correlation Heatmap
plt.figure(figsize=(12, 8))
corr = data.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
heatmap_path = os.path.join(output_dir, 'correlation_heatmap.png')
plt.savefig(heatmap_path)
plt.show()
print(f"Correlation heatmap saved to '{heatmap_path}'.")

# Create a summary report (pseudo-code)
# Uncomment and modify as needed
# report_path = os.path.join(output_dir, 'eda_summary_report.txt')
# with open(report_path, 'w') as report:
#     report.write('Exploratory Data Analysis Summary\n')
#     report.write('=================================\n')
#     report.write(f'Target Variable: {target_column}\n')
#     report.write(f'Predictors: {", ".join(predictors)}\n')
#     report.write('Summary Statistics:\n')
#     report.write(summary_stats.to_string())
#     # Add more details as needed

print("EDA completed. Check visualizations and summary statistics for insights.")
