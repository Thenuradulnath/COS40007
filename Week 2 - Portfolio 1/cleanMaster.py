import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read data
data = pd.read_csv(r'Portfolio 1\dataSet\converted_data.csv')

# Remove duplicates
data = data.drop_duplicates()

# Check for missing values
print("Missing values in each column:")
print(data.isnull().sum())

# Filter using IQR
def replace_outliers_with_whiskers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    whisker_upper = Q3 + 1.5 * IQR
    whisker_lower = Q1 - 1.5 * IQR
    
    upper_outliers = df[df[column] > whisker_upper].index
    df.loc[upper_outliers, column] = whisker_upper
    
    lower_outliers = df[df[column] < whisker_lower].index
    df.loc[lower_outliers, column] = whisker_lower
    
    return df

# Columns to check for outliers
columns = [ 'AP', 'RH']

for column in columns:
    data = replace_outliers_with_whiskers(data, column)

# plot
plt.title("Boxplot of Numerical Features After Outlier Removal")
plt.figure(figsize=(10, 8))
sns.boxplot(data=data)
plt.show()

# Save 
data.to_csv('cleaned_data.csv', index=False)
print("Data cleaning completed and saved to 'cleaned_data.csv'.")