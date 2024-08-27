import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the cleaned dataset
data = pd.read_csv(r'Portfolio 1\Outs\cleaned_data.csv')

# Class labeling (creating categories based on the target variable 'PE')
bins = [420, 435, 450, 465, 480, 496]
labels = [1, 2, 3, 4, 5]
data['PE_Class'] = pd.cut(data['PE'], bins=bins, labels=labels, right=False)

# Normalization
scaler = MinMaxScaler()
data[['AT', 'V', 'AP', 'RH']] = scaler.fit_transform(data[['AT', 'V', 'AP', 'RH']])

# Save the modified dataframe
print(data.head())
data.to_csv(r'Portfolio 1\Outs\normalized_data.csv', index=False)
print("Class labeling and normalization completed and saved to 'normalized_data.csv'.")
