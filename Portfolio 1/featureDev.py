import os
import pandas as pd

# Load the dataset with engineered features
data = pd.read_csv(r'Portfolio 1\Outs\features_data.csv')

# Selecting relevant features based on correlation or EDA results
selected_features = data[['AT', 'V', 'AP', 'RH', 'AT_V', 'AT_AP', 'V_RH', 'AP_RH', 'PE_Class']]

# Define the output directory
output_dir = r'Portfolio 1\Outs'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Define the output file path
output_file_path = os.path.join(output_dir, 'selected_features_data.csv')

# Save the selected features dataframe
selected_features.to_csv(output_file_path, index=False)

# Print confirmation
print(f"Feature selection completed and saved to '{output_file_path}'.")
