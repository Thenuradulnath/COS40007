import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import os

# Load the datasets
data = pd.read_csv(r'Portfolio 1\Outs\normalized_data.csv')
features_data = pd.read_csv(r'Portfolio 1\Outs\features_data.csv')
selected_features_data = pd.read_csv(r'Portfolio 1\Outs\selected_features_data.csv')

# Ensure 'PE_Class' exists in 'data' if using it in 'Model 2'
if 'PE_Class' not in data.columns:
    data['PE_Class'] = selected_features_data['PE_Class']

# Define feature sets
feature_sets = {
    "Model 1": data[['AT', 'V', 'AP', 'RH']],  # features from normalized_data.csv
    "Model 2": data[['AT', 'V', 'AP', 'RH', 'PE_Class']],  # features from normalized_data.csv with class labels
    "Model 3": features_data[['AT', 'V', 'AP', 'RH', 'AT_V', 'AT_AP', 'V_RH', 'AP_RH']],  # composite features
    "Model 4": selected_features_data[['AT', 'V', 'AP', 'RH', 'AT_V', 'AT_AP', 'V_RH', 'AP_RH']],  # selected features
    "Model 5": selected_features_data[['AT', 'V', 'AP', 'RH']]  # selected features without composite features
}

# Initialize results dictionary
results = {}

# Train and evaluate models
for model_name, features in feature_sets.items():
    X = features
    y = selected_features_data['PE_Class']  # Ensure 'PE_Class' is available in selected_features_data
    
    # Ensure that 'X' and 'y' have the same number of rows
    if len(X) != len(y):
        raise ValueError(f"Feature set '{model_name}' has a different number of rows compared to target variable.")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    
    # Initialize the model
    clf = DecisionTreeClassifier()
    
    # Train the model
    clf.fit(X_train, y_train)
    
    # Predict
    y_pred = clf.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    results[model_name] = accuracy

# Create a DataFrame for results
results_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy'])

# Define the output directory
output_dir = 'Portfolio 1\\Outs'

# Define the output file path for the table
output_file_path = os.path.join(output_dir, 'model_comparison.csv')

# Save the results to a CSV file
results_df.to_csv(output_file_path, index=False)

print(f"Model development completed and accuracies saved to '{output_file_path}'.")
