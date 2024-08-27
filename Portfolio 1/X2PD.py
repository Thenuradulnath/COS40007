import pandas as pd

def convert_excel_to_csv(input_file, output_file):
    df = pd.read_excel(input_file, engine='openpyxl' if input_file.endswith('.xlsx') else 'xlrd')
    df.to_csv(output_file, index=False)
    print(f"New file saved as --> '{output_file}'.")

# Specify the input Excel file and output CSV file
input_file = r'Portfolio 1\dataSet\Folds5x2_pp.xlsx'  
output_file = 'Portfolio 1\dataSet\converted_data.csv'

convert_excel_to_csv(input_file, output_file)
