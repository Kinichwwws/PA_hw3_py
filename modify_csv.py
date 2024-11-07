import pandas as pd
import numpy as np


def convert_csv_types(input_file, output_file):
    """
    Convert CSV data types according to specified schema

    Parameters:
    input_file (str): Path to input CSV file
    output_file (str): Path to output CSV file
    """
    # Read the CSV file
    df = pd.read_csv(input_file)

    # Define column types based on the schema
    column_types = {
        'Class': 'category',  # binomial type as category
        'F1': 'int32',
        'F2': 'int32',
        'F3': 'int32',
        'F4': 'int32',
        'F5': 'int32',
        'F6': 'int32',
        'F7': 'int32',
        'F8': 'int32',
        'F9': 'int32',
        'F10': 'int32',
        'F11': 'int32',
        'F12': 'int32',
        'F13': 'int32',
        'F14': 'float64',  # real type
        'F15': 'int32',
        'F16': 'int32',
        'F17': 'float64',  # real type
        'F18': 'int32',
        'F19': 'int32'
    }

    # Convert data types
    for column, dtype in column_types.items():
        if dtype == 'int32':
            # Handle any missing values before converting to int
            df[column] = pd.to_numeric(df[column], errors='coerce').fillna(0).astype('int32')
        elif dtype == 'float64':
            df[column] = pd.to_numeric(df[column], errors='coerce').astype('float64')
        elif dtype == 'category':
            df[column] = df[column].astype('category')

    # Save the converted dataframe
    df.to_csv(output_file, index=False)

    # Print information about the converted dataset
    print("\nDataset Information:")
    print("-" * 50)
    print("Number of rows:", len(df))
    print("\nColumn Types:")
    for col in df.columns:
        print(f"{col}: {df[col].dtype}")

    return df


# Example usage
if __name__ == "__main__":
    # Replace these with your actual file paths
    input_file = "hepatitis/hepatitis.data.csv"
    output_file = "output_converted.csv"

    try:
        df = convert_csv_types(input_file, output_file)
        print("\nData conversion completed successfully!")

        # Display first few rows of converted data
        print("\nFirst few rows of converted data:")
        print(df.head())

    except Exception as e:
        print(f"An error occurred: {str(e)}")