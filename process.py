from operator import index
from unicodedata import numeric

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import seaborn as sns
import os

# basic data information
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)

        numeric_columns = []
        categorical_columns = []

        for column in df.columns:
            if df[column].dtype in ['int64', 'float64']:
                numeric_columns.append(column)
            else:
                categorical_columns.append(column)

        print("\n=== Basic Information ===")
        print(df.info())

        print("\n=== Descriptive Statistics ===")
        print(df.describe(include='all'))

        print("\n=== Missing Value Analysis ===")
        missing_values = df.isnull().sum()
        print(missing_values[missing_values > 0])

        # Get numeric columns for later processing
        print("\nNumeric columns:", numeric_columns)
        print("Categorical columns:", categorical_columns)

        return df, numeric_columns, categorical_columns

    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None


def handle_missing_values(df, numeric_columns, categorical_columns, n_neighbors=10):
    try:
        # copy
        df_imputed = df.copy()

        #numberic columns with KNN
        if numeric_columns:
            df_imputed[numeric_columns] = df_imputed[numeric_columns].replace(['?', 'NA', '-'], np.nan)
            num_imputer = KNNImputer(n_neighbors=n_neighbors)
            df_imputed[numeric_columns] = num_imputer.fit_transform(df_imputed[numeric_columns])

        if categorical_columns:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            df_imputed[categorical_columns] = cat_imputer.fit_transform(df_imputed[categorical_columns])

        missing_after = df_imputed.isnull().sum()
        print("\nMissing values after imputation:")
        print(missing_after[missing_after > 0])

        return df_imputed

    except Exception as e:
        print(f"Error in handling missing values: {str(e)}")
        return df

def encode_categorical_features(df, categorical_columns, label_column):
    df_encoded = df.copy()

    encoding_columns = [col for col in categorical_columns if col != label_column]

    if encoding_columns:
        onehot = OneHotEncoder(sparse_output=False, handle_unknown='ignore', min_frequency=0.01 )
        encoded_features = onehot.fit_transform(df_encoded[encoding_columns])
        feature_names = onehot.get_feature_names_out(encoding_columns)

        encoded_df = pd.DataFrame(
            encoded_features,
            columns=feature_names,
            index = df_encoded.index
        )

        # Drop original categorical columns and add encoded ones
        df_encoded = df_encoded.drop(columns=encoding_columns)
        df_encoded = pd.concat([df_encoded, encoded_df], axis=1)

    return df_encoded

def detect_outliers(df, numeric_columns, n_neighbors=10, contamination=0.1):
    """
    Detect outliers using Local Outlier Factor (LOF)
    """
    if len(numeric_columns) < 2:
        print("Not enough numeric features for outlier detection")
        return df

    try:
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        outliers = lof.fit_predict(df[numeric_columns])

        df['is_outlier'] = outliers

        n_outliers = sum(outliers == -1)
        print(f"\nOutliers detected: {n_outliers} cases ({n_outliers / len(df) * 100:.2f}%)")

        if len(numeric_columns) >= 2:
            plt.figure(figsize=(10, 6))
            plt.scatter(df[df['is_outlier'] == 1][numeric_columns[0]],
                        df[df['is_outlier'] == 1][numeric_columns[1]],
                        c='blue', label='Normal')
            plt.scatter(df[df['is_outlier'] == -1][numeric_columns[0]],
                        df[df['is_outlier'] == -1][numeric_columns[1]],
                        c='red', label='Outlier')
            plt.xlabel(numeric_columns[0])
            plt.ylabel(numeric_columns[1])
            plt.title('Outliers Distribution')
            plt.legend()
            plt.show()

        return df

    except Exception as e:
        print(f"Error in outlier detection: {str(e)}")
        return df


def standardize_features(df, numeric_columns, label_column):
    """
    Standardize numeric features
    """
    df_scaled = df.copy()
    scaling_columns = [col for col in numeric_columns if col != label_column]

    if scaling_columns:
        scaler = StandardScaler()
        df_scaled[scaling_columns] = scaler.fit_transform(df[scaling_columns])

    return df_scaled

def reduce_dimensions(df, numeric_columns, label_column= None, variance_threshold=0.90):
    """
    Perform PCA for dimensionality reduction
    检查一下占比 95% 降维
    """
    labels = df[label_column] if label_column in df.columns else None
    # Remove label column from PCA if it's in numeric columns
    feature_columns = [col for col in numeric_columns if col != label_column]

    if len(feature_columns) < 2:
        print("Not enough numeric features for PCA")
        return df, None, None

    pca = PCA()
    pca.fit(df[feature_columns])

    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i + 1}' for i in range(len(pca.components_))],
        index=feature_columns
    )

    # 计算并输出特征重要性排名（前20个特征）
    feature_importance = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': [sum(abs(loadings.iloc[i]) * pca.explained_variance_ratio_)
                       for i in range(len(feature_columns))]
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    feature_importance['Rank'] = range(1, len(feature_columns) + 1)

    print("\n=== Top 20 Feature Importance ===")
    pd.set_option('display.max_rows', None)
    print(feature_importance.head(20).to_string(index=False))

    # 计算和显示前20个主成分的方差解释比例
    explained_variance = pca.explained_variance_ratio_[:20]  # 只取前20个
    cumulative_variance = np.cumsum(explained_variance)

    variance_df = pd.DataFrame({
        'PC': [f'PC{i + 1}' for i in range(len(explained_variance))],
        'Variance_Ratio': explained_variance,
        'Cumulative_Ratio': cumulative_variance
    })
    variance_df['Variance_Ratio'] = variance_df['Variance_Ratio'].map('{:.4f}'.format)
    variance_df['Cumulative_Ratio'] = variance_df['Cumulative_Ratio'].map('{:.4f}'.format)

    print("\n=== Top 20 PCA Components Variance ===")
    print(variance_df.to_string(index=False))

    # 原有的PCA降维逻辑
    n_components = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= variance_threshold) + 1
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(df[feature_columns])

    pca_df = pd.DataFrame(
        data=pca_result,
        columns=[f'PC{i + 1}' for i in range(n_components)]
    )

    if label_column in df.columns:
        pca_df[label_column] = df[label_column]

    return pca_df, pca.explained_variance_ratio_, np.cumsum(pca.explained_variance_ratio_)

def encode_labels(df, label_column):
    """
    Encode categorical labels to numerical values
    """
    df_encoded = df.copy()
    if label_column in df.columns:
        # 将 'Alive' 编码为 1, 'Dead' 编码为 0
        label_map = {'Alive': 1, 'Dead': 0}
        df_encoded[label_column] = df_encoded[label_column].map(label_map)
        print(f"\nLabel encoding mapping: {label_map}")
        print(f"Label distribution after encoding:\n{df_encoded[label_column].value_counts()}")
    return df_encoded

def preprocess_data(file_path, label_column=None):
    """
    Main function to run the entire preprocessing pipeline
    """
    # Load and examine data
    df, numeric_columns, categorical_columns = load_data(file_path)

    if df is not None:
        df = encode_labels(df, label_column)
        # Handle missing values
        df_clean = handle_missing_values(df, numeric_columns, categorical_columns)

        # Encode categorical features
        df_encoded = encode_categorical_features(df_clean, categorical_columns, label_column)

        # Update numeric columns after encoding
        numeric_columns = df_encoded.select_dtypes(include=['float64', 'int64']).columns.tolist()

        # Detect outliers
        feature_columns = [col for col in numeric_columns if col != label_column]
        df_encoded = detect_outliers(df_encoded, feature_columns)

        # Standardize features
        df_scaled = standardize_features(df_encoded, feature_columns, label_column)

        if label_column:
            df_scaled[label_column] = df_encoded[label_column]

        # Perform dimensionality reduction
        pca_df, explained_variance, cumulative_variance = reduce_dimensions(
            df_scaled, numeric_columns, label_column
        )

        print("\nSaving processed data...")
        print(f"Preprocessed data shape: {df_scaled.shape}")
        print(f"PCA data shape: {pca_df.shape if pca_df is not None else None}")

        # Save processed data
        if label_column:
            print(f"Label column values in preprocessed data: {df_scaled[label_column].value_counts()}")
            if pca_df is not None:
                print(f"Label column values in PCA data: {pca_df[label_column].value_counts()}")

        df_scaled.to_csv('preprocessed_data.csv', index=False)
        if pca_df is not None:
            pca_df.to_csv('pca_data.csv', index=False)

        print("\nPreprocessing completed successfully!")
        return df_scaled, pca_df

    return None, None

# 使用示例
if __name__ == "__main__":
    file_path = 'Breast_Cancer_dataset.csv'
    label_column = 'Status'
    df_scaled, pca_df = preprocess_data(file_path, label_column)




