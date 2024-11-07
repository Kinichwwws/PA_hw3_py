from pyexpat import features

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_split_data(filename, label_column):
    df = pd.read_csv(filename)
    print("load PCA data shape:", df.shape)

    # Separate features (PC columns) and label
    features_cols = [col for col in df.columns if col.startswith('PC')]
    X = df[features_cols]
    y = df[label_column]

    print("\nFeature columns:", features_cols)
    print(f"number of principal components:{len(features_cols)}")

    X = X.to_numpy()
    y = y.to_numpy()

    print(f"\nFeature array shape: {X.shape}")
    print(f"Label array shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test

def grid_search(X_train, y_train, model_type):
    if model_type == 'RandomForest':
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif model_type == 'GradientBoosting':
        model = GradientBoostingClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [3, 4, 5],
            'min_samples_split': [2, 5, 10]
        }
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='accuracy',
        verbose=1,
    )

    grid_result = grid.fit(X_train, y_train)

    print(f"\n===== {model_type} Grid Search Results =====")
    print(f"Best parameters:")
    for param, value in grid.best_params_.items():
        print(f"{param}: {value}")
    print(f"Best cross-validation score: {grid.best_score_:.4f}")

    cv_results = pd.DataFrame(grid.cv_results_)
    params_scores = cv_results[['params', 'mean_test_score', 'std_test_score']]
    params_scores = params_scores.sort_values('mean_test_score', ascending=False)
    print("\nTop 5 Parameter Combinations:")
    for i, row in params_scores.head().iterrows():
        print(f"\nRank {i + 1}:")
        print(f"Parameters: {row['params']}")
        print(f"Mean CV Score: {row['mean_test_score']:.4f} (+/- {row['std_test_score'] * 2:.4f})")

    plt.figure(figsize=(12, 6))
    scores = cv_results['mean_test_score']
    plt.plot(range(len(scores)), sorted(scores, reverse=True))
    plt.title(f'{model_type} - Parameter Combinations Performance')
    plt.xlabel('Parameter Combination Rank')
    plt.ylabel('Mean CV Score')
    plt.grid(True)
    plt.savefig(f'{model_type}_parameter_performance.png')
    plt.close()

    plt.figure(figsize=(15, 5))
    n_params = len(grid.best_params_)
    for i, (param_name, param_values) in enumerate(param_grid.items()):
        plt.subplot(1, n_params, i + 1)
        param_scores = []
        for value in param_values:
            scores = cv_results[cv_results[f'param_{param_name}'].astype(str) == str(value)]['mean_test_score']
            param_scores.append(scores.values)
        plt.boxplot(param_scores, labels=[str(x) for x in param_values])
        plt.title(f'{param_name}\nEffect on Score')
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{model_type}_parameter_analysis.png')
    plt.close()

    return grid.best_estimator_

def euclidean_distances(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def KNN(X_train, y_train, X_test, k):
    max_k = min(19, len(X_train))
    k_range = range(1, max_k + 1)

    best_score = -1
    best_k = 1

    indices = np.random.permutation(len(X_train))
    split = int(0.8 * len(X_train))
    train_idx, val_idx = indices[:split], indices[split:]

    X_train_split = X_train[train_idx]
    y_train_split = y_train[train_idx]
    X_val = X_train[val_idx]
    y_val = y_train[val_idx]

    for curr_k in k_range:
        predictions = []
        for x_val in X_val:
            distances = np.array([euclidean_distances(x_val, x_train) for x_train in X_train_split])
            k_indices = np.argsort(distances)[:curr_k]
            k_nearest_labels = y_train_split[k_indices]
            labels, counts = np.unique(k_nearest_labels, return_counts=True)
            most_common = labels[np.argmax(counts)]
            predictions.append(most_common)

        # accuracy
        score = np.mean(np.array(predictions) == y_val)
        if score > best_score:
            best_score = score
            best_k = curr_k

    print(f"Best k = {best_k} with validation accuracy = {best_score:.4f}")

    prediction = []
    for x_test in X_test:
        distance = np.array([euclidean_distances(x_test, x_train) for x_train in X_train])
        k_indices = np.argsort(distance)[:k]
        k_nearest_labels = y_train[k_indices]
        labels, counts = np.unique(k_nearest_labels, return_counts=True)
        most_common = labels[np.argmax(counts)]
        prediction.append(most_common)
    return np.array(prediction)

def eval(model_name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)

    print(f"\n=== {model_name} Results ===")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'confusion_matrix_{model_name}.png')
    plt.close()

    return accuracy

def plot_res(results):
    results_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy'])
    results_df.to_csv('model_results.csv', index=False)
    print("\nResults saved to 'model_results.csv'")

def main():
    X_train, X_test, y_train, y_test = load_and_split_data('pca_data.csv', 'Status')

    results = {}
    #knn
    print("\nTraining KNN...")
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    k = min(3, len(X_train))
    y_pred_knn = KNN(X_train, y_train, X_test, k=k)
    results["KNN"] = eval("KNN", y_test, y_pred_knn)
    #other model
    models ={
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(),
        # 'Random Forest': RandomForestClassifier(),
        # 'Gradient Boosting': GradientBoostingClassifier(),
        'Neural Net': MLPClassifier()
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = eval(name, y_test, y_pred)

    print("\nPerforming Grid Search for Random Forest...")
    rf_best = grid_search(X_train, y_train, "RandomForest")
    y_pred_rf = rf_best.predict(X_test)
    results["Random Forest (Optimized)"] = eval("Random Forest", y_test, y_pred_rf)

    print("\nPerforming Grid Search for Gradient Boosting...")
    gb_best = grid_search(X_train, y_train, "GradientBoosting")
    y_pred_gb = gb_best.predict(X_test)
    results["Gradient Boosting (Optimized)"] = eval("Gradient Boosting", y_test, y_pred_gb)

    plot_res(results)

if __name__ == '__main__':
    main()

