from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, accuracy_score,
                             confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # Add seaborn for confusion matrix visualization

file_path = r'C:\Users\1\Desktop\pca-result.xlsx'

# Read and process data
pca_data = pd.read_excel(file_path)
pca_data = pca_data.dropna()
pca_data[['pca1', 'pca2', 'pca3', 'pca4', 'pca5']] = pca_data[['pca1', 'pca2', 'pca3', 'pca4', 'pca5']].apply(
    pd.to_numeric, errors='coerce')

X = pca_data[['pca1', 'pca2', 'pca3', 'pca4', 'pca5']]
y = pca_data['group'].map({'AB': 0, 'HC': 1})

# Stratified data splitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# Create a pipeline with standardization
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(random_state=42))
])

# Parameter grid
param_grid = {
    'rf__n_estimators': [100, 200],
    'rf__max_depth': [10, 20],
    'rf__min_samples_split': [2, 5],
    'rf__min_samples_leaf': [1, 2],
    'rf__max_features': ['sqrt', 'log2'],
    'rf__bootstrap': [True]
}

# Stratified cross-validation setup
stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Grid search
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=stratified_cv,
    n_jobs=-1,
    verbose=2,
    scoring='roc_auc',
    error_score='raise'
)

try:
    grid_search.fit(X_train, y_train)

    # Output best parameters
    print("Best Parameters:", grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # ========== Cross-Validation ROC Curve ==========
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)

    plt.figure(figsize=(8, 6))
    for i, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
        model = clone(best_model)
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

        model.fit(X_tr, y_tr)
        y_pred_prob = model.predict_proba(X_val)[:, 1]

        fpr, tpr, _ = roc_curve(y_val, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'Fold {i} (AUC = {roc_auc:.2f})')
        print(f"Fold {i} AUC: {roc_auc:.4f}")

    # Calculate mean curve
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    plt.plot(mean_fpr, mean_tpr, color='b',
             label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})',
             lw=2, alpha=0.8)
    plt.fill_between(mean_fpr,
                     np.maximum(mean_tpr - np.std(tprs, axis=0), 0),
                     np.minimum(mean_tpr + np.std(tprs, axis=0), 1),
                     color='grey', alpha=0.2)
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for 5-Fold Cross-Validation of RF Model')
    plt.legend(loc='lower right')
    plt.show()

    # ========== Test Set Evaluation ==========
    # Predict on the test set
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]  # Probability of HC class

    # 1. Accuracy
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")

    # 2. Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['AB', 'HC']))

    # 3. Confusion matrix (square)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True,
                cbar=False,
                xticklabels=['AS', 'HC'],
                yticklabels=['AS', 'HC'])
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix for RF Model', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12, rotation=0)
    plt.tight_layout()
    plt.show()

    # 4. Test set ROC curve
    fpr_test, tpr_test, _ = roc_curve(y_test, y_prob)
    roc_auc_test = auc(fpr_test, tpr_test)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_test, tpr_test, color='darkorange', lw=2,
             label=f'Test ROC (AUC = {roc_auc_test:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.fill_between(fpr_test, tpr_test, alpha=0.2, color='darkorange')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for RF Model on the Test Set')
    plt.legend(loc='lower right')
    plt.show()

except Exception as e:
    print(f"Execution Error: {str(e)}")
