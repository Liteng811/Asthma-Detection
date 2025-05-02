from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, accuracy_score,
                             confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Restore default font settings
plt.rcParams.update(plt.rcParamsDefault)  # Reset previous Chinese font settings

file_path = r'C:\Users\1\Desktop\pca-result.xlsx'

try:
    # ==================== Data Preprocessing ====================
    pca_data = pd.read_excel(file_path).dropna()
    pca_cols = ['pca1', 'pca2', 'pca3', 'pca4', 'pca5']
    pca_data[pca_cols] = pca_data[pca_cols].apply(pd.to_numeric, errors='coerce')

    X = pca_data[pca_cols]
    y = pca_data['group'].map({'AB': 0, 'HC': 1})

    # ==================== Data Splitting ====================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    # ==================== Model Training ====================
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(probability=True))
    ])

    param_grid = {
        'svc__C': [0.1, 1, 10, 100],
        'svc__gamma': ['scale', 'auto', 0.1, 1],
        'svc__kernel': ['linear', 'rbf']
    }

    grid_search = GridSearchCV(
        pipe,
        param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        refit=True,
        verbose=2,
        scoring='roc_auc'
    )
    grid_search.fit(X_train, y_train)

    print("\nBest Parameters:", grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # ==================== Cross-Validation ROC ====================
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    mean_fpr = np.linspace(0, 1, 100)
    tprs, aucs = [], []

    plt.figure(figsize=(8, 6))
    for i, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
        model = clone(best_model).fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        y_prob = model.predict_proba(X_train.iloc[val_idx])[:, 1]

        fpr, tpr, _ = roc_curve(y_train.iloc[val_idx], y_prob)
        roc_auc = auc(fpr, tpr)

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc)

        plt.plot(fpr, tpr, alpha=0.3, lw=1,
                 label=f'Fold {i} (AUC={roc_auc:.2f})')

    # Mean ROC curve
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    plt.plot(mean_fpr, mean_tpr, color='b', lw=2,
             label=f'Mean ROC (AUC={mean_auc:.2f}±{std_auc:.2f})')
    plt.fill_between(mean_fpr,
                     np.maximum(mean_tpr - np.std(tprs, axis=0), 0),
                     np.minimum(mean_tpr + np.std(tprs, axis=0), 1),
                     color='grey', alpha=0.2)

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve for 5-Fold Cross-Validation of SVM Model', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.show()

    # ==================== Test Set Evaluation ====================
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['AB', 'HC']))

    # Confusion Matrix (English Version)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        square=True,
        annot_kws={'size': 14, 'color': 'black'},
        cbar=False,
        xticklabels=['AS', 'HC'],
        yticklabels=['AS', 'HC']
    )
    plt.gca().set_aspect('equal')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix for SVM Model', fontsize=14)
    plt.xticks(fontsize=12, rotation=0)
    plt.yticks(fontsize=12, rotation=0)
    plt.tight_layout()
    plt.show()

    # Test Set ROC Curve (English Version)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC Curve (AUC={roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.fill_between(fpr, tpr, alpha=0.1, color='darkorange')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve for SVM Model on the Test Set', fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.show()

except Exception as e:
    print(f"Error occurred: {str(e)}")
