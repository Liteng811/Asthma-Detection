from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = r'C:\Users\1\Desktop\pca-result.xlsx'

try:
    pca_data = pd.read_excel(file_path)

    pca_data = pca_data.dropna()
    pca_data[['pca1', 'pca2', 'pca3', 'pca4', 'pca5']] = pca_data[['pca1', 'pca2', 'pca3', 'pca4', 'pca5']].apply(pd.to_numeric, errors='coerce')

    X = pca_data[['pca1', 'pca2', 'pca3', 'pca4', 'pca5']]
    y = pca_data['group'].map({'AB': 0, 'HC': 1})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 超参数调优：使用 GridSearchCV 调整 C 和 gamma 参数
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 1],
        'kernel': ['linear', 'rbf']
    }
    grid_search = GridSearchCV(SVC(probability=True), param_grid, refit=True, verbose=2, cv=5)
    grid_search.fit(X_train, y_train)

    # 输出最佳参数
    print("最佳参数: ", grid_search.best_params_)

    # 使用最佳参数的模型预测
    model = grid_search.best_estimator_
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # 获取预测的概率

    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # 生成混淆矩阵
    cm = confusion_matrix(y_test, y_pred)

    # 绘制混淆矩阵
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['AB', 'HC'], yticklabels=['AB', 'HC'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # 绘制 ROC 曲线
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='b', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')  # 对角线（随机猜测）
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

