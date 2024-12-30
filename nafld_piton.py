import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'C:\\Users\\ZEHRA\\Desktop\\veri ödev 1\\NAFLD.xlsx'
data = pd.read_excel(file_path)

#Sütunlardaki eksik veri yüzdesinin kontrolü
missing_percentage = data.isnull().mean() * 100

#Çok eksik sütunları çıkarma (sınırı %50 belirledik )
data.drop(columns=missing_percentage[missing_percentage > 50].index, inplace=True)

#Eksik verileri uygun metodlarla doldurma
fill_strategies = {
    'numeric_median': ['Age', 'Body Mass Index', 'Waist Circumference', 'Hemoglobin - A1C'],
    'forward_fill': ['AST', 'ALT', 'GGT'],
    'mode': ['Diyabetes Mellitus (No=0, Yes=1)', 'NAS score according to Kleiner']
}

for col in fill_strategies['numeric_median']:
    if col in data.columns:
        data[col] = data[col].fillna(data[col].median())
for col in fill_strategies['forward_fill']:
    if col in data.columns:
        data[col] = data[col].ffill()
for col in fill_strategies['mode']:
    if col in data.columns:
        data[col] = data[col].fillna(data[col].mode()[0])

#Hedef ve özelliklerin belirlenmesi
selected_features = [
    'Age', 'Body Mass Index', 'Waist Circumference', 'Diyabetes Mellitus (No=0, Yes=1)',
    'AST', 'ALT', 'GGT', 'NAS score according to Kleiner', 'Hemoglobin - A1C'
]
selected_features = [col for col in selected_features if col in data.columns]
X = data[selected_features]
y = data['Fibrosis status (No=0, Yes=1) (Fibrosis 1 and above, there is Fibrosis)']

#Eğitim ve Test verilerine bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Modellerin Tanımlanması
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
}

#Model Eğitimi ve Değerlendirme
results = {}
conf_matrices = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    conf_matrices[name] = confusion_matrix(y_test, y_pred)
    results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_proba) if y_proba is not None else None,
    }

#Performans Sonuçları
results_df = pd.DataFrame(results).T
print("Model Performans Sonuçları:")
print(results_df)

# Confusion Matrix
for name, matrix in conf_matrices.items():
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Fibrosis', 'Fibrosis'],
                yticklabels=['No Fibrosis', 'Fibrosis'])
    plt.title(f'Confusion Matrix: {name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

# ROC Eğrisi
plt.figure(figsize=(8, 5))
for name, model in models.items():
    if hasattr(model, 'predict_proba'):
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        plt.plot(fpr, tpr, label=f'{name} (AUC = {results[name]["AUC"]:.2f})')
plt.title('ROC Eğrisi')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.show()

#En iyi model
best_model = results_df['AUC'].idxmax()
print(f"En iyi performans gösteren model: {best_model}")

#Logistic Regression ve Random Forest Accuracy Karşılaştırma Tablosu
accuracy_comparison = results_df.loc[['Logistic Regression', 'Random Forest'], ['Accuracy']]
print("\nLogistic Regression ve Random Forest Accuracy Karşılaştırma Tablosu:")
print(accuracy_comparison)
plt.figure(figsize=(6, 4))
accuracy_comparison.plot(kind='bar', legend=False, color=['orange', 'blue'])
plt.title('Logistic Regression vs Random Forest Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Model')
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
