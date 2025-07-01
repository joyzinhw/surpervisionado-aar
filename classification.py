import matplotlib.pyplot as plt
import numpy as np
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.under_sampling import RandomUnderSampler
from util import read_results

np.random.seed(42)

# Carregar os dados
training_corpus = read_results('resultados_training')
testing_corpus = read_results('resultados_testing')

X = training_corpus[['boolean', 'tf', 'embeddings', 'st', 'wmd']]
y = training_corpus['label']
X_test = testing_corpus[['boolean', 'tf', 'embeddings', 'st', 'wmd']]
y_test = testing_corpus['label']

# Balanceamento
ros = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Dicionário com os modelos
modelos = {
    'SVM': SVC(),
    'Naïve Bayes (BernoulliNB)': BernoulliNB(),
    'Decision Tree (DT)': DecisionTreeClassifier(),
    'MultiLayer Perceptron (MLP)': MLPClassifier(max_iter=1000),
    'Random Forest (RF)': RandomForestClassifier(),
    'XGBoost (XGB)': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# Treinar e avaliar cada modelo
for nome, clf in modelos.items():
    print(f"\n===== {nome} =====")
    clf.fit(X_resampled, y_resampled)
    y_pred = clf.predict(X_test)
    
    print("Relatório de Classificação:")
    print(classification_report(y_test, y_pred))
    
    print("Relatório de Classificação (Imbalanced):")
    print(classification_report_imbalanced(y_test, y_pred))
    
    acc_bal = balanced_accuracy_score(y_test, y_pred)
    print(f"Acurácia balanceada: {acc_bal:.4f}")
    
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_ if hasattr(clf, 'classes_') else np.unique(y_test))
    print("Matriz de confusão:")
    print(cm)
