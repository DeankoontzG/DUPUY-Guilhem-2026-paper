import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_recall_curve, auc

def train_and_test_xgboost(dataFrame, features=None, parameters=None, plot=False):
    X = dataFrame[features] if features else dataFrame.drop(["target", "u", "v", "label"], axis=1)
    y = dataFrame['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
            
    if parameters is None:
        parameters = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_child_weight': 5,
            'objective': 'binary:logistic',
            'tree_method': 'hist',
            'random_state': 42,
            'n_jobs': -1 # Utilise tous les cœurs pour aller plus vite
        }
    model = XGBClassifier(**parameters)
    model.fit(X_train, y_train)
    
    test_stats = get_performance_metrics(model, X_test, y_test, prefix="Test_")

    return test_stats, model, X_train, y_train, X_test, y_test

def get_performance_metrics(model, X, y, prefix=""):
    probs = model.predict_proba(X)[:, 1]
    preds = (probs > 0.5).astype(int)
    
    precision, recall, _ = precision_recall_curve(y, probs)

    stats = {
        f'{prefix}AP': average_precision_score(y, probs),
        f'{prefix}AUC-ROC': roc_auc_score(y, probs),
        f'{prefix}F1-Score': f1_score(y, preds),
        f'{prefix}AUC-PR': auc(recall, precision)
    }
    
    return pd.DataFrame([stats])