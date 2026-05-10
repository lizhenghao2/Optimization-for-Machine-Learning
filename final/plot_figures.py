# plot the results of the prediction
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix


def _save_figure(save_path, default_filename):
    if save_path is None:
        save_path = os.path.join('figures', default_filename)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes=None, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    if classes is None:
        classes = np.unique(np.concatenate([y_true, y_pred]))
        
    plt.figure(figsize=(6, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    # Add numbers inside cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                cm[i, j],
                ha='center',
                va='center',
                color='black',
                fontsize=12
            )

    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    _save_figure(save_path, 'confusion_matrix.png')
    
    
def plot_precision_recall_curve(y_true, y_scores, save_path=None):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, marker='.')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid()
    _save_figure(save_path, 'precision_recall_curve.png')
    
def plot_roc_curve(y_true, y_scores, save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, marker='.', label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid()
    _save_figure(save_path, 'roc_curve.png')
    
def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
    if hasattr(model, 'coef_'):
        importance = model.coef_[0]
    elif hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        raise ValueError("Model does not have feature importance attribute.")
    
    indices = np.argsort(importance)[::-1][:top_n]
    plt.figure(figsize=(8, 6))
    plt.bar(range(top_n), importance[indices], align='center')
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=90)
    plt.title('Top Feature Importance')
    plt.tight_layout()
    _save_figure(save_path, 'feature_importance.png')