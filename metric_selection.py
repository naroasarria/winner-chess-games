"""
================================================================================
3. Funcions de mètriques i visualització de models de classificació
================================================================================

Aquest mòdul agrupa totes les funcions utilitzades per avaluar i analitzar 
el rendiment dels models del projecte de predicció del guanyador de partides 
d’escacs. Inclou eines per calcular mètriques numèriques i per generar 
visualitzacions estàndard en classificació binària.

Funcionalitats principals:
    - Informe complet amb accuracy, precision, recall i F1 (classification_report).
    - Corba ROC i càlcul de l’AUC.
    - Corba Precision–Recall i càlcul del Average Precision (AP).
    - Matriu de confusió amb visualització integrada.
    - Funció individual per dibuixar únicament la corba ROC.

Aquest mòdul està pensat per ser importat i utilitzat en el notebook 
`model_partida.ipynb` juntament amb els models entrenats.

Autor: Naroa Sarrià Gil & Inés Gómez Carmona
Projecte: Predicció del guanyador d’escacs (Lichess)
UAB - Aprenentatge Computacional (2025)
"""


import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.metrics import accuracy_score, roc_curve,auc, precision_recall_curve, classification_report, confusion_matrix, ConfusionMatrixDisplay, average_precision_score

sns.set_style('whitegrid')

def metriques(model,X_test,y_test):
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    
    
def grafiques(model,X_test,y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)


    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 2, figure=fig) 

    # ROC Curve
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax1.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend(loc="lower right")

    # Precision-Recall Curve 
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(recall, precision, label=f'AP = {ap:.2f}',color='purple', lw=2)
    ax2.set_ylim(-0.05,1.05)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc="lower left")

    # Confusion Matrix 
    ax3 = fig.add_subplot(gs[1, :])  
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(ax=ax3, colorbar=False)
    plt.grid(False)
    ax3.set_title('Confusion Matrix')

    plt.tight_layout()
    plt.show()

    
def curva_ROC(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig = plt.figure(figsize=(5.5, 5))
    ax1 = fig.add_subplot(111)

    ax1.plot(fpr, tpr, color='blue', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax1.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend(loc="lower right")

    plt.tight_layout()
    plt.show()
