# Librerías
from sklearn.metrics import (accuracy_score, precision_score, recall_score)
from sklearn.metrics import (brier_score_loss, roc_auc_score, confusion_matrix)
from sklearn.metrics import auc as func_auc
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import numpy as np
from sklearn.preprocessing import LabelBinarizer
warnings.filterwarnings('ignore')

def one_hot_encode_labels(y):
    lb = LabelBinarizer()
    y_binarized = lb.fit_transform(y)
    # Asegurarse de que y_binarized tiene dos columnas si es un problema binario
    if len(lb.classes_) == 2:
        y_binarized = np.hstack([1 - y_binarized, y_binarized])
    return y_binarized

def eval_perform(Y,y_pred = None, y_prob = None, train_or_test: str = None, print_results = True):
    """Evalúa el performance de cada modelo

    Args:
        Y (array): Variable objetivo original
        y_pred (array): Salida Y & "hat" (ŷ) que denota predicciones estimadas.
        y_prob (array): Salida Y & "hat" (ŷ) que denota probabilidades estimadas.
        train_or_test (string): "Entrenamiento" o "Prueba"
        print_results(bool): True if is desired to print the results.
    """
    accu, prec, reca, brier_score, auc, cfm, pr_auc = None, None, None, None, None, None, None
    if print_results:
            print(f"\nPerformance del modelo de {train_or_test}")
    if y_pred is not None:
        accu = accuracy_score(Y,y_pred)
        prec = precision_score(Y,y_pred,average='weighted')
        reca = recall_score(Y,y_pred,average='weighted')
        cfm = confusion_matrix(Y, y_pred)
        if print_results:
            print(f' Accu {accu} \n Prec {prec} \n Reca {reca} \n Confusión matrix:\n {cfm}')
    
    if y_prob is not None:
        print(f"\nMétricas de Probabilidad:")
        # Calcular el Brier Score
        brier_score = brier_score_loss(Y, y_prob)
        auc = roc_auc_score(Y, y_prob)
        precision, recall, __ = precision_recall_curve(Y, y_prob)
        pr_auc = func_auc(recall, precision)
        if print_results:
            print(f' Brier Score: {brier_score} \n AUC {auc:.4f} \n PR AUC {pr_auc}')

    return accu, prec, reca, brier_score, auc, cfm, pr_auc


def eval_perform_multi_class(Y, y_pred=None, y_prob=None, n_classes = None, train_or_test: str=None, print_results=True):
    """Evalúa el rendimiento de cada modelo para clasificación multiclase.

    Args:
        Y (array): Variable objetivo original.
        y_pred (array): Salida Y & "hat" (ŷ) que denota predicciones estimadas.
        y_prob (array): Salida Y & "hat" (ŷ) que denota probabilidades estimadas.
        train_or_test (string): "Entrenamiento" o "Prueba".
        print_results(bool): True si se desean imprimir los resultados.
    """
    accu, prec, reca, brier_score, cfm = None, None, None, None, None
    if print_results:
        print(f"\nPerformance del modelo de {train_or_test}")
    if y_pred is not None:
        accu = accuracy_score(Y, y_pred)
        prec = precision_score(Y, y_pred, average='weighted')
        reca = recall_score(Y, y_pred, average='weighted')
        cfm = confusion_matrix(Y, y_pred)
        if print_results:
            print(f' Accu {accu} \n Prec {prec} \n Reca {reca} \n Matriz de confusión:\n {cfm}')
    
    if y_prob is not None and y_prob.shape[1] > 1:
        print("\nMétricas de Probabilidad:")
        brier_score = np.mean([brier_score_loss(Y == i, y_prob[:, i]) for i in range(y_prob.shape[1])])
        # average='macro' para promediar sin considerar el desbalance de clases o average='weighted' para considerar el desbalance de clases).
        # auc = roc_auc_score(Y, y_prob, multi_class="ovr", average="weighted")
        if print_results:
            print(f" Brier Score: {brier_score}")
            plot_precision_recall_curve(n_classes, Y, y_prob, train_or_test)
            plot_roc_curve(n_classes, Y, y_prob, train_or_test)


    return accu, prec, reca, brier_score, cfm


import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve
def plot_precision_recall_curve(n_classes, y, y_prob, model_name):
    y = one_hot_encode_labels(y)
    # precision recall curve
    precision = dict()
    recall = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y[:, i],
                                                            y_prob[:, i])
        plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))
        
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(loc="best")
    plt.title(f"precision vs. recall curve {model_name}")
    plt.show()

def plot_roc_curve(n_classes, y, y_prob, model_name):
    y = one_hot_encode_labels(y)
    # roc curve
    fpr = dict()
    tpr = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y[:, i],
                                    y_prob[:, i])
        plt.plot(fpr[i], tpr[i], lw=2, label='class {}'.format(i))

    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.legend(loc="best")
    plt.title(f"ROC curve {model_name}")
    plt.show()


def show_target_balance(y):
    """Shows the balance of a class in a given dataset with target variable only

    Args:
        y (Dataframe): Pandas dataframe with target variable only
    """
    conteo_clases = y.value_counts()

    # Calcular el porcentaje de cada clase
    total = len(y)

    # Establecer el estilo del gráfico
    sns.set_theme(style="whitegrid")

    # Crear el gráfico de barras
    plt.figure(figsize=(10, 6))
    barplot = sns.barplot(x=conteo_clases.index, y=conteo_clases.values)

    # Añadir etiquetas con el número y porcentaje de cada clase
    for p in barplot.patches:
        altura = p.get_height()
        barplot.text(p.get_x() + p.get_width()/2., altura + 0.1, f'{int(altura)}\n({altura/total:.2%})', ha="center")

    # Añadir títulos y etiquetas
    plt.title('Distribución de Clases')
    plt.xlabel('Clase')
    plt.ylabel('Número de Instancias')

    # Mostrar el gráfico
    plt.show()