# Librerías
from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score)
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def eval_perform_class(Y,Yhat, train_or_test = None, print_results = False):
    """Evalúa el performance de cada modelo

    Args:
        Y (array): Variable objetivo original
        Yhat (array): Salida Y & "hat" (ŷ) que denota predicciones estimadas.
        train_or_test (string): "Entrenamiento" o "Prueba"
        print_results(bool): True if is desired to print the results.
    """
    accu = accuracy_score(Y,Yhat)
    prec = precision_score(Y,Yhat,average='weighted')
    reca = recall_score(Y,Yhat,average='weighted')
    if print_results:
        print(f"\nPerformance del modelo de {train_or_test}")
        print(f'Accu {accu} \n Prec {prec} \n Reca {reca}')
    return (accu, prec, reca)