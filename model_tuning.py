# Librerías
import pandas as pd
from sklearn.model_selection import train_test_split
# Warnings
import warnings
warnings.filterwarnings('ignore')
# Internal tool and helpers
import model_helpers as mh
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
CLASS_NAMES = ["Negras", "Blancas"]

data_path = "../CHESS/data/df_3_cod.csv"
df_3 = pd.read_csv(data_path)

# X e Y 
X = df_3.copy()
y_name = "winner_cod"
# X es el dataframe eliminando la variable de salida. Eliminando también 'moves' que ya está representado
X = X.drop(columns=['rated','winner',y_name, 'current_turn','time_increment','opening_code','opening_fullname','opening_shortname','opening_variation','moves_fen'])
X = X.drop(columns=['game_id', 'white_rating', 'black_rating', 'moves', 'current_turn_cod', 'opening_moves', 'rated_cod', 'current_turn_cod', 'time_increment_cod', 'opening_code_cod', 'opening_fullname_cod', 'opening_shortname_cod', 'opening_variation_cod', 'moves_fen_cod'])# Y es un array unidimensional (ravel) de la variable de salida
Y = df_3[y_name].ravel()
print(X.columns)
# división en train y test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.3)

# XGBOOST Simple
# Crear un clasificador XGBoost
xgboost = xgb.XGBClassifier()
# Entrenar el modelo en los datos de entrenamiento
xgboost.fit(X_train, Y_train)
# Evaluación del modelo
Yhat_xgboost_test = xgboost.predict(X_test)
Yhat_xgboost_train = xgboost.predict(X_train)
Yhat_xgboost_test_prob = xgboost.predict_proba(X_test)
Yhat_xgboost_train_prob = xgboost.predict_proba(X_train)
mh.eval_perform_multi_class(Y_test,Yhat_xgboost_test, Yhat_xgboost_test_prob,CLASS_NAMES,"Prueba Xgboost blancas y negras modelo sencillo")
mh.eval_perform_multi_class(Y_train,Yhat_xgboost_train, Yhat_xgboost_train_prob,CLASS_NAMES,"Entrenamiento Xgboost blancas y negras  modelo sencillo")

# Optimización de hiperparámetros XGboost
# Define the hyperparameter space
space = {
    'max_depth': hp.choice('max_depth', np.arange(1, 14, dtype=int)),
    'learning_rate': hp.loguniform('learning_rate', -5, -2),
    'subsample': hp.uniform('subsample', 0.5, 1)
}

# Define the objective function to minimize
def objective(params):
    xgb_model = xgb.XGBClassifier(**params)
    xgb_model.fit(X_train, Y_train)
    y_pred = xgb_model.predict(X_test)
    score = accuracy_score(Y_test, y_pred)
    return {'loss': -score, 'status': STATUS_OK}

trials = Trials()
# Perform the optimization
best_params = fmin(objective, space, algo=tpe.suggest, max_evals=250, trials=trials)
print("Best set of hyperparameters: ", best_params)
# Extraer la función de pérdida de cada resultado en trials
losses = [x['result']['loss'] for x in trials.trials]

# Graficar la función de pérdida
plt.figure(figsize=(10, 5))
plt.plot(losses, label='Función de Pérdida')
plt.xlabel('Iteraciones')
plt.ylabel('Pérdida Negativa de Exactitud')
plt.title('Función de Pérdida durante la Optimización de Hiperparámetros')
plt.legend()
plt.show()

# Crear un clasificador XGBoost a partir de los mejores parámetros
xgboost_best = xgb.XGBClassifier(**best_params)
# Entrenar el modelo en los datos de entrenamiento
xgboost_best.fit(X_train, Y_train)
# Evaluación del modelo
Yhat_xgboost_test = xgboost_best.predict(X_test)
Yhat_xgboost_train = xgboost_best.predict(X_train)
Yhat_xgboost_test_prob = xgboost_best.predict_proba(X_test)
Yhat_xgboost_train_prob = xgboost_best.predict_proba(X_train)
mh.eval_perform_multi_class(Y_test,Yhat_xgboost_test, Yhat_xgboost_test_prob,CLASS_NAMES,"Prueba Xgboost blancas y negras optimización de hiperparámetros")
mh.eval_perform_multi_class(Y_train,Yhat_xgboost_train, Yhat_xgboost_train_prob,CLASS_NAMES,"Entrenamiento Xgboost blancas y negras optimización de hiperparámetros")

# Feature importance
import matplotlib.pyplot as plt
# Feature importance
importance = xgboost.feature_importances_

df_importancia = pd.DataFrame({
    'Caracteristica': X.columns,
    'Importancia': importance
})

umbral = 0
# Filtrar los nombres de las características importantes
df_importancia = df_importancia[df_importancia['Importancia'] > umbral]
df_importancia = df_importancia.sort_values(by='Importancia', ascending=True)

features = df_importancia['Caracteristica']
importance = df_importancia['Importancia']


# Crear un gráfico de barras horizontal
plt.figure(figsize=(10, 8))
plt.barh(features, importance, color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature importance')
plt.show()


# Validación del modelo
import extract_features as ef
fen = '1r1q1rk1/pb2bppp/4p3/6N1/2pPQ3/4P2P/PP3PP1/R1B2RK1 w - - 1 17'
moves = 4
new_sample_df = ef.count_all_features(fen, moves)
print(new_sample_df.T)

print("Posición Blancas: ")
pd.set_option('display.max_columns', None)
print(ef.get_all_features_uf(fen, True))

print("Posición Negras: ")
pd.set_option('display.max_columns', None)
print(ef.get_all_features_uf(fen, False))


# Decisión del modelo para clasificación 
prob = xgboost.predict_proba(new_sample_df)
print(xgboost.classes_) # 0: negras. 1: blancas
print(f"Negras: {'{:.6f}'.format(prob[0][0])}. Blancas:{'{:.6f}'.format(prob[0][1])}")
prob = xgboost_best.predict_proba(new_sample_df)
print(xgboost_best.classes_) # 0: negras. 1: blancas
print(f"Negras: {'{:.6f}'.format(prob[0][0])}. Blancas:{'{:.6f}'.format(prob[0][1])}")


'''
Guardar modelo elegido
En un entorno de producción tiene menos costo computacional seleccionar el modelo más sencillo para entrenar con nuevos datos, y la variación es despreciable. Por tanto se selecciona ese modelo. 
'''
import pickle
pickle.dump(xgboost, open("./pickles/models/xgoboost_model0416_18:06.pkl", 'wb'))