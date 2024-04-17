# Librerías
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from model_helpers import eval_perform_multi_class
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
CLASS_NAMES = ["Negras", "Blancas"]

# Carga de datos
data_path = "../CHESS/data/df_3_cod.csv"
df_3 = pd.read_csv(data_path)
df_3.columns

# Selección de columnas
X = df_3.copy()
y_name = "winner_cod"
# X es el dataframe eliminando la variable de salida
X = X.drop(columns=['rated','winner',y_name, 'current_turn','time_increment','opening_code','opening_fullname','opening_shortname','opening_variation','moves_fen'])
X = X.drop(columns=['game_id', 'white_rating', 'black_rating', 'moves', 'current_turn_cod', 'opening_moves', 'rated_cod', 'current_turn_cod', 'time_increment_cod', 'opening_code_cod', 'opening_fullname_cod', 'opening_shortname_cod', 'opening_variation_cod', 'moves_fen_cod'])
# Y es un array unidimensional (ravel) de la variable de salida
Y = df_3[y_name].ravel()
print(X.columns)

# División train/test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.3)

# SVM
# Crear un clasificador SVM para clasificación con kernel lineal, polinomial y de funcion de base radial
mod_linear = svm.SVC(kernel='linear',C=1, probability=True)
mod_poly = svm.SVC(kernel='poly',degree=2,C=1, probability=True)
mod_rbf = svm.SVC(kernel='rbf',C=1,gamma='auto', probability=True)
# Entrenamiento del kernel lineal
mod_linear.fit(X_train, Y_train)
# Evaluación: Salida Y & "hat" (ŷ) que denota predicciones estimadas.
Yhat_linear_test = mod_linear.predict(X_test)
Yhat_linear_train = mod_linear.predict(X_train)
Yhat_linear_test_prob = mod_linear.predict_proba(X_test)
Yhat_linear_train_prob = mod_linear.predict_proba(X_train)
eval_perform_multi_class(Y_test,Yhat_linear_test,Yhat_linear_test_prob,CLASS_NAMES,"Prueba SVM kernel lineal características blancas y negras")
eval_perform_multi_class(Y_train,Yhat_linear_train,Yhat_linear_train_prob,CLASS_NAMES, "Entrenamiento SVM kernel lineal características blancas y negras")

# Entrenamiento del kernel polinomial
mod_poly.fit(X_train, Y_train)
# Evaluación
Yhat_poly_test = mod_poly.predict(X_test)
Yhat_poly_train = mod_poly.predict(X_train)
Yhat_poly_test_prob = mod_poly.predict_proba(X_test)
Yhat_poly_train_prob = mod_poly.predict_proba(X_train)
eval_perform_multi_class(Y_test,Yhat_poly_test, Yhat_poly_test_prob,CLASS_NAMES, "Prueba SVM kernel polinomial características blancas y negras")
eval_perform_multi_class(Y_train,Yhat_poly_train, Yhat_poly_train_prob,CLASS_NAMES, "Entrenamiento SVM kernel polinomial características blancas y negras")

# Entrenamiento del kernel rbf
mod_rbf.fit(X_train, Y_train)
# Evaluación
Yhat_rbf_test = mod_rbf.predict(X_test)
Yhat_rbf_train = mod_rbf.predict(X_train)
Yhat_rbf_test_prob = mod_rbf.predict_proba(X_test)
Yhat_rbf_train_prob = mod_rbf.predict_proba(X_train)
eval_perform_multi_class(Y_test,Yhat_rbf_test, Yhat_rbf_test_prob,CLASS_NAMES,"Prueba SVM kernel rbf características blancas y negras")
eval_perform_multi_class(Y_train,Yhat_rbf_train, Yhat_rbf_train_prob,CLASS_NAMES,"Entrenamiento SVM kernel rbf características blancas y negras")

# Xgboost
import xgboost as xgb
# Crear un clasificador XGBoost
xgboost = xgb.XGBClassifier()
# Entrenar el modelo en los datos de entrenamiento
xgboost.fit(X_train, Y_train)
# Evaluación
Yhat_xgboost_test = xgboost.predict(X_test)
Yhat_xgboost_train = xgboost.predict(X_train)
Yhat_xgboost_test_prob = xgboost.predict_proba(X_test)
Yhat_xgboost_train_prob = xgboost.predict_proba(X_train)
eval_perform_multi_class(Y_test,Yhat_xgboost_test, Yhat_xgboost_test_prob,CLASS_NAMES,"Prueba Xgboost características blancas y negras")
eval_perform_multi_class(Y_train,Yhat_xgboost_train, Yhat_xgboost_train_prob,CLASS_NAMES,"Entrenamiento Xgboost características blancas y negras")

#LightGBM
from lightgbm import LGBMClassifier
lgbm = LGBMClassifier()
lgbm.fit(X_train, Y_train)
# Evaluación
Yhat_lgbm_test = lgbm.predict(X_test)
Yhat_lgbm_train = lgbm.predict(X_train)
Yhat_lgbm_test_prob = lgbm.predict_proba(X_test)
Yhat_lgbm_train_prob = lgbm.predict_proba(X_train)
eval_perform_multi_class(Y_test,Yhat_lgbm_test,Yhat_lgbm_test_prob, CLASS_NAMES, "Prueba LightGBM características blancas y negras")
eval_perform_multi_class(Y_train,Yhat_lgbm_train,Yhat_lgbm_train_prob, CLASS_NAMES,"Entrenamiento LightGBM características blancas y negras")
