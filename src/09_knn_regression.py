from mlxtend.feature_selection import ColumnSelector
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle

from plydata.one_table_verbs import pull
from mizani.formatters import comma_format, dollar_format
from plotnine import *
from siuba import *

import pandas as pd
import numpy as np


#### CARGA DE DATOS ####
ames = pd.read_csv("data/ames.csv")

ames_y = ames >> pull("Sale_Price")    # ames[["Sale_Price"]]
ames_x = select(ames, -_.Sale_Price)   # ames.drop('Sale_Price', axis=1)

#### DIVISIÓN DE DATOS ####
ames_x_train, ames_x_test, ames_y_train, ames_y_test = train_test_split(
 ames_x, ames_y, 
 test_size = 0.20, 
 random_state = 195
 )


#### FEATURE ENGINEERING ####

## SELECCIÓN DE VARIABLES

# Seleccionamos las variales numéricas de interés
num_cols = ["Full_Bath", "Half_Bath"]

# Seleccionamos las variables categóricas de interés
cat_cols = ["Overall_Cond"]

# Juntamos todas las variables de interés
columnas_seleccionadas = num_cols + cat_cols

pipe = ColumnSelector(columnas_seleccionadas)
ames_x_train_selected = pipe.fit_transform(ames_x_train)

ames_train_selected = pd.DataFrame(
  ames_x_train_selected, 
  columns = columnas_seleccionadas
  )

ames_train_selected.info()


## TRANSFORMACIÓN DE COLUMNAS

# ColumnTransformer para aplicar transformaciones
preprocessor = ColumnTransformer(
    transformers = [
        ('scaler', StandardScaler(), num_cols),
        ('onehotencoding', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
    ],
    verbose_feature_names_out = False,
    remainder = 'passthrough'  # Mantener las columnas restantes sin cambios
).set_output(transform ='pandas')

transformed_df = preprocessor.fit_transform(ames_train_selected)
#new_column_names = preprocessor.get_feature_names_out()

# transformed_df = pd.DataFrame(
#   transformed_data,
#   columns=new_column_names
#   )

transformed_df
transformed_df.info()


#### PIPELINE Y MODELADO

# Crear el pipeline con la regresión lineal
pipeline = Pipeline([
   ('preprocessor', preprocessor),
   ('regressor', KNeighborsRegressor(n_neighbors=5))
])

# Entrenar el pipeline
results = pipeline.fit(ames_train_selected, ames_y_train)
pipeline.predict(ames_train_selected)

## PREDICCIONES
y_pred = pipeline.predict(ames_x_test)

ames_test = (
  ames_x_test >>
  mutate(Sale_Price_Pred = y_pred, Sale_Price = ames_y_test)
)

ames_test.info()

(
ames_test >>
  select(_.Sale_Price, _.Sale_Price_Pred)
)


##### Métricas de desempeño

pd.options.display.float_format = '{:.2f}'.format

y_obs = ames_test["Sale_Price"]
y_pred = ames_test["Sale_Price_Pred"]

me = np.mean(y_obs - y_pred)
mae = mean_absolute_error(y_obs, y_pred)
mape = mean_absolute_percentage_error(y_obs, y_pred)
mse = mean_squared_error(y_obs, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_obs, y_pred)

n = len(y_obs)  # Número de observaciones
p = 9  # Número de predictores 
r2_adj = 1 - (n - 1) / (n - p - 1) * (1 - r2)

metrics_data = {
    "Metric": ["ME", "MAE", "MAPE", "MSE", "RMSE", "R^2", "R^2 Adj"],
    "Value": [me, mae, mape, mse, rmse, r2, r2_adj]
}

metrics_df = pd.DataFrame(metrics_data)
metrics_df

#### Gráficos de desempeño de modelo

(
  ames_test >>
    ggplot(aes(y = "Sale_Price", x = "Sale_Price_Pred")) +
    geom_point() +
    scale_y_continuous(labels = dollar_format(digits=0, big_mark=','), limits = [0, 600000] ) +
    scale_x_continuous(labels = dollar_format(digits=0, big_mark=','), limits = [0, 500000] ) +
    geom_abline(color = "red") +
    coord_equal() +
    labs(
      title = "Comparación entre predicción y observación",
      y = "Predicción",
      x = "Observación")
)


(
ames_test >>
  select(_.Sale_Price, _.Sale_Price_Pred) >>
  mutate(error = _.Sale_Price - _.Sale_Price_Pred) >>
  ggplot(aes(x = "error")) +
  geom_histogram(color = "white", fill = "black") +
  geom_vline(xintercept = 0, color = "red") +
  scale_x_continuous(labels=dollar_format(big_mark=',', digits=0)) + 
  ylab("Conteos de clase") + xlab("Errores") +
  ggtitle("Distribución de error")
)


(
ames_test >>
  select(_.Sale_Price, _.Sale_Price_Pred) >>
  mutate(error = _.Sale_Price - _.Sale_Price_Pred) >>
  ggplot(aes(sample = "error")) +
  geom_qq(alpha = 0.3) + stat_qq_line(color = "red") +
  scale_y_continuous(labels=dollar_format(big_mark=',', digits = 0)) + 
  xlab("Distribución normal") + ylab("Distribución de errores") +
  ggtitle("QQ-Plot")
)


(
ames_test >>
  select(_.Sale_Price, _.Sale_Price_Pred) >>
  mutate(error = _.Sale_Price - _.Sale_Price_Pred) >>
  ggplot(aes(x = "Sale_Price")) +
  geom_linerange(aes(ymin = 0, ymax = "error"), colour = "purple") +
  geom_point(aes(y = "error"), size = 0.05, alpha = 0.5) +
  geom_abline(intercept = 0, slope = 0) +
  scale_x_continuous(labels=dollar_format(big_mark=',', digits=0)) + 
  scale_y_continuous(labels=dollar_format(big_mark=',', digits=0)) +
  xlab("Precio real") + ylab("Error de estimación") +
  ggtitle("Relación entre error y precio de venta")
)

############################
#### Validación cruzada ####
############################


# Definir el objeto K-Fold Cross Validator
k = 10
kf = KFold(n_splits=k, shuffle=True, random_state=42)

param_grid = {
 'n_neighbors': range(2, 21),
 'weights': ['uniform', 'distance'],
 'metric': ['euclidean', 'manhattan']
 #'p': [1, 2]
}

# Algunas otras posibles distancias son:
# ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'hamming', 'jaccard', 'cosine']

# Definir las métricas de desempeño que deseas calcular como funciones de puntuación

def adjusted_r2_score(y_true, y_pred, n, p):
  r2 = r2_score(y_true, y_pred)
  adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
  return adjusted_r2

scoring = {
    'neg_mean_squared_error': make_scorer(mean_squared_error, greater_is_better=False),
    'r2': make_scorer(adjusted_r2_score, 
                      greater_is_better=True, 
                      n=np.ceil(len(ames_train_selected)), 
                      p=len(ames_train_selected.columns)),
    'neg_mean_absolute_error': make_scorer(mean_absolute_error, greater_is_better=False),
    'mape': make_scorer(mean_absolute_percentage_error, greater_is_better=False)
}

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', GridSearchCV(
      KNeighborsRegressor(), 
      param_grid, 
      cv=kf, 
      scoring=scoring, 
      refit='neg_mean_squared_error',
      verbose=3, 
      n_jobs=7)
     )
])

pipeline.fit(ames_train_selected, ames_y_train)

results_cv = pipeline.named_steps['regressor'].cv_results_

# Convierte los resultados en un DataFrame
pd.set_option('display.max_columns', 500)
results_df = pd.DataFrame(results_cv)
results_df.columns

# Puedes seleccionar las columnas de interés, por ejemplo:

summary_df = (
  results_df >>
  select(-_.contains("split"), -_.contains("time"), -_.params)
)
summary_df

(
  summary_df >>
  ggplot(aes(x = "param_n_neighbors", y = "mean_test_r2", #size = "param_p",
             color = "param_metric", shape = "param_weights")) +
  geom_point(alpha = 0.65) +
  ggtitle("Parametrización de KNN vs R^2") +
  xlab("Parámetro: Número de vecinos cercanos") +
  ylab("R^2 promedio")
)

(
  summary_df >>
  filter(
    _.param_weights == "uniform",
    #_.param_p == 1,
    _.param_metric == "manhattan") >>
  mutate(
    ymin = np.maximum(0, _.mean_test_r2 - _.std_test_r2),
    ymax = np.minimum(1, _.mean_test_r2 + _.std_test_r2)) >>
  ggplot(aes(x = "param_n_neighbors", y = "mean_test_r2")) +
  geom_errorbar(aes(ymin='ymin', ymax='ymax'),
    width=0.3, position=position_dodge(0.9)) +
  geom_point(alpha = 0.65) +
  ggtitle("Parametrización de KNN vs R^2") +
  xlab("Parámetro: Número de vecinos cercanos") +
  ylab("R^2 promedio")
)


best_params = pipeline.named_steps['regressor'].best_params_
best_params
best_estimator = pipeline.named_steps['regressor'].best_estimator_
best_estimator


## PREDICCIONES FINALES

final_knn_pipeline = Pipeline([
   ('preprocessor', preprocessor),
   ('regressor', best_estimator)
])

# Entrenar el pipeline
final_knn_pipeline.fit(ames_train_selected, ames_y_train)

## Predicciones finales
y_pred_knn = final_knn_pipeline.predict(ames_x_test)

results_reg = (
  ames_x_test >>
  mutate(final_knn_pred = y_pred_knn, Sale_Price = ames_y_test) >>
  select(_.Sale_Price, _.final_knn_pred)
)
results_reg

## Métricas de desempeño

me = np.mean(y_obs - y_pred_knn)
mae = mean_absolute_error(y_obs, y_pred_knn)
mape = mean_absolute_percentage_error(y_obs, y_pred_knn)
mse = mean_squared_error(y_obs, y_pred_knn)
rmse = np.sqrt(mse)
r2 = r2_score(y_obs, y_pred_knn)
r2_adj = adjusted_r2_score(y_true = y_obs, y_pred = y_pred_knn,
  n=np.ceil(len(ames_train_selected)), p=len(ames_train_selected.columns))

metrics_data = {
    "Metric": ["ME", "MAE", "MAPE", "MSE", "RMSE", "R^2", "R^2 Adj"],
    "Value": [me, mae, mape, mse, rmse, r2, r2_adj]
}

metrics_df = pd.DataFrame(metrics_data)
metrics_df

(
  results_reg >>
    ggplot(aes(x = "final_knn_pred", y = "Sale_Price")) +
    geom_point() +
    geom_abline(color = "red") +
    xlab("Prediction") +
    ylab("Observation") +
    ggtitle("Comparisson")
)


#### Importancia de variables


# 1 permutación

np.zeros(3)
importance = np.zeros(ames_x_test[columnas_seleccionadas].shape[1])

# Realiza el procedimiento de permutación
for i in range(ames_x_test[columnas_seleccionadas].shape[1]):
    ames_x_test_permuted = ames_x_test[columnas_seleccionadas].copy()
    ames_x_test_permuted.iloc[:, i] = shuffle(ames_x_test_permuted.iloc[:, i], random_state=42)  
    # Permuta una característica
    y_pred_permuted = final_knn_pipeline.predict(ames_x_test_permuted)
    mse_permuted = mean_squared_error(ames_y_test, y_pred_permuted)
    importance[i] = mse_permuted - mse

# Calcula la importancia relativa
importance = importance / importance.sum()
importance

importance_df = pd.DataFrame({
  'Variable': columnas_seleccionadas, 
  'Importance': importance
  })

# Crea la gráfica de barras
(
  importance_df >>
  ggplot(aes(x= 'reorder(Variable, Importance)', y='Importance')) + 
  geom_bar(stat='identity', fill='blue', color = "black") + 
  labs(title='Importancia de las Variables', x='Variable', y='Importancia') +
  coord_flip()
)


# N permutaciones

n_permutations = 50
performance_losses = []

for i in range(ames_x_test[columnas_seleccionadas].shape[1]):
    loss = []
    for j in range(n_permutations):
        ames_x_test_permuted = ames_x_test[columnas_seleccionadas].copy()
        ames_x_test_permuted.iloc[:, i] = np.random.permutation(ames_x_test_permuted.iloc[:, i])
        y_pred_permuted = final_knn_pipeline.predict(ames_x_test_permuted)
        mse_permuted = mean_squared_error(ames_y_test, y_pred_permuted)
        loss.append(mse_permuted)
    performance_losses.append(loss)

performance_losses = performance_losses/np.sum(performance_losses, axis=0)
mean_losses = np.mean(performance_losses, axis=1)
std_losses = np.std(performance_losses, axis=1)

importance_df = pd.DataFrame({
  'Variable': columnas_seleccionadas, 
  'Mean_Loss': mean_losses, 
  'Std_Loss': std_losses
  })

(
  importance_df >>
  mutate(
    ymin = _.Mean_Loss - _.Std_Loss,
    ymax = _.Mean_Loss + _.Std_Loss) >>
  ggplot(aes(x = 'reorder(Variable, Mean_Loss)', y = "Mean_Loss")) +
  geom_errorbar(aes(ymin='ymin', ymax='ymax'),
    width=0.2, position=position_dodge(0.9)) +
  geom_point(alpha = 0.65) +
  labs(title='Importancia de las Variables', x='Variable', y='Importancia') +
  coord_flip()
)










