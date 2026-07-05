import pandas as pd
import numpy as np
from siuba import select, _
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression

ames = pd.read_csv("data/ames.csv")
print("Tamaño de conjunto completo: ", ames.shape)

Y = select(ames, _.Sale_Price)
Y = select(ames, "Sale_Price")
X = select(ames, -_.Sale_Price)
#X = select(ames, -"Sale_Price") # No Funciona

X_train, X_test, y_train, y_test = train_test_split(
 X, Y, 
 #test_size = 0.20, 
 train_size = 0.80, 
 random_state = 54321
 )

print("Tamaño de conjunto de entrenamiento: ", X_train.shape)
print("Tamaño de conjunto de prueba: ", X_test.shape)


################################################################################

numeric_column = ames["Sale_Price"]
quartiles = np.percentile(numeric_column, [25, 50, 75])

# Crea una nueva variable categórica basada en los cuartiles
stratify_variable = pd.cut(
 numeric_column, 
 bins=[float('-inf'), quartiles[0], quartiles[1], quartiles[2], float('inf')],
 labels=["Q1", "Q2", "Q3", "Q4"]
 )

X_train, X_test, y_train, y_test = train_test_split(
 X, Y, 
 test_size = 0.20, 
 random_state = 12345, 
 stratify = stratify_variable
 )
 

################################################################################

# Dividir los datos en entrenamiento (60%) y el resto (40%)
X_train, X_temp, y_train, y_temp = train_test_split(
 X, Y, 
 test_size = 0.4, 
 random_state = 12345
 )

# Dividir el resto en conjuntos de prueba (15%) y validación (25%)
X_test, X_val, y_test, y_val = train_test_split(
 X_temp, y_temp, 
 test_size = 0.625, 
 random_state = 42
 )

# Training (60%), validation (25%), testing (15%)

# Imprimir los tamaños de los conjuntos resultantes
print("Tamaño de conjunto de entrenamiento: ", X_train.shape)
print("Tamaño de conjunto de prueba: ", X_test.shape)
print("Tamaño de conjunto de validación: ", X_val.shape)


# Dividir los datos en entrenamiento (60%) y el resto (40%)
X_temp, X_test, y_temp, y_test = train_test_split(
 X, Y, 
 train_size= 0.8, 
 random_state = 12345
 )

# Dividir el resto en conjuntos de prueba (15%) y validación (25%)
X_train, X_val, y_train, y_val = train_test_split(
 X_temp, y_temp, 
 train_size = 0.75, 
 random_state = 42
 )






################################################################################
# KFCV

y = ames["Sale_Price"]
X = select(ames, _.Gr_Liv_Area)

# Crea el objeto K-Fold Cross-Validation con K=5 (puedes cambiar el valor de K según tus necesidades)
kf = KFold(n_splits = 20, shuffle = True, random_state = 42)

# Crea el regresor lineal que deseas evaluar
regressor = LinearRegression()

# Realiza la validación cruzada KFCV y obtén los scores de cada iteración
scores = cross_val_score(
 estimator = regressor, 
 X = X, 
 y = y, 
 cv = kf, 
 scoring='neg_mean_squared_error'
 )

# Calcula el promedio y la desviación estándar de los scores
mean_score = -scores.mean()
std_score = scores.std()

# Imprime los resultados
print("Scores de cada iteración:", scores)


np.sort(-scores)








