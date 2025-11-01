from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import pandas as pd

# Pipeline(steps, memory, verbose, .....):
#     paso 1: z= par1 + par2
#     paso2: w = z / par3
#     return w 
# 
# x + y


Pipeline()


# par1: "texto" str
# par2: 10  int
# par3: 15.37   float
# par4: [1, 2, 3, 4, 5] list
# par5: [(clave1, valor1), (clave2, valor2), (clave3, valor3)] list
# par6: [[clave1, valor1], [clave2, valor2], [clave3, valor3]] list

[
    ('standardscaler', StandardScaler()), 
    ('pca', PCA(n_components=2)), 
    ('svm', SVC())
    ]


pipeline1 = Pipeline(steps = [
    ('standardscaler', StandardScaler()),
    ('pca', PCA(n_components=2)),
    ('svm', SVC())
])
pipeline1


pipeline1 = Pipeline([
    ('paso1', StandardScaler()),
    ('paso2', PCA(n_components=2)),
    ('paso3', SVC())
])
pipeline1


pipeline2 = make_pipeline(
    StandardScaler(),
    PCA(n_components=2),
    SVC()
)
pipeline2

######### NORMALIZAR ############

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import numpy as np

X_train = np.array([[ 1., -1.,  5.],
                    [ 0.,  0., 10.],
                    [ 1., -1.,  5.],
                    [ 0.,  0., 10.],
                    [ 1., -1.,  5.],
                    [ 0.,  0., 10.],
                    [ 1., -1.,  5.],
                    [ 0.,  0., 10]])

X_train

scaler = StandardScaler().fit(X_train)

scaler.mean_
scaler.scale_

X_scaled = scaler.transform(X_train)
X_scaled

X_scaled.mean(axis=0)
X_scaled.std(axis=0)

### NUEVOS DATOS

X_test = np.array([[ 0., 1.,  1.],
                   [ 2., 0.,  1.],
                   [ 0., 1.,  1.]])

X_test

X_test_scaled = scaler.transform(X_test)
X_test_scaled

######### DICOTOMIZAR ############

from sklearn.preprocessing import OneHotEncoder

# Datos de entrenamiento

X_train_cat = pd.DataFrame({
 "Cat1": ['A', 'B', 'A', 'C', 'B', 'B'],
 "Cat2": ['Z2', 'X2', 'W2', 'W2', 'X2', 'Z2']
})
X_train_cat

# Crear una instancia de OneHotEncoder
encoder1 = OneHotEncoder(
 drop=None, 
 handle_unknown='ignore', 
 sparse_output=False
 ).set_output(transform ='pandas')

encoder2 = OneHotEncoder(
 drop='first', 
 handle_unknown='ignore', 
 sparse_output=False
 ).set_output(transform ='pandas')


# Ajustar y transformar el encoder en los datos de entrenamiento
X_train_encoded = encoder2.fit_transform(X_train_cat)
X_train_encoded

# Datos de prueba
X_test_cat = pd.DataFrame({
 "Cat1": ['A', 'C', 'B', 'D'],
 "Cat2": ['X2', 'Z2', 'Z2', 'W2']
})

# Transformar los datos de prueba utilizando el encoder ajustado
X_test_encoded = X_train_encoded.transform(X_test_cat)
X_test_encoded

######### IMPUTACIÓN ############

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Caso numérico:

# Datos de ejemplo con valores faltantes
data = np.array([[1, 1, 30, np.nan, 1],
                 [2, 2, np.nan, 100, 3],
                 [1, 5, 6, 15, np.nan],
                 [2, 5, 6, 15, 8],
                 [1, 5, 6, 15, 9],
                 [2, 3, 6, 50, -7],
                 [np.nan, 4, 19, 0, 5]])

data

# Definir las estrategias de imputación

imputer_mean = SimpleImputer(strategy='mean')  # Imputación con la media
imputer_median = SimpleImputer(strategy='median')  # Imputación con la mediana
imputer_mode = SimpleImputer(strategy='most_frequent')  # Imputación con la moda
imputer_arbitrary = SimpleImputer(strategy='constant', fill_value=0)  # Imputación constante
imputer_knn2 = KNNImputer(n_neighbors=2)

# Crear un ColumnTransformer para aplicar diferentes estrategias a diferentes columnas
column_transformer2 = ColumnTransformer(
 transformers=[
    ('KNN_imputer', imputer_knn2, [0]),
    ('median_imputer', imputer_median, [2]),
    ('mode_imputer', imputer_mode, [3]),
    ('arbitrary_imputer', imputer_arbitrary, [4])
]).set_output(transform ='pandas')

# Crear el pipeline con el ColumnTransformer
pipeline2 = Pipeline(steps=[
    ('column_transformer', column_transformer2)
])

# Aplicar el pipeline a los datos
pd.set_option('display.max_columns', 5)

pipeline2.fit(data)
pipeline2.transform(data)

imputed_data = pipeline2.fit_transform(data)
imputed_data

pipeline2.transform(new_data)



params = pipeline2.fit(data)
imputer = params.named_steps['column_transformer'].named_transformers_['median_imputer']
imputer.statistics_

imputed_data = pipeline.transform(data)
imputed_data

### Otro ejemplo de imputación basado en KNN
X = np.array([[1, 2, np.nan], [3, 4, 3], [np.nan, 6, 5], [8, 8, 7]])
imputer = KNNImputer(n_neighbors=2)
imputer.fit_transform(X)

imputer = params.named_steps['column_transformer'].named_transformers_['KNN_imputer']
imputer.weights


# Caso categórico:

# Datos de ejemplo con valores faltantes
data = pd.DataFrame({
    'Columna1': ['A', 'A', 'A', np.nan, 'B'],
    'Columna2': ['X', np.nan, 'Y', 'Z', 'Z']
})
data

# Definir las estrategias de imputación
imputer_most_frequent = SimpleImputer(strategy='most_frequent')
imputer_constant = SimpleImputer(strategy='constant', fill_value='missing')

# Crear un ColumnTransformer para aplicar diferentes estrategias a diferentes columnas
column_transformer = ColumnTransformer(transformers=[
    ('most_frequent_imputer', imputer_most_frequent, ['Columna1']),  
    ('constant_imputer', imputer_constant, ['Columna2']) 
]).set_output(transform ='pandas')

# Aplicar el ColumnTransformer a los datos
imputed_data = column_transformer.fit_transform(data)
imputed_data

######### INTERACCIONES ############

from plotnine import *
from mizani.formatters import comma_format, dollar_format
from sklearn.preprocessing import PolynomialFeatures

(
ggplot(ames, aes(x = "Gr_Liv_Area", y = "Sale_Price") ) + 
  geom_point(alpha = .2) +
  facet_wrap("Bldg_Type") + 
  geom_smooth(method = "lm", se = False, color = "red", alpha = 0.1)  + 
  scale_x_log10(labels = comma_format()) + 
  scale_y_log10(labels = dollar_format(prefix='$', digits=0, big_mark=',')) + 
  labs(
   title = "Relación entre precio y tamaño con tipo de vivienda",
   x = "Gross Living Area", 
   y = "Sale Price (USD)")
)


# Ejemplo de datos
data = pd.DataFrame({
    'C1': [1, 0,  1, 0, -1, 2,  -2, 5],
    'C2': [1, 5, -5, 3, -1, 0.5, 1, 10]
})

# Crear interacciones polinómicas
interaction_transformer = PolynomialFeatures(
 degree = 2, 
 interaction_only = True, 
 include_bias = False
 )

# ColumnTransformer para aplicar transformaciones
preprocessor = ColumnTransformer(
    transformers=[
        ('interactions', interaction_transformer, ['C1', 'C2'])
    ],
    remainder='passthrough'  # Mantener las columnas restantes sin cambios
).set_output(transform ='pandas')

# Ajustar el pipeline y transformar a los datos
preprocessor.fit_transform(data)

test = pd.DataFrame({
    'C1': [1,  2, 3, -4, 5],
    'C2': [5, -4, 3, -2, 1]
})

# aplicar en el conjunto de prueba
preprocessor.transform(test)


######### RENOMBRAMIENTO DE DATOS ############

data = pd.DataFrame({
 'edad': [20, 30, 40], 
 'ingreso': [50000, 60000, 70000],
 'sexo': ['M', 'F', 'M'],
 'educacion': ['secundaria', 'universidad', 'preparatoria']
 })

# ColumnTransformer para características numéricas y categóricas
ct = ColumnTransformer(
 transformers=[ 
  ('num', StandardScaler(), ['edad', 'ingreso']),
  ('cat', OneHotEncoder(drop='first'), ['sexo', 'educacion'])],
  verbose_feature_names_out = False
  )

# Ajuste y transformación en data
transformed_data = ct.fit_transform(data)

# Abtener los nombres de las características de salida del transformador
new_column_names = ct.get_feature_names_out()
new_column_names
# Crear un DataFrame con los datos transformados y los nuevos nombres de las columnas
transformed_df = pd.DataFrame(transformed_data, columns=new_column_names)

# Imprimir el DataFrame resultante
print(transformed_df)







