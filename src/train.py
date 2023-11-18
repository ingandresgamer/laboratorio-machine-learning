from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score,accuracy_score,precision_score


import logging
import sys
import pandas as pd

## Esta Parte se debe ajustar en cada entorno usando el path local de cada desarrollador. 
path_local_dir='/home/andres/Insync/decano.fibem@uan.edu.co/Google Drive/Ing Andres Molano 8 Agosto/Learning/Platzi/Laboratorio de Machine Learning Puesta en Producción de Modelos/laboratorio-machine-learning/'
sys.path.append(path_local_dir) 

import root

logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(name)s: %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S',
    stream=sys.stderr
)

logger= logging.getLogger(__name__)

logger.info("Loading Data..")
# Cargar los datos que tenemos disponibles
data = pd.read_csv(root.DIR_DATA+'/churn.csv')
logger.info("Droping Columns with no information")
# Nos deshacemos de las columnas que no contribuyen en mucho
data = data.drop(data.columns[0:3], axis=1)
"""Ingenieria de Caracteristicas 
Se toman los datos categoricos y se codifican en datos numericos, 
si hay entradas con categorias faltantes se remplaza la categoria faltante 
por la moda y si hay datos numericos faltantes
se remplaza por la media """
logger.info("Categorical Data is Codified...")
column_equivalence = {}
features = list(data.columns)
for i, column in enumerate(list([str(d) for d in data.dtypes])): ### Construye una Lista con los Tipos de Datos del DataFrame
    if column == "object":
        data[data.columns[i]] = data[data.columns[i]].fillna(data[data.columns[i]].mode()) ## Remplaza con la Moda si Hay Faltantes.
        categorical_column = data[data.columns[i]].astype("category") ## Se Crea una columna como categoria 
        current_column_equivalence = dict(enumerate(categorical_column.cat.categories)) ## Se Crea un Diccionario con las Cateogiras
        column_equivalence[i] = dict((v,k) for k,v in current_column_equivalence.items()) ## Se contruye la Codificacion 
        data[data.columns[i]] = categorical_column.cat.codes ## Se Remplaza la Columna por Datos Numericos 
    else:
        data[data.columns[i]] = data[data.columns[i]].fillna(data[data.columns[i]].median())

logger.info('Separating Dataset into train and test')
# Generar los datos para poder separar la variable de respuesta de los datos que tenemos disponibles
X = data.drop('Exited',axis='columns')
y = data['Exited']
# Separar los datos en datos de entrenamiento y testing
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=88)        
logger.info("Creating And Training the Model...")
# Crear el modelo y entrenarlo
clf_lin =  LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
y_pred_test=clf_lin.predict(X_test)
y_pred_train=clf_lin.predict(X_train)
accu_test=accuracy_score(y_test,y_pred_test)
accu_train=accuracy_score(y_train,y_pred_train)
logger.info(f'Accuracy Score in Train:{accu_train:.3f}')
logger.info(f'Accuracy Score in Test:{accu_test:.3f}')
assert accu_test > 0.75
# Generar el binario del modelo para reutilizarlo, equivalencia de variables categoricas y caracteristicas del modelo
import pickle
logger.info(f'Serealizing the Model...')
pickle.dump(clf_lin, open(root.DIR_CHURN+"/models/model.pk", "wb"))
pickle.dump(column_equivalence, open(root.DIR_CHURN+"/models/column_equivalence.pk", "wb"))
pickle.dump(features, open(root.DIR_CHURN+"/models/features.pk", "wb"))


