# Predictor Afluente de Energía
___

## Introducción
Este predictor corresponde a una **red neuronal recurrente** de tipo **GRU** el cual ayuda a que al momento de aprender, debido a sus capas, detiene el desvanicimiento del gradiente, en simples palabras; el modelo es capaz de recordar lo observado en un inicio de la serie de tiempo. Este modelo recibe una secuencia de 24 observaciones y 5 variables, como output sería otra secuencia de 24 observaciones y 1 variable. Las variables corresponden a _afluente sistema enel_, _lluvias_, _coberturas nivales_, _afluente sistema BioBio_, _afluente sistema Maule_. el input son 24 semanas y el output serían las 24 semanas consiguientes de la variable alfuente enel.

## Base de Datos
Los datos de los Afluentes Sistema Enel son obtenidos de tres archivos:
  - `EnelAflu2018b_JA_2000-2010.xlsx`
  - `EnelAflu2018b_JA.xlsx` (datos desde el 2010 hasta el 2019)
  - `EnelAflu2018b.xlsx` (2019-hoy)

solo el último dataset es el que hay que sincronizar. Los dos primeros basta con sincronizarlos una sola vez.

Los datos de las lluvias son obtenidos del dataset `lluvias.xlsx` que a su vez se encuentran actualizados en `Datos_New.xlsm`.

Las coberturas nivales se obtienen de la página [observatorio andino](https://observatorioandino.com/nieve/). Aquí hay que obtener las Series diarias desde el 2000 hasta el 2023 con las unidades de área. Esto para 3 cuencas: Cachapoal, Maule y Ñuble. Guardarlos con los Nombres respectivos; `SCA_Cachapoal.csv`,`SCA_Maule.csv` y `SCA_Ñuble.csv`.
 
Para obtener los afluentes tanto de los sistemas del BioBio y del Maule hay que Utilizar los archivos:
  - `Datos_Hist_v2.xlsx`
  - `Datos_NEW.xlsm`
  - `CaudalRalcoIPLP.xlsx` (éste es para tener los datos de Ralco)

Todos estos datos se encuentran en BBDD. Pero es imnportante ir sincronizarlos semana a semana.

## ¿Cómo utilizar el este modelo?

Lo primero que hay que tener en cuenta es que los archivos previamente mencionados deben estar sincronizados al día. Si se quiere hacer una predicción desde la semana 4 de Abril y estamos en la semana 1 de Abril. El modelo igual hará predicciones pero tomará ceros como valores faltantes, por ende es mejor tener los datos actualizados. 

Sin más preámbulos, el modelo ya está entrenado hasta la semana 4 de febrero, pero se puede volver a entrenar. Para el **reentrenamiento** se debe escribir en su terminal : 

`python main_2.py --mes Abril --semana 2 --año 2023 --epocas 500 --path PATH_DEL_ARCHIVO --path-pesos nombre_de_carpeta_donde_está_inicializado_el modelo/nombre_archivo_pesos.pt`

Considerar `año` como el año hídrico. Con este comando uno le puede dar los parámetros necesarios al modelo escribir el `mes` en minúscula, las `epocas` refieren a la cantidad de interaciones en el entrenamiento, y el resto son los path correspondiente a donde se encuentra el archivo python, donde se guardarán los pesos y con qué nombre guardarlos.

Si se quiere hacer un **predicción**:

`python main_2.py --mes Abril --semana 2 --train False --año 2023 --epocas 500 --path PATH_DEL_ARCHIVO --path-pesos nombre_de_carpeta_donde_está_inicializado_el modelo/nombre_archivo_pesos.pt`

Como resultado de la predicción, se creará un archivo csv con las predicciones de las 24 semanas que suceden a la fecha indicada como input. 
