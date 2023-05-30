import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#from pytorch_forecasting import prepare_data
from models import Afluent_Forecasting_Model_RNN, Procesamiento_DatosEnel, Preparacion_Datos

import os
import itertools
from shutil import rmtree

import configparser 

import warnings
warnings.filterwarnings("ignore")

#########################################################################################
config  = configparser.ConfigParser()
config.read('config.ini')
##########################################################################################
#Igresar fecha de predicción: (modificar) 
MES= config.get('general','mes')
SEMANA= config.getint('general','semana')
AÑO=config.getint('general','año')
AÑO_INICIO = config.getint('general','año_inicio')
####
####
dic_mes={ 'Abril'.lower():1,'Mayo'.lower():2,'Junio'.lower():3,'Julio'.lower():4,'Agosto'.lower():5,'Septiembre'.lower():6,'Octubre'.lower():7,'Noviembre'.lower():8,'Diciembre'.lower():9,'Enero'.lower():10,'Febrero'.lower():11,'Marzo'.lower():12}
NF=config.getint('general','num_feat')
####
TRAIN = config.getboolean('general','train')
EPOCAS = config.getint('general','epocas')
########################################################################################
#DEFINIENDO PATHS 
PATH = config.get('general','path')  

PATH_MODEL = PATH + config.get('general','path_model')    

#Creando un directorio para guardar las matrices creadas.
PATH_MAT = PATH+"BBDD/matrices"

if os.path.exists(PATH_MAT):
    rmtree(PATH_MAT)

if os.path.exists(PATH_MAT)==False:
    os.mkdir(PATH_MAT)
    print("\n### Creando matrices de las variables a considerar como Input ###")

    #CREANDO LAS MATRICES QUE CORRESPONDERAN A LOS FEATURES PARA GENERAR EL INPUT
    #Creando los afluentes:
    prep_enel_afluente = Preparacion_Datos.ProcesamientoDatos1(PATH+"BBDD/",kind="enel afluent",last_month=dic_mes[MES],last_year=AÑO,last_week=SEMANA)
    enel_matx = prep_enel_afluente.crear_matriz_afluente_enel()
    enel_matx.to_csv(PATH_MAT+"/enel_aflu.csv")
    prep_afluentes=Preparacion_Datos.ProcesamientoDatos1(PATH+"BBDD/",kind="extra afluent",last_month=dic_mes[MES],last_year=AÑO,last_week=SEMANA)
    biobio_matx, maule_matx = prep_afluentes.crear_matriz_afluentes()
    biobio_matx.to_csv(PATH_MAT+"/biobio_aflu.csv")
    maule_matx.to_csv(PATH_MAT+"/maule_aflu.csv")

    #Creando lluvias:
    prep_lluvia = Preparacion_Datos.ProcesamientoDatos1(PATH+"BBDD/lluvias.xlsx",kind="lluvia",sheet_name="dat",last_month=dic_mes[MES],last_year=AÑO,last_week=SEMANA)
    lluvia_matx = prep_lluvia.crear_matriz_lluvia()
    lluvia_matx.to_csv(PATH_MAT+"/lluvia.csv")

    #Creando Nieves:
    prep_cob = Preparacion_Datos.ProcesamientoDatos1(PATH+"BBDD/",kind="nieve",last_month=dic_mes[MES],last_year=AÑO,last_week=SEMANA)
    cob_matx = prep_cob.crear_matrix_nieve()
    cob_matx.to_csv(PATH_MAT+"/cobertura_nival.csv")

#Creando una lista con los nombres de los paths de las matrices.
paths = []
#paths.append(os.path.join(PATH,"BBDD/Caudales.xlsx"))
paths.append(os.path.join(PATH_MAT,"enel_aflu.csv"))
for f in itertools.cycle(os.listdir(PATH_MAT)):
    if len(paths)==1 and "lluvia" in f :
        paths.append(os.path.join(PATH_MAT,f))
    elif len(paths)==2 and "cobertura" in f:
        paths.append(os.path.join(PATH_MAT,f))
    elif len(paths)==3 and "biobio" in f:
        paths.append(os.path.join(PATH_MAT,f))
    elif len(paths)==4 and "maule" in f:
        paths.append(os.path.join(PATH_MAT,f))
    if len(paths)==NF:
        break
#########################################

##########################################

#HACIENDO EL PROCESAMIENTO DE LAS MATRICES Y PREPARANDO TENSORES
print("\n### Preparando las matrices para el modelo ###")
preprocesamiento = Procesamiento_DatosEnel.ProcesamientoDatos(num_feat=NF,
                                                              num_prediction=24,
                                                              train = TRAIN,
                                                              month=MES,
                                                              week_limit=SEMANA,
                                                              year=AÑO,
                                                              year_init=AÑO_INICIO                                  
                                                           )
DEVICE = config.get('general','device')
if TRAIN:
    dl_, norm_params = preprocesamiento.procesando_datos(paths)
    afm = Afluent_Forecasting_Model_RNN.AfluentEnelForecasting(Train=TRAIN,
                                                       	       path_model_to_load=None,
                                                               path_model_to_save=PATH_MODEL,
                                                               device=DEVICE
                                                               )
    #HACIENDO EL ENTRENAMIENTO
    afm.training(dl_,EPOCAS)
    
elif TRAIN == False: 
    dl_, out ,norm_params = preprocesamiento.procesando_datos(paths)
    afm = Afluent_Forecasting_Model_RNN.AfluentEnelForecasting(Train=TRAIN,
                                                               path_model_to_load=PATH_MODEL,
                                                               path_model_to_save=None,
                                                               device=DEVICE
                                                               )
    #HACIENDO LA PREDICCIÓN                                                           
    inp,prediction = afm.prediction(dl_)

    #PREPARANDO EL OUTPUT
    preprocesamiento.parametros_norm(norm_params)

    in_unnorm, out_unnorm, real_unnorm = preprocesamiento.desnormalizacion(
        x_norm1=inp[:,:,0],
        x_norm2=prediction,
        real = out[:,:,0]
        )
    
    dic_enel_mes = {"abril":1,"mayo":2,"junio":3,"julio":4,"agosto":5,"septiembre":6,"octubre":7,"noviembre":8,"diciembre":9,"enero":10,"febrero":11,"marzo":12}
    if dic_enel_mes[MES.lower()]>7:
        año = str(AÑO)+"-"+str(AÑO+1)
    elif dic_enel_mes[MES.lower()]<7:
        año = str(AÑO-1)+"-"+str(AÑO)
    else :
        año = str(AÑO)
    #preparando datos para crear el archivo de la predicción.
    df = pd.DataFrame(data=out_unnorm,columns=preprocesamiento.week_columns[24:48])
    df_path = PATH+"prediccion_afluente_energía_"+MES+"_"+str(SEMANA)+"_"+año+"_"+str(AÑO_INICIO)+".csv"
    df.to_csv(df_path,sep=',')
    print("\n### las predicciones fueron guardadas en",df_path,"###")
    if real_unnorm.shape[1] < 24:
        real_unnorm = np.pad(real_unnorm,([0,0],[0,np.abs(real_unnorm.shape[1]-24)]),mode='constant',constant_values=0)
    




