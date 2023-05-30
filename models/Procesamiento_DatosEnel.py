
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import itertools


#from pytorch_forecasting import prepare_data
import torch
from torch.utils.data import DataLoader, TensorDataset

class ProcesamientoDatos():
    def __init__(self,
                 num_feat=6,
                 num_prediction=25,
                 train=False,
                 normalization="standard",
                 shift=1,
                 month = "abril",
                 week_limit=1,
                 year=2022,
                 year_init=2000
                ):
        self.nf = num_feat
        self.np = num_prediction
        self.train = train
        self.norm = normalization
        self.shift = shift

        self.month = month.lower()
        assert week_limit >=1 and week_limit<=4, "Se debe señalar la semana del mes en cual quiere iniciar la semana, este tiene que estar dentro del rango de entero entre 1 y 4."
        self.week = str(week_limit)
        assert year>=2022 and (year<year+2 ), "Ingresar años disponibles 2022 y "+str(year+1)+", ya que el modelo se entrenó hasta los datos del 2021."
        self.year = year
        self.year_init = year_init

        self.fecha_inicio = self.month+"/"+self.week

        self.weeks =  ["abril/1", "abril/2", "abril/3", "abril/4","mayo/1","mayo/2","mayo/3","mayo/4","junio/1", "junio/2","junio/3","junio/4",
                    "julio/1","julio/2","julio/3","julio/4","agosto/1", "agosto/2","agosto/3","agosto/4","septiembre/1","septiembre/2","septiembre/3",
                    "septiembre/4", "octubre/1", "octubre/2","octubre/3","octubre/4", "noviembre/1","noviembre/2","noviembre/3","noviembre/4","diciembre/1",
                    "diciembre/2","diciembre/3","diciembre/4","enero/1","enero/2", "enero/3","enero/4","febrero/1","febrero/2","febrero/3","febrero/4",
                    "marzo/1","marzo/2","marzo/3","marzo/4" ]


        
    
    def normalizacion(self,x): 
        if self.norm == "standard":
            norm = StandardScaler()
            x_norm = norm.fit_transform(x)
            self.sumando = norm.mean_
            self.factor = norm.scale_
            
        elif self.norm == "minmax":
            norm = MinMaxScaler()
            x_norm = norm.fit_transform(x)
            self.sumando = norm.data_min_
            self.factor = norm.data_range_
            
        else:
            x_norm = x
            
        return x_norm
    
    def inverse_norm_func(self,x,mean,std):
        '''
        Este método transformará cada variable del del arreglo o variable x dada su misma media y desviación 
        estandar.
        '''
        raw_x = x*std + mean
        return np.array(raw_x)
    
    def shifted_matrix2(self,x,l=48,ws=24,lo=24,s=1,cat_samples=False,target=None,zero_index=None):
        x = x[:-1,:,:]
        s1,s2,s3 = x.shape
        #Flatten
        flatt_x_2D = x.reshape(s1*s2,s3) #[bs * seq, feat]
        x_new = flatt_x_2D[:len(zero_index),:] #[bs * seq2 , feat]; seq > seq2
        nw = int((x_new.shape[0] - ws - lo + s)/s)
        new_matrix = np.array([x_new[j*s:j*s+ws,:] for j in range(nw)])
        if target is not None:
            y = np.array([x_new[j*s+ws:j*s+ws+lo,:] for j in range(nw)])
            if cat_samples:
                new_matrix = new_matrix.reshape(-1,ws,s3)
                y = y.reshape(-1,lo,s3)
            return new_matrix, y#.reshape(-1,ws,s3)
        elif target is None:
            if cat_samples:
                new_matrix = new_matrix.reshape(-1,ws,s3)
            return new_matrix

    
    def parametros_norm(self,parametros):
        """
        Este método tiene que servir para preparar un vector de medias y desviaciones para la desnormalización.

        Parámetro "fecha de inicio" tiene que venir en formato mes/semana (1-4)
        """
        #Aquí definimos los diccionarios para guardar los parámetros de normalización
        params_mean = {"abril/1":0, "abril/2":0, "abril/3":0, "abril/4":0,"mayo/1":0,"mayo/2":0,"mayo/3":0,"mayo/4":0,"junio/1":0, "junio/2":0,"junio/3":0,"junio/4":0,
        "julio/1":0,"julio/2":0,"julio/3":0,"julio/4":0,"agosto/1":0, "agosto/2":0,"agosto/3":0,"agosto/4":0,"septiembre/1":0,"septiembre/2":0,"septiembre/3":0,
        "septiembre/4":0, "octubre/1":0, "octubre/2":0,"octubre/3":0,"octubre/4":0, "noviembre/1":0,"noviembre/2":0,"noviembre/3":0,"noviembre/4":0,"diciembre/1":0,
        "diciembre/2":0,"diciembre/3":0,"diciembre/4":0,"enero/1":0,"enero/2":0, "enero/3":0,"enero/4":0,"febrero/1":0,"febrero/2":0,"febrero/3":0,"febrero/4":0,
        "marzo/1":0,"marzo/2":0,"marzo/3":0,"marzo/4":0,              
        }

        params_std = {"abril/1":0, "abril/2":0, "abril/3":0, "abril/4":0,"mayo/1":0,"mayo/2":0,"mayo/3":0,"mayo/4":0,"junio/1":0, "junio/2":0,"junio/3":0,"junio/4":0,
        "julio/1":0,"julio/2":0,"julio/3":0,"julio/4":0,"agosto/1":0, "agosto/2":0,"agosto/3":0,"agosto/4":0,"septiembre/1":0,"septiembre/2":0,"septiembre/3":0,
        "septiembre/4":0, "octubre/1":0, "octubre/2":0,"octubre/3":0,"octubre/4":0, "noviembre/1":0,"noviembre/2":0,"noviembre/3":0,"noviembre/4":0,"diciembre/1":0,
        "diciembre/2":0,"diciembre/3":0,"diciembre/4":0,"enero/1":0,"enero/2":0, "enero/3":0,"enero/4":0,"febrero/1":0,"febrero/2":0,"febrero/3":0,"febrero/4":0,
        "marzo/1":0,"marzo/2":0,"marzo/3":0,"marzo/4":0,              
        }
        #completamos el diccionario con los parámetros de normalización.
        for e,k in enumerate(params_mean.keys()):
            params_mean[k] = parametros['means'][0][e]
            params_std[k] = parametros['std'][0][e]
        #generamos los vectores correspondiente
        mean_in, mean_out, std_in, std_out = [],[],[],[]
        

        pivote = self.fecha_inicio
        #for input
        c=0
        for mes in itertools.cycle(reversed(params_mean.keys())):
            if mes == pivote:
                c+=1
            elif mes != pivote and c>0:
                mean_in.append(params_mean[mes])
                std_in.append(params_std[mes])
                c+=1
        
            if c==25:
                break
        
        mean_in = list(reversed(mean_in))
        std_in = list(reversed(std_in))

        #for output
        c=0
        for mes in itertools.cycle(params_mean.keys()):
            if mes == pivote:
                mean_out.append(params_mean[mes])
                std_out.append(params_std[mes])
                c+=1
            elif mes != pivote and c>0:
                mean_out.append(params_mean[mes])
                std_out.append(params_std[mes])
                c+=1
            
            
            if c==24:
                break
        
        self.mean_in = mean_in
        self.mean_out = mean_out
        self.std_in = std_in
        self.std_out = std_out


    #Función para el proceso iterativo de desnormalización
    def desnormalizacion(self,x_norm1,x_norm2,real):
        '''
        Este método va a recibir "x_norm1" que corresponde a los datos reales de entrada e "x_norm2" corresponde
        a las predicciones. Además recibe el parámetro "iteraciones" el cual este indica la posición inicial
        de donde empieza la ventana para procesar.
        
        nw corresponderá al número de ventanas que se tienen que utilizar para la desnormalización. 
        '''
        
        
        if self.norm == "standard":

            if self.train: #arreglar.
                in_unnorm = self.inverse_norm_func(x_norm1,self.mean_in,self.std_in)
                out_unnorm= self.inverse_norm_func(x_norm2,self.mean_out,self.std_out)
                end_real = real.shape[1]
                real_unnorm = self.inverse_norm_func(real,self.mean_out[:end_real],self.std_out[:end_real]) if real is not None else None
            
            else: #Para hacer la predicción.                
                in_unnorm = self.inverse_norm_func(x_norm1,self.mean_in,self.std_in)
                end_real = real.shape[1]
                zero_count = np.abs(np.count_nonzero(in_unnorm[0,:].round(3)) - 24)
                assert zero_count==0 , "El vector de input está incompleto. Debería tener 24 datos de entrada pero solo tiene "+str(np.count_nonzero(in_unnorm[0,:].round(3)))+" datos." 
                #assert 0.0 not in in_unnorm, "A su input le faltan datos, por favor completrarlos."
                out_unnorm = self.inverse_norm_func(x_norm2,self.mean_out,self.std_out)
                real_unnorm = self.inverse_norm_func(real,self.mean_out[:end_real],self.std_out[:end_real]) if real is not None else None
           
        return in_unnorm, out_unnorm, real_unnorm
    

    def preparando_tensores(self,x,batch_size=8,zero_index=None): #x:=[bs,seq,feat]
        '''
        x será el tensor seteado donde cada fila corresponde a cada año y el largo de la secuencia corresponde a la cantidad de semanas.
        '''
        año_actual = self.year +1
        if self.train==False: #prediction
            #Aplanando el dataset
            index = -(año_actual-self.year) if self.year < año_actual else año_actual-self.year #número negativo
            flatten_x = x[index-1:index-1+1,-24:,:]
        
            for i in range(-index,-1,-1): #decremento desde index-1 hasta cero de 1 en 1.
                
                if i==0 :
                    break
                elif i==1:
                    flatten_x = np.concatenate([flatten_x,x[-i:,:,:]],axis=1)
                else: 
                    flatten_x = np.concatenate([flatten_x,x[-i:-i+1,:,:]],axis=1)


            weeks_columns = []
            pivote=self.fecha_inicio
            len_end = flatten_x.shape[1] - 24 

            c=0
            for mes in itertools.cycle(reversed(self.weeks)):
                if mes == pivote:
                    c+=1            
                
                elif mes != pivote and c>0:
                    weeks_columns.append(mes)
                    c+=1
                
                if c==25:
                    break
            
            weeks_columns = list(reversed(weeks_columns))
            #for output
            c=0
            for mes in itertools.cycle(self.weeks):
                if mes == pivote:
                    weeks_columns.append(mes)
                    c+=1
                elif mes != pivote and c>0:
                    weeks_columns.append(mes)
                    c+=1
        
                if c==len_end:
                    break
            #preparando las semanas
            self.week_columns = weeks_columns
            to_find_pivote = self.weeks[24:] +self.weeks #juntamos todas las semanas para encontrar el pivote de donde separar el input del output.
            #buscando el pivote
            indice_pivote = to_find_pivote.index(pivote,24)# + 48*factor[str(self.year)]))
            #Seteando el input y el output para la predicción deseada.
            input_tensor = flatten_x[:,indice_pivote-24:indice_pivote,:]
            output_tensor = flatten_x[:,indice_pivote:indice_pivote+24,:]


            dl = DataLoader(torch.tensor(input_tensor, dtype = torch.float),batch_size=1,shuffle=False)
            return dl,output_tensor
        
        else: #Train 
            #x está normalizado y tiene la forma [Años, seq,1]
            x_in, x_out = self.shifted_matrix2(x=x,cat_samples=True,target=x,zero_index=zero_index)
            # añadiendo ruido a los datos
            ruido = x_in + np.random.normal(0,0.15,size=x_in.shape)
            ruido2 = x_in + np.random.normal(0,0.05,size=x_in.shape)
            ######
            x_in = np.concatenate([x_in,ruido,ruido2],axis=0)
            x_in = torch.tensor(x_in,dtype=torch.float)
            x_out = np.concatenate([x_out,x_out,x_out],axis=0)
            x_out = torch.tensor(x_out[:,:,:1],dtype=torch.float)
            x_tensor = TensorDataset(x_in,x_out)
            dl = DataLoader(x_tensor,batch_size=batch_size,shuffle=True)

            return dl
    
    def procesando_datos(self,paths):
        #paths debe ser una lista
        #colocando el alto de que el largo de la lista de directorios debe coinsidir con el número de features.
        assert len(paths)==self.nf, "La cantidad de datos en la lista paths debe coincidir con el número de features"

        dic_norm_params = {'means':[],'std':[]}     
        df_list = []
        #ciclo para cargar datos
        for d in paths:
            df_ = pd.read_csv(d,index_col="Unnamed: 0")
            df_array = np.array(df_)
            if self.year_init>2000:
                inicio = self.year - self.year_init 
                df_list.append(df_array[-inicio:])
            else:
                df_list.append(df_array[:])
        
        #ciclo para normalizar datos. Esta normalización es apta para ambas tareas (Entrenamiento y Predicción)
        for i,matx in enumerate(df_list):
            y,seq = matx.shape
            #Agregar por aquí la data augmentation con la tendencia.
            if i==0:
                zero_index = np.where(matx[:-1,:].reshape(-1) > 0.0)[0] 
                x_norm = self.normalizacion(matx).reshape(y,seq,1)
                #Buscando en qué índice hay ceros.
                dic_norm_params['means'].append(self.sumando)
                dic_norm_params['std'].append(self.factor)
            else:
                x_norm = np.concatenate([x_norm,
                                         self.normalizacion(matx).reshape(y,seq,1)],axis=2)
                dic_norm_params['means'].append(self.sumando)
                dic_norm_params['std'].append(self.factor)

        if self.train==False:
            x_dl,x_out = self.preparando_tensores(x_norm) #para la predicción
            return x_dl, x_out, dic_norm_params
        
        elif self.train:

            x_dl = self.preparando_tensores(x_norm,batch_size=32,zero_index=zero_index)
            return x_dl, dic_norm_params
            
                
