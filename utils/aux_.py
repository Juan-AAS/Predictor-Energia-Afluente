import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def register(PATH,variables,name_file):
    '''
    PATH: Path where it will save all registers from model trained.
    variables: dictionary with all things to save.    
    '''
    path_file = PATH+'/'+name_file
    file = open(path_file,"w")
    for k,v in variables.items():
        file.write(str(k)+" --> "+str(v) + os.linesep)
    file.close()


def step_chart(hist,y,y_hat,x1,x2,figsize=[10,10],excedentes=None,title="",save=False):
    #y := [bs, seq, 1] := y_hat,hist; Take only one row per batch
    #x is a x_axis)
    if len(hist.shape)==3:
        hist = hist[:,:,0]
    if len(hist.shape)==3:
        y = y.squeeze(2)
        
    plt.figure(figsize=figsize)
    plt.title(title)
    if excedentes is not None:
        plt.plot(x1+x2,excedentes[0],'--+',label="P50",alpha=0.5)
        plt.plot(x1+x2,excedentes[1],'--+',label="P70",alpha=0.5)
        plt.plot(x1+x2,excedentes[2],'--+',label="P90",alpha=0.5)
        plt.plot(x1+x2,excedentes[3],'--+',label="P96",alpha=0.5)
        plt.plot(x1+x2,excedentes[4],'--+',label="P99",alpha=0.5)
    if hist.shape[0]>1:
        plt.plot(x1,hist[2,:],"-o",label="input")
        if y is not None:
            plt.plot(x2,y[2,:],'-o',label='Real')
        plt.plot(x2,y_hat[2],'-o',label="Prediction")
    else:
        plt.plot(x1,hist[0,:],"-o",label="input")
        if y is not None:
            plt.plot(x2,y[0,:],'-o',label='Real')
        plt.plot(x2,y_hat[0],'-o',label="Prediction")
    plt.legend()
    plt.grid()
    plt.ylabel("GWh/dia")
    plt.xlabel("Semanas") 
    plt.xticks(x1+x2,rotation=45)
    if save:
        plt.savefig("prediccion "+title)
    plt.show()
    

def shifted_matrix(x, l, ws, lo, s, cat_samples=False,target=None):
    '''
    Function for make dataset with sliding windows.

    x: Original matrix (DataFrame, Array or Tensor)
    l: sequence lenght
    ws: Window Size
    s: Skip or stride
    '''
    nw = int((l - ws - lo + s)/s)
    s1,s2,s3 = x.shape 
    new_matrix = np.array([[x[i,j*s:j*s+ws,:] for j in range(nw)] for i in range(x.shape[0])])
    if target is not None:
        y = np.array([[target[i,j*s+ws:j*s+ws+lo,:] for j in range(nw)] for i in range(x.shape[0])])
        if cat_samples:
            new_matrix = new_matrix.reshape(-1,ws,s3)
            y = y.reshape(-1,lo,s3)
        return new_matrix, y#.reshape(-1,ws,s3)
    elif target is None:
        if cat_samples:
            new_matrix = new_matrix.reshape(-1,ws,s3)
        return new_matrix    

class Normalizer():
    def __init__(self):
        self.weight_vector=[]
    
    def fit_transform(self,x):
        '''
        x must be a matrix (2D array)
        '''
        y = []
        for row in range(x.shape[0]):
            w = np.sqrt(np.sum(x[row,:]**2))
            norm = x[row,:]/w
            self.weight_vector.append(w)
            y.append(norm)

        self.weight_vector = np.array(self.weight_vector).reshape(x.shape[0],1)
        
        return np.array(y)

    def inverse_transform(self,x):
        #assert len(self.weight_vector) != 0 \
        #    "There no a previous fit" 

        #these function is in progress...
        y = x*self.weight_vector
        return y
        
def enel_week(df,mes,dia):
    l_week = []
    for i, mesh in enumerate(df[mes]):
        if mesh == 1 or mesh == 3 or mesh == 6 or mesh == 8:
            if df[dia][i] in range(1,8):
                l_week.append(1)
            elif df[dia][i] in range(8,16):
                l_week.append(2)
            elif df[dia][i] in range(16,23):
                l_week.append(3)
            else :
                l_week.append(4)
    
        elif mesh == 2 or mesh == 4 or mesh == 5 or mesh == 7 or mesh == 9 or mesh == 10 or mesh == 12:
            if df[dia][i] in range(1,8):
                l_week.append(1)
            elif df[dia][i] in range(8,16):
                l_week.append(2)
            elif df[dia][i] in range(16,24):
                l_week.append(3)
            else :
                l_week.append(4)
        elif mesh == 11:
            if df[dia][i] in range(1,8):
                l_week.append(1)
            elif df[dia][i] in range(8,15):
                l_week.append(2)
            elif df[dia][i] in range(15,22):
                l_week.append(3)
            else :
                l_week.append(4)
        
    return l_week
    
def fit_enel_date(df,date_col,islluvia=False):
    df_x_año = df.fillna(value=0) if df.isna().sum().sum() > 0 else df
    df_x_año[date_col] = pd.to_datetime(df_x_año[date_col])
    df_x_año['dia'] = df_x_año[date_col].dt.day
    df_x_año['mes'] = df_x_año[date_col].dt.month
    df_x_año['año'] = df_x_año[date_col].dt.year
    mesh, añoh= [],[]
    for i,mes in enumerate(df_x_año['mes']):
        if mes>=4:
            value = mes-3
            value_a=df_x_año['año'][i]
            mesh.append(value)
            añoh.append(value_a)
            
        if mes<4:
            value = mes+(12-3)
            value_a = df_x_año['año'][i]-1
            añoh.append(value_a)
            mesh.append(value)
            
    df_x_año['mesh'] = mesh
    df_x_año['añoh'] = añoh
    week = enel_week(df_x_año,'mesh','dia')
    df_x_año['semh'] = week

    if islluvia:
        df_x_año2 = df_x_año.loc[:,['Abanico','Canutil2','Cipreses','Colbun','Molles','Pangue','Pehuench','Pilmaiqu','Pullinqu','Rapel','Sauzal']].mean(axis=1)
        df_x_año = pd.concat([df_x_año.loc[:,['mes','año','añoh','mesh','semh']],df_x_año2],axis=1)

    return df_x_año
    
def making_matrix(df,col):
    df_week = df.groupby(['añoh','mesh','semh'])
    años = df['añoh'].unique()
    
    dic={str(i):[] for i in range(1,49)}
    for a in años:
        k=1
        for m in range(1,13):
            for s in range(1,5):
                if (a==2022 and m>=9):
                    dic[str(k)].append(0)
                else:
                    media = df_week.get_group((a,m,s))[col].mean()
                    dic[str(k)].append(media)
                k+=1                
    
    return pd.DataFrame(dic)
