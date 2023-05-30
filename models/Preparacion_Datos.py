import numpy as np
import pandas as pd
import os

class ProcesamientoDatos1():
    '''
    Esta clase está diseñada para Automatizar la obtención de los datos (no confundir con
    el webscrappin de datos), entonces, se ingresarán a los métodos los datos brutos,
    e intarnamente se limpiarán y se agruparán en semanas ENEL para finalemente obtener 
    las matrices para cada dato (que corresponderá a cada feature).
    '''
    
    def __init__(self,path_df,kind=None,sheet_name=None,last_month=12,last_year=2022,last_week=4):
        self.lm = last_month
        self.ly = last_year
        self.lw = last_week
        
        if kind=='enel afluent':
            self.enel1 = pd.read_excel(path_df+"/EnerAflu2018b_JA_2000-2010.xlsx",sheet_name="EnerAfluente",header=2,usecols=[3,5,7,8,9,10,11,12])
            #Cambiar el path de enel2 con el actual. 
            self.enel2 = pd.read_excel(path_df+"/EnerAflu2018b_JA.xlsx",sheet_name="EnerAfluente",header=2,usecols=[3,5,7,8,9,10,11,12])
            self.enel3 = pd.read_excel(path_df+"/EnerAflu2018b.xlsx",sheet_name="EnerAfluente",header=2,usecols=[3,5,7,8,9,10,11,12])
            self.enel3.rename(columns={"Unnamed: 3":"fecha"},inplace=True)
            self.enel1.set_index("fecha",drop=False,inplace=True)
            self.enel2.set_index("fecha",drop=False,inplace=True)
            self.enel3.set_index("fecha",drop=False,inplace=True)            

        elif kind=='extra afluent':
            self.df_hist = pd.read_excel(path_df+"/Datos_Hist_v2 Copia3.xlsx",sheet_name='caudales')
            self.df_new = pd.read_excel(path_df+"/Datos_New.xlsm",sheet_name="caudales")
            self.ralco_hist = pd.read_excel(path_df+"/CaudalRalcoIPLP.xlsx",sheet_name="Hoja1")

        elif kind=='lluvia':
            self.df = pd.read_excel(path_df,sheet_name=sheet_name,header=0)

        elif kind=='nieve':
            self.cob_maule = pd.read_csv(path_df+"SCA_Maule.csv")
            self.cob_maule.set_index("fecha",drop=False,inplace=True)
            self.cob_maule = self.cob_maule.loc['2000-04,01':,:]
            self.cob_ñuble = pd.read_csv(path_df+"SCA_Ñuble.csv")
            self.cob_ñuble.set_index("fecha",drop=False,inplace=True)
            self.cob_ñuble=self.cob_ñuble.loc['2000-04,01':,:]
            self.cob_cach = pd.read_csv(path_df+"SCA_Cachapoal.csv")
            self.cob_cach.set_index("fecha",drop=False,inplace=True)
            self.cob_cach = self.cob_cach.loc['2000-04,01':,:]

    def semanas_enel(self,df,mes,dia):
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
    
    def determinar_meses_semanas_años_enel(self,df):
         
        mesh, añoh= [],[]
        for i,mes in enumerate(df['mes']):
            if mes>=4:
                value = mes-3
                value_a=df['año'][i]
                mesh.append(value)
                añoh.append(value_a)
            
            if mes<4:
                value = mes+(12-3)
                value_a = df['año'][i]-1
                añoh.append(value_a)
                mesh.append(value)
            
        df['mesh'] = mesh
        df['añoh'] = añoh
        week = self.semanas_enel(df,'mesh','dia')
        df['semh'] = week

        return df
        
    
    def crear_matriz(self,df,col_value):
        df_week = df.groupby(['añoh','mesh','semh'])
        años = df['añoh'].unique()

        dic = {str(i):[] for i in range(1,49)}
        for a in años:
            k=1
            for m in range(1,13):
                for s in range(1,5):
                    if a > 1999:
                        if (a==self.ly and m==self.lm and s>=self.lw): 
                            dic[str(k)].append(0)
                        elif (a==self.ly and m>self.lm):
                            dic[str(k)].append(0)
                        elif (a>self.ly):
                            break
                        else :
                            media = df_week.get_group((a,m,s))[col_value].mean()
                            dic[str(k)].append(media)
                        k+=1 
        return pd.DataFrame(dic)
    
    def crear_matriz_afluente_enel(self):
        print("afluente enel")
        afluente_enel = pd.concat([self.enel1.loc['2000-04-01':'2011-03-31',:],self.enel2.loc['2011-04-01':'2019-03-31',:],self.enel3.loc['2019-04-01':,:]],
                          axis=0).reset_index(drop=True)
        afluente_enel.dropna(subset=["ENEL"],inplace=True)
        afluente_enel["fecha"] = pd.to_datetime(afluente_enel['fecha'])
        afluente_enel["dia"] = afluente_enel['fecha'].dt.day
        afluente_enel["mes"] = afluente_enel['fecha'].dt.month
        afluente_enel['año'] = afluente_enel['fecha'].dt.year
        afluente_enel1 = self.determinar_meses_semanas_años_enel(afluente_enel)
        afluente_enel2 = self.crear_matriz(afluente_enel1,"ENEL")
        
        return afluente_enel2
        
    
    def crear_matriz_afluentes(self):
        print("afluentes extras")
        #Para ralco primero:
        df_ralco1 = self.df_hist.loc[5204:6574,[51,"Ralco"]].reset_index(drop = True)
        df_ralco2 = self.df_new.loc[:,[51,"Ralco"]].reset_index(drop=True)
        df_ralco2.dropna(subset=["Ralco"],inplace=True)
        df_ralco3 = pd.concat([df_ralco1,df_ralco2],axis=0).reset_index(drop=True)
        sub_ralco = self.ralco_hist.iloc[46:52,2:]
        sub_ralco.columns = [str(i) for i in range(1,49)]
        df_ralco3[51] = pd.to_datetime(df_ralco3[51])
        df_ralco3['dia'] = df_ralco3[51].dt.day
        df_ralco3['mes'] = df_ralco3[51].dt.month
        df_ralco3['año'] = df_ralco3[51].dt.year
        df_ralco4 = self.determinar_meses_semanas_años_enel(df_ralco3)
        df_ralco5 = self.crear_matriz(df_ralco4,"Ralco")
        df_ralco6 = pd.concat([sub_ralco,df_ralco5],axis=0,ignore_index=True)

        #Ajustando los afluentes de otras centrales
        self.df_hist = self.df_hist.loc[2922:6574,[51,"AfluLaja","ANetInve","HIntIsla","Polcura","qbomau","qrmela"]].reset_index(drop=True)
        self.df_new = self.df_new.loc[:,[51,"AfluLaja","ANetInve","HIntIsla","Polcura","qbomau","qrmela"]].reset_index(drop=True)
        self.df_new.dropna(subset=["AfluLaja"],inplace=True)
        self.df1 = pd.concat([self.df_hist,self.df_new],axis=0).reset_index(drop=True) 
        self.df1[51] = pd.to_datetime(self.df1[51])
        self.df1['dia'] = self.df1[51].dt.day
        self.df1['mes'] = self.df1[51].dt.month
        self.df1['año'] = self.df1[51].dt.year
        self.df2 = self.determinar_meses_semanas_años_enel(self.df1)
        df_biobio = self.df2[[51,"AfluLaja","Polcura","dia","mes","año","mesh","añoh","semh"]]
        df_maule = self.df2[[51,"ANetInve","HIntIsla","qbomau","qrmela","dia","mes","año","mesh","añoh","semh"]]
        df_biobio['media'] = np.sum(df_biobio[["AfluLaja","Polcura"]],axis=1)
        df_maule["media"] = np.mean(df_maule[["ANetInve","HIntIsla","qbomau","qrmela"]],axis=1)
        
        self.biobio = self.crear_matriz(df_biobio,"media")
        self.biobio = (self.biobio + df_ralco6)/3
        self.maule = self.crear_matriz(df_maule,"media")

        return self.biobio, self.maule

    def crear_matriz_lluvia(self):
        print("lluvia")
        self.df = self.df.iloc[91:,:] #Para tomar desde el año 2000
        self.df = self.df.rename(columns={'Unnamed: 0':'fecha'})
        self.df.index = self.df['fecha']

        self.df = self.df.drop(columns=['Unnamed: 16', 'Unnamed: 17','Unnamed: 18',
        'Unnamed: 19', 'Unnamed: 20', 'Unnamed: 21'],axis=1)
        self.df = self.df.fillna(value=0.0)

        self.df['dia'] = self.df['fecha'].dt.day

        self.df=self.determinar_meses_semanas_años_enel(self.df) #se actualiza el DataFrame

        new_df = self.df.loc[:,['Abanico','Canutil2','Cipreses','Colbun','Molles','Pangue','Pehuench','Pilmaiqu','Pullinqu','Rapel','Sauzal']].mean(axis=1)
        new_df = pd.concat([self.df.loc[:,['mes','año','añoh','mesh','semh']],new_df],axis=1)

        matrix = self.crear_matriz(new_df,0)

        return matrix


    def crear_matrix_nieve(self):
        print("Nieve")
        self.cob_cach = self.cob_cach.fillna(value=0.0) if self.cob_cach.isna().sum().sum() > 0 else self.cob_cach
        self.cob_maule = self.cob_maule.fillna(value=0.0) if self.cob_maule.isna().sum().sum() > 0 else self.cob_maule
        self.cob_ñuble = self.cob_ñuble.fillna(value=0.0) if self.cob_ñuble.isna().sum().sum() > 0 else self.cob_ñuble

        self.cob_cach["fecha"] = pd.to_datetime(self.cob_cach["fecha"])
        self.cob_maule["fecha"] = pd.to_datetime(self.cob_maule["fecha"])
        self.cob_ñuble["fecha"] = pd.to_datetime(self.cob_ñuble["fecha"])

        self.cob_cach['dia'] = self.cob_cach["fecha"].dt.day
        self.cob_cach['mes'] = self.cob_cach["fecha"].dt.month
        self.cob_cach['año'] = self.cob_cach["fecha"].dt.year

        self.cob_maule['dia'] = self.cob_maule["fecha"].dt.day
        self.cob_maule['mes'] = self.cob_maule["fecha"].dt.month
        self.cob_maule['año'] = self.cob_maule["fecha"].dt.year

        self.cob_ñuble['dia'] = self.cob_ñuble["fecha"].dt.day
        self.cob_ñuble['mes'] = self.cob_ñuble["fecha"].dt.month
        self.cob_ñuble['año'] = self.cob_ñuble["fecha"].dt.year

        self.cob_cach = self.determinar_meses_semanas_años_enel(self.cob_cach)
        self.cob_maule = self.determinar_meses_semanas_años_enel(self.cob_maule)
        self.cob_ñuble = self.determinar_meses_semanas_años_enel(self.cob_ñuble)
        
        matrix_cach = self.crear_matriz(self.cob_cach,"SCA_km2")
        matrix_maule = self.crear_matriz(self.cob_maule,"SCA_km2")
        matrix_ñuble = self.crear_matriz(self.cob_ñuble,"SCA_km2")


        def_matix = (matrix_cach + matrix_maule + matrix_ñuble)/3

        return def_matix








