import torch
import numpy as np 
from models import Seq2SeqAttended
from tqdm import tqdm

import configparser 

config  = configparser.ConfigParser()
config.read('config.ini')

NF = config.getint('general','num_feat')
ED = None if config.get('hiperparametros','embedding_dim') else int(config.get('hiperparametros','embedding_dim'))
HD = config.getint('hiperparametros','hidden_dim')
LR = config.getfloat('hiperparametros','learning_rate')
D = config.getfloat('hiperparametros','dropout')

class AfluentEnelForecasting():
    def __init__(self,Train=False,path_model_to_load=None,path_model_to_save=None,device='cuda'):
        '''
        Modelo que manejará lo que es predicción o entrenamiento de modelos Sequence2Sequence para los datos en ENEL. En este modelo se definirá sí se quiere entrenar o predecir
        Si se desea predecir, 'path_model' debe seguir siendo None, y es imperativo ingresar un diccionario con los hiperparámetros a entrenar 'train_hyp'. Si se quiere 
        predecir, es necesario cargar el modelo entrenado con sus hiperparámetros.

        En un futuro esta clase se ajustará para que sea capaz de aprender mientras se hacen predicciones.
        '''
        
        self.device=device
        if Train:
            if path_model_to_load is not None:
                self.path = path_model_to_load
                self.dicc_model = torch.load(path_model_to_load)
                self.model = self.dicc_model['model']
            
            else: 
                enc = Seq2SeqAttended.Encoder(
                    input_dim=NF,
                    embedding_dim=ED,
                    hidden_dim=HD,
                    n_layer=1,
                    seq=24,
                    dropout=D,
                    rnn_type='gru',
                    act='elu',
                    bidirectional=True
                    )
            
                dec = Seq2SeqAttended.Decoder(
                    out_dim=1,
                    embedding_dim=ED,
                    hidden_dim=HD,
                    dropout=D,
                    rnn_type='gru',
                    act='elu',
                    with_attention=True,
                    )
            
                self.model= Seq2SeqAttended.Forecasting(enc,dec,device).to(device)
                self.path = path_model_to_save 
            
            self.model.train()
            self.optim = torch.optim.Adam(self.model.parameters(), lr=LR, weight_decay=1e-9)
            self.mse = torch.nn.MSELoss(reduction="mean")
        
        elif Train==False:
            self.path = path_model_to_load
            self.dicc_model = torch.load(self.path)
            self.model = self.dicc_model['model']
            self.model.eval()

    def prediction(self,dl):
        input,pred = [],[]
        for x in dl:
            x = x.unsqueeze(2).to(self.device) if len(x.shape)==2 else x.to(self.device)
            y_hat = self.model(x,None,0)
            y_hat = y_hat.squeeze(2)
            pred.append(y_hat.cpu().detach().numpy())
            input.append(x.cpu().detach().numpy())
        pred = np.array(pred).squeeze(1)
        input = np.array(input).squeeze(1)
        return input,pred


    def training(self,dl,epoca):
        for e in tqdm(range(epoca)):
            train_loss=0
            for x,y in dl:
                x = x.unsqueeze(2).to(self.device) if len(x.shape)==2 else x.to(self.device)
                y = y.unsqueeze(2).to(self.device) if len(y.shape)==2 else y.to(self.device)

                self.optim.zero_grad()
                output = self.model(x,y)
                loss = self.mse(output,y)
                loss.backward()
                self.optim.step()
                train_loss += loss.item()

            if e%50 == 0:
                print("TRAIN ==============> Época: "+str(e+1)+" Loss promedio: "+str(train_loss))
        
        torch.save(
            {
            'model':self.model,
            'epoch':epoca,
            'lr':LR,
            },self.path
        )

            





