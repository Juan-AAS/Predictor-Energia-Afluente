import torch
import torch.nn as nn
import random
#Sequence to sequence models with attention for Time Series Forecasting

class CNN(nn.Module):
    def __init__(self, in_channel, out_channel, ks, s, activation):
        super(CNN,self).__init__()
        self.inp = in_channel #num_feat
        self.out = out_channel #emb
        self.ks = ks
        self.s = s

        self.cnn1 = nn.Conv1d(in_channels=in_channel,out_channels=out_channel,kernel_size=ks,stride=s)
        self.maxpool = nn.MaxPool1d(kernel_size=ks,stride=s)
        self.activation = activation


    def forward(self,x):
        #x must be: [bs,feat or channel_in,seq]
        #x = [bs, seq, feat]
        x = x.transpose(2,1) #[bs,feat, seq]
        #x = self.batch1d(x)
        x = self.cnn1(x) #[bs, emb, seq_out]
        x = self.activation(x)

        return x
    
    


class Encoder(nn.Module):
    def __init__(self,input_dim, embedding_dim, hidden_dim, n_layer, seq, dropout, rnn_type='rnn',act='relu',bidirectional=False):
        super(Encoder,self).__init__()
        self.hid_dim = hidden_dim
        self.rnn_type = rnn_type
        self.bidi = bidirectional
        self.emb_dim =embedding_dim
        self.act=act
        self.activation = {
            'relu':nn.ReLU(), 
            'sig':nn.Sigmoid(),
            'tanh':nn.Tanh(),
            'elu':nn.ELU(),
            'sin':torch.sin,
            'cos':torch.cos}
        self.D = n_layer
        self.n_layer = n_layer*2 if bidirectional else n_layer
        self.input_dim = input_dim

        
        if embedding_dim is not None:
            self.embedding = CNN(input_dim,embedding_dim,5,1,self.activation[act])
            self.embedding2 = CNN(embedding_dim,embedding_dim,3,1,self.activation[act])
            seq_out = int((seq - (5-1) -1)/1 +1)
            seq_out2 = int((seq_out -(3-1) -1)/1 +1)
            self.linear_emb1 = nn.Linear(seq_out2*embedding_dim,seq*embedding_dim) 
            self.input_dim=embedding_dim

        self.layernorm = nn.LayerNorm(self.input_dim)
        

        if rnn_type=='rnn' or rnn_type=='gru':
            if rnn_type=='rnn': 
                self.rnn = nn.RNN(
                    input_size=self.input_dim,
                    hidden_size=hidden_dim, 
                    num_layers=self.D, 
                    dropout=dropout, 
                    batch_first=True,
                    bidirectional=bidirectional)
            else: 
                self.rnn = nn.GRU(
                    input_size=self.input_dim, 
                    hidden_size=hidden_dim, 
                    num_layers=self.D, 
                    dropout=dropout, 
                    batch_first=True,
                    bidirectional=bidirectional
                    )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, device):
        batch, seq_len, in_dim = x.shape 
        #x = x.reshape(batch,seq_len,in_dim)
        
        if self.emb_dim is not None:
            x = self.activation[self.act](self.embedding(x)) #[bs,emb,seq_out]
            x = self.activation[self.act](self.embedding2(x.transpose(2,1)))
            x = torch.flatten(x,start_dim=1) #[bs,seq_out*emb]
            x = self.activation[self.act](self.linear_emb1(self.dropout(x))) #[bs,embedding * seq]
            x = x.view(batch,seq_len,self.emb_dim) 
        
        x = self.layernorm(x)
        hidden = torch.zeros(self.n_layer,batch,self.hid_dim,requires_grad=True).to(device)
        if self.rnn_type=='rnn' or self.rnn_type=='gru':
            out, hidden = self.rnn(x,hidden)
        
            if self.n_layer>1 and self.bidi==False: 
                hidden = torch.sum(hidden,dim=0)

            if self.bidi:
                hidden = torch.sum(hidden,dim=0)
                out = out.view(batch,seq_len,self.n_layer,self.hid_dim) 
                out = torch.sum(out,dim=2)
            
            
            return out, hidden

####################################################################################################################################################

class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, emb_size):
        super(Attention,self).__init__()
        self.hidden_dim = hidden_dim
        self.tanh=nn.Tanh()
        self.elu=nn.ELU()
        self.softmax=nn.Softmax(dim=-1)
        self.attn_linear = nn.Linear(hidden_dim*2, hidden_dim)
        self.energy_linear = nn.Linear(hidden_dim, 1)
        self.out_inp = nn.Linear(hidden_dim + emb_size, emb_size)
        self.layernorm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self,encoder_out, prev_hidden, prev_y):
        #prev_hidden: [bs, hd]
        #prev_y: [bs,feat or emb]
        #encoder_out: [bs,seq,hd]
        #First part:
        seq = encoder_out.shape[1]
        prev_y = prev_y.unsqueeze(1) #[bs,1,feat]
        prev_hidden = prev_hidden.unsqueeze(1) #[bs,1,hd]
        prev_hidden = prev_hidden.repeat(1,seq,1) #[bs,seq,hd]
        cat = torch.cat([prev_hidden,encoder_out],dim=2) #[bs,seq,hd+hd]
        attn = self.tanh(self.attn_linear(cat)) #[bs,seq,hd]
        attn = self.layernorm2(attn)
        energy = self.softmax(self.energy_linear(attn)) #[bs,seq,1]

        energy = energy.transpose(1,2) #[bs,1,seq]
        w = torch.bmm(energy,encoder_out) #[bs,1,hd] #context
        context = torch.cat([prev_y,w],dim=-1)
        input_dec = self.out_inp(context) #[bs,1,feat]

        return input_dec.squeeze(1)

#######################################################################################################################################################

class Decoder(nn.Module):
    def __init__(self,out_dim, embedding_dim, hidden_dim, dropout, rnn_type='rnn',act='relu',with_attention=False):
        super(Decoder,self).__init__()
        self.out_dim = out_dim
        self.embedding_dim = embedding_dim
        self.hid_dim = hidden_dim
        self.rnn_type = rnn_type
        self.with_attention = with_attention
        self.act = act
        self.activation = {
            'relu':nn.ReLU(), 
            'sig':nn.Sigmoid(),
            'tanh':nn.Tanh(),
            'elu':nn.ELU(),
            'sin':torch.sin,
            'cos':torch.cos
            }
        self.out_dim = out_dim

        if embedding_dim is not None:
            self.embedding = nn.Linear(out_dim,embedding_dim) 
            self.out_dim=embedding_dim
            self.last = nn.Linear(self.out_dim,1)
        
        self.attention = Attention(self.out_dim,hidden_dim,self.out_dim)
        
        self.layernorm3 = nn.LayerNorm(hidden_dim)

        if rnn_type=='rnn' or rnn_type=='gru':
            if rnn_type=='rnn': 
                self.rnn = nn.RNNCell(
                    input_size=self.out_dim,
                    hidden_size=hidden_dim, 
                    )
            else: 
                self.rnn = nn.GRUCell(
                    input_size=self.out_dim, 
                    hidden_size=hidden_dim, 
                    )
        
        self.pre_out= nn.Linear(self.hid_dim + self.hid_dim, self.hid_dim)
        self.out = nn.Linear(self.hid_dim, self.out_dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self,x, enc_out, hidden):
        #x:=[batch, 1]
        #hidden := [batch, hid_dim]
        #cell := [batch, hid_dim]
        if self.embedding_dim is not None:
            x = self.activation[self.act](self.dropout(self.embedding(x)))   #[bs, ouput_dim or emb_dim]         
        
        #attention layer
        if self.with_attention:
            x = self.attention(enc_out, hidden, x)
            x = self.activation[self.act](x)

        if self.rnn_type=='rnn' or self.rnn_type=='gru':
            #rnn cell
            hidden = self.rnn(x,hidden)
            hidden = self.layernorm3(hidden)
            #predicction layer
            pred = self.out(hidden)
            
            if self.embedding_dim is not None:
                pred = self.activation[self.act](pred)
                pred = self.last(pred)

            #pred:=[batch,out_dim]
            return pred, hidden
########################################################################################################################################

class Forecasting(nn.Module):
    def __init__(self,encoder,decoder,device):
        super(Forecasting,self).__init__()
        self.enc = encoder
        self.dec = decoder
        self.device = device

    def forward(self,inp,outp=None,teacher_forcing=0.6):
        #inp := [batch,seq,features] =: outp
        bs,seq_in,out_dim = inp.shape

        seq = seq_in if outp is None else outp.shape[1] 
        
        #tensor to fill the output
        output = torch.zeros(bs,seq,1).to(self.device)
        
        #Encoder Part
        out_enc, hidden = self.enc(inp, self.device)
        
        #Decoder Part
        new_input = inp[:,-1,:1]  
        for obs in range(0,seq):
            #calling decoder
            y, hidden = self.dec(new_input,out_enc,hidden)
            output[:,obs,:1] = y
            teacher_force = random.random() < teacher_forcing
            if teacher_force:
                new_input = outp[:,obs,:] 
            else:
                new_input = y 
        return output
#############################################################################################################################################