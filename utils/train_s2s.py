import torch
from utils import aux_

def train(model, optim, loss_function, dl, device, clip, seq2seq=True, dense =False, visualization=False,epoch=0):
    model.train()
    loss_value = 0
    for i, (x,y) in enumerate(dl):
        #x, y := [batch, seq]
        x = x.unsqueeze(2).to(device) if len(x.shape)==2 else x.to(device) #[bs,seq,input_dim]
        y = y.unsqueeze(2).to(device) if len(y.shape)==2 else y.to(device)#[bs,seq,feat]

        optim.zero_grad()
        if seq2seq:
            output = model(x,y)

        else:
            if dense:
                output = model(x)
            else:
                output,h = model(x)
        #output : = [batch, seq, 1]
        #output = output.squeeze(2)
        #output := [batch, seq]
        loss = loss_function(output,y)
        loss.backward()
        #torch.nn.utils.clip_grad_norm(model.parameters(),clip)
        optim.step()
        loss_value+=loss.item()

    if visualization:
        if epoch%100==0:
            aux.step_chart(x.cpu().detach().numpy(),y.cpu().detach().numpy(),output.cpu().detach().numpy(),[i for i in range(1,25)],[j for j in range(25,35)])

    return loss_value/len(dl)

def test(model, dl, loss_function, device, seq2seq=True, dense=False, visualization=False,epoch=0):
    model.eval()
    loss_value=0
    with torch.no_grad():
        out = torch.zeros(1,24,1).to(device)
        for i, (x,y) in enumerate(dl):
            x = x.unsqueeze(2).to(device) if len(x.shape)==2 else x.to(device) #[bs,seq,feat]
            y = y.to(device) #[bs,seq,feat]
 
            new_input = x[:,:24,:]
            if seq2seq:
                out = model(x,outp=y,teacher_forcing=0)
            else:
                if dense:
                    out = model(x)
                else:
                    for k in range(24):
                        output,h = model(new_input)
                        out[:,k,:] = output[:,-1,:]
                        new_input = output
                        #new_input[:,:,1:] = torch.zeros(output.shape[0],output.shape[1],1)
            #output=output.squeeze(2) #[bs,seq]
            loss = loss_function(out,y)
            loss_value += loss.item()

    if visualization:
        if epoch%100==0:
            aux.step_chart(
                x.cpu().detach().numpy(),
                y.cpu().detach().numpy(),
                out.cpu().detach().numpy(),
                [i for i in range(1,25)],
                [j for j in range(25,49)])

    return loss_value/len(dl)

def predict(model,dl,model_path,device,seq2seq=True): #mejorarlo
    if model_path is not None:
        m=torch.load(model_path)
        model.load_state_dict(m['weights'])
    
    model.eval()
    for i, x in enumerate(dl):
        if seq2seq:
            out = model(x.to(device),None,0)
        else: 
            out,h = model(x)
        
    
    return out.cpu().detach().numpy()


