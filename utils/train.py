import torch
from torch.autograd import Variable
import numpy as np
from utils import aux

def train(model, loss, optim, epochs, dl_train, dl_test, device,visualization=False):
    train_loss, test_loss = [], []
    for epoch in range(epochs):
        model.train()
        losst=0
        for batch_idx, (x,y) in enumerate(dl_train):
            x,y = Variable(x.unsqueeze(2)).to(device),Variable(y.unsqueeze(2)).to(device)
            optim.zero_grad()
            y_hat, hidden_out = model(x)
            l = loss(y_hat,y)
            l.backward()
            losst+=l.item()
            optim.step()

        print("train =====> Epoch:{} Average loss:{:.4f}".format(epoch,losst / len(dl_train.dataset)))
        train_loss.append(losst / len(dl_train.dataset))
        test_l,y_hat_test = test(model, loss, epoch, dl_test, device)

        test_loss.append(test_l)
    
    #embedding = torch.cat(emb)
    return model, train_loss, test_loss 

def test(model, loss, epochs, dl, device):
    model.eval()
    loss_=0
    for batch_idx, (x,y,c) in enumerate(dl):
        x,y = x.unsqueeze(2).to(device),y.unsqueeze(2).to(device)
        y_hat, hidden = model(x)
        l = loss(y_hat,y)
        loss_ += l.item()

        print('(test) ====> Epoch:{} Average loss:{:.4f}'.format(epochs, loss_ / len(dl.dataset)))

        loss_test = loss_ / len(dl.dataset)
        print("")
 
        return loss_test, y_hat
