import numpy as np
import torch

def SIDn(x,y):
    # print(np.sum(x))
    p = np.zeros_like(x,dtype=np.float)
    q = np.zeros_like(y,dtype=np.float)
    Sid = 0
    for i in range(len(x)):
        p[i] = x[i]/np.sum(x)
        q[i] = y[i]/np.sum(y)
        # print(p[i],q[i])
    for j in range(len(x)):
        Sid += p[j]*np.log10(p[j]/q[j])+q[j]*np.log10(q[j]/p[j])
    return Sid

def SID(x,y):
    # print(np.sum(x))
    p = torch.zeros_like(x)
    q = torch.zeros_like(y)
    Sid = 0
    for i in range(len(x)):
        p[i] = x[i]/torch.sum(x)
        q[i] = y[i]/torch.sum(y)
        # print(p[i],q[i])
    for j in range(len(x)):
        Sid += p[j]*torch.log10(p[j]/q[j])+q[j]*torch.log10(q[j]/p[j])
    #print("Sid",Sid)
    return Sid

def Dark_prior(img,batchsize):
    w0 = 0.95
    h = img.size()[2]
    w = img.size()[3]
    #print(img.size())
    darkchannel_img,darkchannel_img_indice = torch.max(img,1)
    #print(darkchannel_img.size())
    darkchannel_img = torch.reshape(darkchannel_img, (batchsize, 1, h, w))
    #print(darkchannel_img.size())
    #Air = torch.max(img)
    #t = 1-w0*(darkchannel_img/Air)
    return darkchannel_img

def SAMn(x,y):
    s = np.sum(np.dot(x,y))
    t = np.sqrt(np.sum(x**2))*np.sqrt(np.sum(y**2))
    th = np.arccos(s/t)
    # print(s,t)
    return th

def SAM(x,y):
    s = torch.sum(torch.mul(x,y))
    t = torch.sqrt(torch.sum(torch.mul(x,x)))*torch.sqrt(torch.sum(torch.mul(y,y)))
    th = torch.acos(torch.div(s,t))
    # print(s,t)
    return th