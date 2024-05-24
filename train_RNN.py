from torch import nn
import torch

class RNN_Cell(nn.Module):
    def __init__(self,in_dim,hidden_dim):
        super(RNN_Cell,self).__init__()
        self.Wx=nn.Parameter(torch.rand(in_dim,hidden_dim))
        self.Wh=nn.Parameter(torch.rand(hidden_dim,hidden_dim))
        self.b=nn.Parameter(torch.rand(1,hidden_dim))

    def forward(self,x,h_1):
        h=torch.tanh(torch.matmul(x,self.Wx)+torch.matmul(h_1,self.Wh)+self.b)
        return h

class RNN(nn.Module):
    def __init__(self,in_dim,hidden_dim):
        super(RNN,self).__init__()
        self.hidden_dim=hidden_dim
        self.rnn_cell=RNN_Cell(in_dim,hidden_dim)

    def forward(self,x):
        #x:[seq_lens,batch_size,in_dim]
        outs=[]
        h=None
        for seq_x in x:
            if h is None:
                h=torch.randn(x.shape[1],self.hidden_dim)
            h=self.rnn_cell(seq_x,h)
            outs.append(torch.unsqueeze(h,0))
        outs=torch.cat(outs)
        return outs,h

if __name__=='__main__':
    batch_size=24
    seq_lens=7
    in_dim=12
    out_dim=6
    rnn=RNN(in_dim,out_dim)
    x=torch.randn(seq_lens,batch_size,in_dim)
    outs,h=rnn(x)
    print(outs.shape)
    print(h.shape)
