import torch
import torch.nn.functional as F
import torch.nn as nn

class GAEModel(nn.Module):

    def __init__(self, n , d, hidden_size, l1_graph_penalty, alpha, rho, num_hidden_layers=3, input_dim=1, output_dim=1, B_init=None, device='cpu',):

        super(GAEModel, self).__init__()
        # 输入节点的维度
        self.n = n
        # 节点数
        self.d = d
        self.device = device
        self.hidden_size = hidden_size
        self.l1_graph_penalty = l1_graph_penalty
        self.num_hidden_layers = num_hidden_layers
        self.input_dim = input_dim
        self.output_dim  = output_dim


        # self.alpha = torch.Tensor()
        # self.rho = torch.Tensor()
        self.alpha = alpha
        self.rho = rho
        # self.bas = nn.Parameter(torch.zeros(self.d), requires_grad=True)
        self.B_init = B_init
        if self.B_init is not None:
            self._B = nn.Parameter(torch.Tensor(self.B_init), requires_grad=True)
        else:
            self._B = nn.Parameter(torch.zeros(self.d, self.d), requires_grad=True)

        # encoder
        self.h_1 = nn.Linear(d, hidden_size)
        self.h_2 = nn.Linear(hidden_size, d)

        #decoder
        self.g_1 = nn.Linear(d, hidden_size)
        self.g_2 = nn.Linear(hidden_size, d)

        self.hid_1 = nn.Linear(self.input_dim, self.hidden_size)
        self.hid_n = nn.Linear(self.hidden_size, self.hidden_size)
        self.hid_last = nn.Linear(self.hidden_size,self.output_dim)

    def MLP_forward(self, x):
        # x = x.view(-1, self.input_dim)
        x = x.reshape(-1, self.input_dim)

        if self.num_hidden_layers == 0 :
            return x
        elif self.num_hidden_layers == 1 :
            x = F.leaky_relu(self.hid_1(x), negative_slope=0.05)
        else:
            x = F.leaky_relu(self.hid_1(x), negative_slope=0.05)
            for i in range(self.num_hidden_layers-1):
                x = F.leaky_relu(self.hid_n(x),negative_slope=0.05)
        x = self.hid_last(x)
        return x.view(self.n,self.d)

    def encoder_forward(self, x):

        c = F.leaky_relu(self.h_1(x))
        output = F.leaky_relu(self.h_2(c), negative_slope=0.05)
        return output

    def deconder_forward(self, x):
        d = F.leaky_relu(self.g_1(x))
        output = F.leaky_relu(self.g_2(d), negative_slope=0.05)
        return output

    def preprocess_B(self, B):

        if self.device == 'gpu':
            return (1. - torch.eye(self.d)).cuda(device=0) * B
        else:
            return (1. - torch.eye(self.d)) * B

    def forward(self, x):
        self.x = torch.Tensor(x)
        self.B = self.preprocess_B(self._B)
        # self.X = self.encoder_forward(self.x)
        self.X = self.MLP_forward(self.x)
        self.hidden = self.X @ self.B
        # self.x_hat = self.deconder_forward(self.hidden)
        self.x_hat = self.MLP_forward(self.hidden)

        # DAG约束
        self.h = torch.trace(torch.matrix_exp(self.B * self.B)) - self.d
        # # loss：ls_cy
        # self.loss = (0.5 / self.n * torch.sum(torch.square(self.x - self.x_hat))+ self.l1_graph_penalty * torch.norm(self.B, p=1) + self.alpha * self.h)
        # loss:ls_acy
        self.loss = (0.5 / self.n * torch.sum(torch.square(self.x - self.x_hat))- torch.linalg.slogdet(torch.eye(self.d) - self.B)[1]) + \
               self.alpha * self.h + \
               self.l1_graph_penalty * torch.norm(self.B, p=1)
        # loss：mle-ev
        # self.loss = (0.5 * self.d
        #                 * torch.log(torch.square(torch.linalg.norm(self.X - self.x_hat)))
        #                 - torch.linalg.slogdet(torch.eye(self.d) - self.B)[1])+ \
        #        self.alpha * self.h + \
        #             self.l1_graph_penalty * torch.norm(self.B, p=1)
        # loss:mle-nv
        # self.loss = (0.5
        #  * torch.sum(torch.log(torch.sum(torch.square(self.X - self.x_hat), axis=0)))
        #  - torch.linalg.slogdet(torch.eye(self.d) - self.B)[1]) + \
        #             self.alpha * self.h + \
        #             self.l1_graph_penalty * torch.norm(self.B, p=1)


        return self.loss, self.B, self.h








