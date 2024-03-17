import torch
import torch.nn.functional as F
import torch.nn as nn

# class CausalConv1d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
#         super(CausalConv1d, self).__init__()
#         self.padding = (kernel_size - 1) * dilation
#         self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
#
#     def forward(self, x):
#         x = nn.functional.pad(x, (self.padding, 0))
#         x = self.conv(x)
#         x = x[:, :, :-self.padding]
#         return x
#
# class TCN(nn.Module):
#     def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
#         super(TCN, self).__init__()
#         self.input_size = input_size
#         self.output_size = output_size
#         self.num_channels = num_channels
#         self.kernel_size = kernel_size
#         self.dropout = dropout
#         self.layers = []
#
#         # Add causal convolutional layers to the model
#         for i in range(len(num_channels)):
#             dilation = 2 ** i
#             in_channels = num_channels[i - 1] if i > 0 else 5
#             out_channels = num_channels[i]
#             causal_conv = CausalConv1d(in_channels, out_channels, kernel_size, dilation=dilation)
#             self.layers.append(causal_conv)
#             self.layers.append(nn.BatchNorm1d(out_channels))
#             self.layers.append(nn.ReLU())
#             self.layers.append(nn.Dropout(dropout))
#         # self.network = nn.Sequential(
#         #     CausalConv1d(5, 16, kernel_size=kernel_size, dilation=1),
#         #     nn.ReLU(),
#         #     nn.Dropout(dropout),
#         #     CausalConv1d(16, 32, kernel_size=kernel_size, dilation=2),
#         #     nn.ReLU(),
#         #     nn.Dropout(dropout),
#         #     CausalConv1d(32, 64, kernel_size=kernel_size, dilation=4),
#         #     nn.ReLU(),
#         #     nn.Dropout(dropout),
#         #     nn.Flatten(),
#         #     nn.Linear(num_channels*94, output_size)
#         # )
#
#         # Add final linear layer to the model
#         self.layers.append(nn.Linear(num_channels[-1], output_size))
#
#         # Create sequential model from the layers list
#         self.network = nn.Sequential(*self.layers)
#
#     def forward(self, x):
#         # x = x.transpose(1, 2)
#         x = self.network(x)
#         # return x.transpose(1, 2)
#         return x




class CausalConv1d(nn.Conv1d):
    """
    1D causal convolution layer
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super(CausalConv1d, self).__init__(in_channels, out_channels, kernel_size, dilation=dilation, **kwargs)
        self.padding = dilation * (kernel_size - 1)

    def forward(self, x):
        x = super(CausalConv1d, self).forward(x)
        return x[:, :, :-self.padding]


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, num_layers, dilation_base):
        super(TCN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.num_layers = num_layers
        self.dilation_base = dilation_base

        self.first_conv = CausalConv1d(in_channels=input_size, out_channels=num_channels, kernel_size=kernel_size)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i * dilation_base
            conv = CausalConv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=kernel_size,
                                dilation=dilation)
            relu = nn.ReLU()
            self.layers.append(nn.Sequential(conv, relu))

        self.last_conv = nn.Conv1d(in_channels=num_channels, out_channels=output_size, kernel_size=1)

        self.dropout = nn.Dropout(dropout)

        # self.linear = nn.Linear(5,5)

    def forward(self, inputs):
        inputs = torch.Tensor(inputs)
        x = inputs.view(1,200, 5)
        # inputs has shape [batch_size, input_size, sequence_length]
        # x = inputs.transpose(1, 2)  # reshape to [batch_size, sequence_length, input_size]

        # pass input through first convolutional layer
        x = self.first_conv(x)

        # pass input through each TCN layer
        for layer in self.layers:
            residual = x
            x = layer(x)
            x = x + residual  # add residual connection

        # pass output of last TCN layer through final convolutional layer
        x = self.last_conv(x)

        # apply dropout and reshape output to [batch_size, output_size, sequence_length]
        x = self.dropout(x)
        # x = x.transpose(1, 2)  # reshape to [batch_size, output_size, sequence_length]

        return x.view(200,5)


class GRULayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRULayer, self).__init__()
        self.hidden_size = hidden_size

        # GRU gates
        self.reset_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.update_gate = nn.Linear(input_size + hidden_size, hidden_size)

        # GRU candidate
        self.candidate = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, input, hidden):
        # concatenate input and hidden state
        combined = torch.cat((input, hidden), 1)

        # compute reset and update gates
        reset = torch.sigmoid(self.reset_gate(combined))
        update = torch.sigmoid(self.update_gate(combined))

        # compute candidate hidden state
        combined = torch.cat((input, reset * hidden), 1)
        candidate = torch.tanh(self.candidate(combined))

        # compute new hidden state
        new_hidden = update * hidden + (1 - update) * candidate

        return new_hidden



class CAEModel(nn.Module):

    def __init__(self, n , d, hidden_size, l1_graph_penalty, alpha, rho, num_hidden_layers=3, input_dim=1, output_dim=1, B_init=None, device='cpu',):

        super(CAEModel, self).__init__()
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
        self.bas = nn.Parameter(torch.zeros(self.d), requires_grad=True)
        self.hid_state = torch.randn(self.n, self.hidden_size)
        self.B_init = B_init
        if self.B_init is not None:
            self._B = nn.Parameter(torch.Tensor(self.B_init), requires_grad=True)
        else:
            self._B = nn.Parameter(torch.zeros(self.d, self.d), requires_grad=True)

        # encoder
        self.Encoder = TCN(self.n,self.n, 64, 3, 0.2, 1 ,4)

        #decoder
        self.g_1 = GRULayer(d, hidden_size)
        # self.g_1 = nn.Linear(d, hidden_size)
        self.g_2 = nn.Linear(hidden_size, d)




        self.hid_1 = nn.Linear(self.input_dim, self.hidden_size)
        self.hid_n = nn.Linear(self.hidden_size, self.hidden_size)
        self.hid_last = nn.Linear(self.hidden_size,self.output_dim)

    def deconder_forward(self, x, hid):
        d = F.relu(self.g_1(x, hid))
        # output = F.relu(self.g_2(d))
        return output

    def preprocess_B(self, B):

        if self.device == 'gpu':
            return (1. - torch.eye(self.d)).cuda(device=0) * B
        else:
            return (1. - torch.eye(self.d)) * B

    def forward(self, x):
        self.x = torch.Tensor(x)
        self.B = self.preprocess_B(self._B)
        self.x_en = self.Encoder(x)
        # self.hidden = self.x_en @ self.B + self.bas
        self.x_hat = self.x_en @ self.B + self.bas
        # self.x_hat = self.deconder_forward(self.hidden, self.hid_state)


        # DAG约束
        self.h = torch.trace(torch.matrix_exp(self.B * self.B)) - self.d
        # # loss：ls_cy
        # self.loss = (0.5 / self.n * torch.sum(torch.square(self.x - self.x_hat))+ self.l1_graph_penalty * torch.norm(self.B, p=1) + self.alpha * self.h)
        # loss:ls_acy  之前用这个
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
        #  * torch.sum(torch.log(torch.sum(torch.square(self.x - self.x_hat), axis=0)))
        #  - torch.linalg.slogdet(torch.eye(self.d) - self.B)[1]) + \
        #             self.alpha * self.h + \
        #             self.l1_graph_penalty * torch.norm(self.B, p=1)


        return self.loss, self.B, self.h




if __name__ == '__main__':
    # x = torch.randn(200,5)
    # Caenet = CAEModel(200, 5, hidden_size=10, l1_graph_penalty=2e-3, alpha=5.0, rho=2e-4, B_init=None)
    # loss, B, h = Caenet(x)
    # print(loss,B,h)

    # x = torch.Tensor(1,5,200)
    # # input_size, output_size, num_channels, kernel_size, dropout, num_layers, dilation_base
    # TCN = TCN(200,200, 32, 3, 0.2, 2 ,1 )
    # y = TCN(x)
    # print(y.shape)

    x = torch.randn(200,5)
    hidden = torch.randn(200,5)
    gru = GRULayer(5,5)
    output = gru(x,hidden)
    print(output.shape)
