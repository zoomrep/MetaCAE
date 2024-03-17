from common import BaseLearner, Tensor
import torch
from torch import nn
from torch import optim
from torch import autograd

from GAEnet import GAEModel
from CAEnet import CAEModel

class Learner(nn.Module):

    def __init__(self, n, d, meta_step):

        super(Learner, self).__init__()
        self.n = n
        self.d = d
        self.meta_step = meta_step
        self.pars_model1 = []

        self.model_1 = CAEModel(self.n, self.d, hidden_size=7, l1_graph_penalty=5e-2, alpha=10.0, rho=2e-4, B_init=None, device='cpu')
        self.model_2 = CAEModel(self.n, self.d, hidden_size=7, l1_graph_penalty=5e-2, alpha=10.0, rho=2e-4, B_init=None, device='cpu')
        self.model_3 = CAEModel(self.n, self.d, hidden_size=7, l1_graph_penalty=5e-2, alpha=10.0, rho=2e-4, B_init=None, device='cpu')
        self.model_4 = CAEModel(self.n, self.d, hidden_size=7, l1_graph_penalty=5e-2, alpha=10.0, rho=2e-4, B_init=None, device='cpu')

        self.optim_1 = optim.SGD(self.model_2.parameters(),lr=2e-3)
        self.optim_2 = optim.SGD(self.model_3.parameters(),lr=2e-2)
        self.optim_3 = optim.SGD(self.model_4.parameters(),lr=1e-3)

    def parameters(self):
        return self.model_1.parameters()

    def upate_pii(self):
        i = 0
        j = 0
        k = 0
        # 存下model1的参数
        for p in self.model_1.parameters():
            self.pars_model1.append(p.data)
        # 将model1参数传给model2
        for p in self.model_2.parameters():
            p.data = self.pars_model1[i]
            i = i + 1

        for p in self.model_3.parameters():
            p.data = self.pars_model1[j]
            j = j + 1

        for p in self.model_4.parameters():
            p.data = self.pars_model1[k]
            k = k + 1
        self.pars_model1 = []


    def forward(self,data1,data2,data3):
        # 先复制网络的参数
        self.upate_pii()
        para_model2_before = []
        para_model3_before = []
        para_model4_before = []

        for p in self.model_2.parameters():
            para_model2_before.append(p.data)
        for p in self.model_3.parameters():
            para_model3_before.append(p.data)
        for p in self.model_4.parameters():
            para_model4_before.append(p.data)

        for i in range(self.meta_step):

            loss1, _1, h1 = self.model_2(data1)
            self.optim_1.zero_grad()
            loss1.backward()
            self.optim_1.step()

            loss2, _2, h2 = self.model_3(data2)
            self.optim_2.zero_grad()
            loss2.backward()
            self.optim_2.step()

            loss3, _3, h3 = self.model_4(data3)
            self.optim_3.zero_grad()
            loss3.backward()
            self.optim_3.step()

        # 然后计算元梯度并返回，一个epoch一个，上面可以算是训练过程
        # 现在为测试过程：
        test_loss, _, h = self.model_2(data1)
        test_loss, _, h = self.model_3(data2)
        test_loss, _, h = self.model_4(data3)

        para_model2_after = []
        para_model3_after = []
        para_model4_after = []

        for p in self.model_2.parameters():
            para_model2_after.append(p.data)
        for p in self.model_3.parameters():
            para_model3_after.append(p.data)
        for p in self.model_4.parameters():
            para_model4_after.append(p.data)
        # 这里将creat_graph设置为True用来二次反向传播
        # grads_pi = autograd.grad(test_loss, self.model_2.parameters(), create_graph=True, allow_unused=True)
        grads_pi_1 = [para_model2_after[i] - para_model2_before[i] for i in range(len(para_model2_after))]
        grads_pi_2 = [para_model3_after[i] - para_model3_before[i] for i in range(len(para_model2_after))]
        grads_pi_3 = [para_model4_after[i] - para_model4_before[i] for i in range(len(para_model2_after))]

        sum_grads = [(grads_pi_1[i] + grads_pi_2[i] + grads_pi_3[i])/3 for i in range(len(grads_pi_1))]

        return test_loss, sum_grads

    def model_1_forward(self,data):
        return self.model_1(data)

# forward参数为训练data和epoch
# fine_tunning的参数为验证data
class MetaLearner(nn.Module):

    def __init__(self,beta, n, d,):
        # beta: 学习率
        super(MetaLearner, self).__init__()
        self.n = n
        self.d = d
        #小白鼠model2的学习率已经在SGD函数中确定了。
        self.beta = beta

        # 定义一个学习类，将各个学习者的知识汇总起来。
        self.learner = Learner(self.n, self.d, meta_step=1)
        # 定义要学习的主网络的优化器
        self.optimizer = optim.Adam(self.learner.parameters(), lr=self.beta)
        # self.optimizer_ft = optim.Adam(self.learner.parameters(), lr=1e-3)

    def write_grads(self, loss, sum_grads_pi):
        # 更新梯度，梯度来源于第一个网络训练得到的梯度，梯度信息不是通过一般的反向计算的，所以我们需要这个函数来写出正确的梯度
        # 在网洛中的每个参数上注册一个钩子，用来替换当前的虚拟grad
        hooks = []
        for i, v in enumerate(self.learner.parameters()):
            def closure():
                ii = i
                return lambda grad: sum_grads_pi[ii]
            hooks.append(v.register_hook(closure()))
        # 使用求和的gradients来更新第一个网络
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 清内存
        for h in hooks:
            h.remove()

    def forward(self,data,Mdate1,Mdate2,Mdate3,epoch):

        for i in range(epoch):
            loss, grad_pi = self.learner(Mdate1,Mdate2,Mdate3)
            # 这里我们已经获得需要更新的梯度了
            # 要对需要更新的网络进行一次前向和后向传播，将梯度更新到我们需要的网络中，用钩子机制将梯度的和写入网络中
            dummy_loss, pred , h = self.learner.model_1_forward(data)
            # print(dummy_loss)
            self.write_grads(dummy_loss, grad_pi)


    def fine_tunning(self,data,epoch):
        # for i in range(epoch):
        #     self.optimizer_ft.zero_grad()
        #     ft_loss, _ = self.learner.model_1(data)
        #     ft_loss.backward()
        #     self.optimizer_ft.step()
        return self.learner.model_1_forward(data)

class Base_learner(nn.Module):

    def __init__(self,n ,d, hidden_size,l1_graph_penalty, alpha, rho, B_init, beta,epoch):

        super(Base_learner, self).__init__()
        self.n = n
        self.d = d
        self.hidden_size = hidden_size
        self.l1_graph_penalty = l1_graph_penalty
        self.alpha = alpha
        self.rho = rho
        # 学习率
        self.beta = beta
        self.epoch = epoch

        self.model = CAEModel(self.n, self.d, self.hidden_size, self.l1_graph_penalty, self.alpha, self.rho, B_init=B_init,
                                device='cpu')

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.beta)

    def optim(self,data):

        loss,W_est,h = self.model(data)

        for i in range(self.epoch):
            loss, W_est, h = self.model(data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if h <= 1e-8 and i >500:
                break

        return W_est


