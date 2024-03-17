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

        self.model_1 = CAEModel(self.n, self.d, hidden_size=20, l1_graph_penalty=5e-2, alpha=20.0, rho=2e-4, B_init=None, device='cpu')
        self.model_2 = CAEModel(self.n, self.d, hidden_size=20, l1_graph_penalty=5e-2, alpha=20.0, rho=2e-4, B_init=None, device='cpu')

        self.optim = optim.Adam(self.model_2.parameters(),lr=2e-3)

    def parameters(self):
        return self.model_1.parameters()

    def update_pi(self):
        for m_from, m_to in zip(self.model_1.modules(), self.model_2.modules()):
            if isinstance(m_to, nn.Linear) or isinstance(m_to, nn.Conv2d) or isinstance(m_to, nn.BatchNorm2d) :
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()

    def forward(self,data):
        # 先复制网络的参数
        self.update_pi()

        for i in range(self.meta_step):
            # 前向传播
            loss, _, h = self.model_2(data)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        # 然后计算元梯度并返回，一个epoch一个，上面可以算是训练过程
        # 现在为测试过程：
        test_loss, _, h = self.model_2(data)
        # 这里将creat_graph设置为True用来二次反向传播
        grads_pi = autograd.grad(test_loss, self.model_2.parameters(), create_graph=True, allow_unused=True)

        return test_loss, grads_pi

    def model_1_forward(self,data):
        return self.model_1(data)

# forward参数为训练data和epoch
# fine_tunning的参数为验证data
class MetaLearner(nn.Module):

    def __init__(self,beta, n, d,
                 rho_thres:'float' = 1e+30,
                 rho_multiply:'float' = 2.0,
                 h_tol:'float' = 1e-8):
        # beta: 学习率
        super(MetaLearner, self).__init__()
        self.n = n
        self.d = d
        #小白鼠model2的学习率已经在SGD函数中确定了。
        self.beta = beta
        self.rho_thres = rho_thres
        self.rho_multiply = rho_multiply
        self.h_hot = h_tol
        # 定义一个学习类，将各个学习者的知识汇总起来。
        self.learner = Learner(self.n, self.d, meta_step=1)
        # 定义要学习的主网络的优化器
        self.optimizer = optim.Adam(self.learner.parameters(), lr=self.beta)
        self.optimizer_ft = optim.Adam(self.learner.parameters(), lr=1e-3)

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

    def forward(self,data,epoch):

        for i in range(epoch):
            sum_grads_pi = None
            loss, grad_pi = self.learner(data)
            if sum_grads_pi is None:
                sum_grads_pi = grad_pi
            else:  # accumulate all gradients from different episode learner
                # for i in zip(sum_grads_pi):
                #     if i is None:
                #         i = torch.zeros(0)
                # for j in zip(grad_pi):
                #     if j is None:
                #         j = torch.zeros(0)
                sum_grads_pi = [torch.add(i, j) for i, j in zip(sum_grads_pi, grad_pi) ]
            # 这里我们已经获得需要更新的梯度了
            # 要对需要更新的网络进行一次前向和后向传播，将梯度更新到我们需要的网络中，用钩子机制将梯度的和写入网络中
            dummy_loss, pred , h = self.learner.model_1_forward(data)
            # print(dummy_loss)
            self.write_grads(dummy_loss, sum_grads_pi)
            # if self.learner.model_1.rho < self.rho_thres:
            #     self.learner.model_1.rho *= self.rho_multiply
            #
            # self.learner.model_1.alpha += self.learner.model_1.rho * self.learner.model_1.h

            # 提前结束
            # if self.learner.model_1.h <= self.h_hot and i > 3:
            #     break



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


