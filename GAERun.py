import numpy as np
# from Metatrain import MetaLearner, Base_learner
import torch

from Metatrain_V2 import MetaLearner, Base_learner
from GAEnet import GAEModel
from post_preprocess import postprocess
from common import BaseLearner,Tensor
from common.utils import is_cuda_available, set_seed

class MetaGae(BaseLearner):
    def __init__(self,beta):
        super(MetaGae, self).__init__()
        self.beta = beta


    def learn(self, data, columns=None, **kwargs):
        # 设置并运行GAE
        set_seed(1)
        X = Tensor(data, columns=columns)
        # print(X.dtype)
        self.n, self.d = X.shape[:2]
        self.learner = Base_learner(self.n, self.d, hidden_size=5, l1_graph_penalty=5e-2, alpha=20.0, rho=2e-4,B_init=None, beta=self.beta,epoch=5000)
        W_est = self.learner.optim(X)
        # print(W_est)

        # 暂时是cpu的写法  后处理的过程，估计解和计算结果
        B_processed = postprocess(W_est.detach().numpy())
        B_result = (B_processed != 0).astype(int)

        causal_matrix = B_result
        self.causal_matrix = Tensor(causal_matrix, index=X.columns,
                                    columns=X.columns)


    def meta_learn(self,data,M_data):
        set_seed(1)
        X = Tensor(data)
        Mdata = Tensor(M_data)
        M_data1 = Mdata[0:421,:] #real 420 sim 200
        M_data2 = Mdata[421:842,:]
        M_data3 = Mdata[842:1263,:]
        self.mn, self.md = M_data1.shape[:2]
        self.n, self.d = X.shape[:2]

        self.learner = MetaLearner(beta=self.beta, n=self.mn, d=self.md)
        self.learner(X,M_data1,M_data2,M_data3, epoch=5000)
        _, W_est, h= self.learner.fine_tunning(data,1)
        W_est = torch.transpose(W_est,0,1)
        # print(W_est)
        self.learner = Base_learner(self.n, self.d, hidden_size=7, l1_graph_penalty=5e-2, alpha=10.0, rho=2e-4, B_init=W_est,
                                    beta=self.beta, epoch=1000)
        W_est = self.learner.optim(X)
        # print(W_est)

        B_processed = postprocess(W_est.detach().numpy())
        B_result = (B_processed != 0).astype(int)

        causal_matrix = B_result
        self.causal_matrix = Tensor(causal_matrix, index=X.columns,
                                    columns=X.columns)






