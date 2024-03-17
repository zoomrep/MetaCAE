from common import BaseLearner, Tensor
import torch


class MetaTrainer(BaseLearner):

    def __init__(self, B_init=None,
                 alpha=5.0,
                 rho=2e-3,
                 l1_graph_penalty=0,
                 lr = 1e-3,
                 num_iter=1e+4,
                 meta_step = 5,
                 checkpoint_iter=500):

        super(MetaTrainer, self).__init__()
        self.B_init = B_init
        self.alpha = alpha
        self.rho = rho
        self.l1_graph_penalty = l1_graph_penalty
        self.lr = lr
        self.num_iter = num_iter
        self.meta_step = meta_step
        self.chechpoint_iter = checkpoint_iter


    def train(self):
        pass






