import scipy.io as scio
import numpy as np
from common import GraphDAG
from common.evaluation import MetricsDAG
from GAERun import MetaGae

# data = scio.loadmat('data/BOLD_con.mat')
# dataset = np.array(data['BOLD'])
# true_dag = np.array(data['pa']['A'])
# print(true_dag)

true_dag = np.loadtxt('data/BOLD_15.csv', delimiter=',')
data = scio.loadmat('data/BOLD_con.mat')
M_data = scio.loadmat('data/BOLD_con.mat')
dataset = np.array(data['BOLD'])
M_dataset = np.array(data['BOLD'])

CAE = MetaGae(2e-3)
CAE.meta_learn(dataset,M_dataset)

GraphDAG(CAE.causal_matrix, true_dag)
met = MetricsDAG(CAE.causal_matrix, true_dag)
print(met.metrics)