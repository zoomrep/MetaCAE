import numpy as np
from common import GraphDAG
from common.evaluation import MetricsDAG
from GAERun import MetaGae
from datasets.simulator import DAG, IIDSimulation
import os
import scipy.io as scio
import matplotlib.pyplot as plt

# X = np.loadtxt('data/sim8/sim1.csv', delimiter=',')
# true_dag = np.loadtxt('data/sub1_target.csv', delimiter=',')
# ER模拟数据集

# D = [5,10,20]
# I = [200,500]
# for k in range(2):
#     for j in D:
#         for i in I:
#             weighted_random_dag = DAG.erdos_renyi(n_nodes=j, n_edges=(2 * j), weight_range=(0.5, 2.0))#seed=?
#             M_dataset = IIDSimulation(W=weighted_random_dag,n=j, method='nonlinear', sem_type='mlp')
#             dataset = IIDSimulation(W=weighted_random_dag, n=j, method='nonlinear', sem_type='mlp')
#             true_dag, X = dataset.B, dataset.X
#             _, m_dataset = M_dataset.B, M_dataset.X
#             n = MetaGae(beta=2e-3)
#             n.meta_learn(X,m_dataset)
#             GraphDAG(n.causal_matrix, true_dag)
#             met = MetricsDAG(n.causal_matrix, true_dag)
#             print(met.metrics)
#             print(i,j,k)

# weighted_random_dag = DAG.erdos_renyi(n_nodes=5, n_edges=10, weight_range=(0.5, 2.0), seed=1)
# dataset = IIDSimulation(W=weighted_random_dag, n=1000, method='nonlinear', sem_type='quadratic')
# true_dag, X = dataset.B, dataset.X

# n = MetaGae(beta=2e-3)
# n.learn(X)
# GraphDAG(n.causal_matrix, true_dag)
# plt.imshow(n.causal_matrix, interpolation='nearest', cmap='RdBu',vmin=-1,vmax=1.75)
# plt.colorbar(shrink=.92)
# plt.xticks(())
# plt.yticks(())
# plt.show()
# met = MetricsDAG(n.causal_matrix, true_dag)
# print(met.metrics)


# path = r'data/RestingState_MTL/individual_left_mtl_reduced'
# files = os.listdir(path)
# M_data_1 = np.loadtxt('data/RestingState_MTL/individual_left_mtl_reduced/mtl_L_sub_8.txt')
# M_data_2 = np.loadtxt('data/RestingState_MTL/individual_left_mtl_reduced/mtl_L_sub_9.txt')
# M_data_3 = np.loadtxt('data/RestingState_MTL/individual_left_mtl_reduced/mtl_L_sub_10.txt')
# M_data = np.concatenate((M_data_1,M_data_2,M_data_3))
#
# true_dag = np.loadtxt('data/RestingState_MTL/sub1_target.csv', delimiter=',')
# result = np.zeros_like(true_dag)
# for file in files:
#     f_name = str(file)
#     tr = '/'
#     filename = path + tr + f_name
#     print(f_name)
#     print(filename)
#     X = np.loadtxt(filename)
#
#     n = MetaGae(beta=2e-3)
#     n.meta_learn(X,M_data)
#     GraphDAG(n.causal_matrix, true_dag)
#     result[n.causal_matrix != 0] += 1

# np.savetxt("result/left.txt",result)
# print(result)
#     met = MetricsDAG(n.causal_matrix, true_dag)
#     print(met.metrics)



# true_dag = np.loadtxt('data/RestingState_MTL/sub1_target.csv', delimiter=',')
# for i in range(2,22):
#     path_1 = "data/RestingState_MTL/individual_right_mtl_reduced/mtl_R_sub_"+str(i)+".txt"
#     path_2 = "data/RestingState_MTL/individual_right_mtl_reduced/mtl_R_sub_"+str(i+1)+".txt"
#     path_3 = "data/RestingState_MTL/individual_right_mtl_reduced/mtl_R_sub_"+str(i+2)+".txt"
#     M_data_1 = np.loadtxt(path_1)
#     M_data_2 = np.loadtxt(path_2)
#     M_data_3 = np.loadtxt(path_3)
#     M_data = np.concatenate((M_data_1, M_data_2, M_data_3))
#     X = np.loadtxt('data/RestingState_MTL/individual_right_mtl_reduced/mtl_R_sub_2.txt')
#
#     n = MetaGae(beta=2e-3)
#     n.meta_learn(X,M_data)
#     GraphDAG(n.causal_matrix, true_dag)
#     met = MetricsDAG(n.causal_matrix, true_dag)
#     print(met.metrics)
# # print('----------------------------------------------left结束-------------------------------------------------')



# data = scio.loadmat('data/BOLD_con.mat')
# M_data = scio.loadmat('data/BOLD_con.mat')
# dataset = np.array(data['BOLD'])
# M_dataset = np.array(data['BOLD'])
# Meta_X = M_dataset[0:200, :]
# X = dataset[0:200, :]
# CAE = MetaGae(2e-3)
# CAE.learn(X)

# GraphDAG(CAE.causal_matrix, true_dag)
# met = MetricsDAG(CAE.causal_matrix, true_dag)
# print(met.metrics)


# import os
# import sys
#
# # make a copy of original stdout route
# stdout_backup = sys.stdout
#
# log_file = open('result/output.txt', "w")
#
# # redirect print output to log file
# sys.stdout = log_file # 将系统输出切换至log_file
#
# print ("Now all print info will be written to message.log")
# # any command line that you will execute


true_dag = np.loadtxt('data/sub1_target.csv', delimiter=',')
data = scio.loadmat('data/sim10.mat')
M_data = scio.loadmat('data/sim10.mat')
dataset = np.array(data['ts'])
M_dataset = np.array(data['ts'])
Meta_X1 = M_dataset[0:600, :]
# Meta_X2 = M_dataset[200,400 :]
# Meta_X3 = M_dataset[400:600, :]

for i in range(0,50):
    X = dataset[(0+i*200):(200+i*200),:]
    n = MetaGae(beta=2e-3)
    # n.meta_learn(X,Meta_X1)
    n.learn(X)
    # GraphDAG(n.causal_matrix, true_dag)
    met = MetricsDAG(n.causal_matrix, true_dag)
    print(met.metrics)

print('----------------------------------------------sim10结束-------------------------------------------------')

# true_dag = np.loadtxt('data/sim17_target.csv', delimiter=',')
# data = scio.loadmat('data/sim2.mat')
# M_data = scio.loadmat('data/sim11.mat')
# dataset = np.array(data['ts'])
# M_dataset = np.array(M_data['ts'])
# Meta_X1 = M_dataset[5000:5600, :]
#
# for i in range(0,50):
#     X = dataset[(0+i*200):(200+i*200),:]
#     n = MetaGae(beta=3e-2)
#     n.meta_learn(X,Meta_X1)
#     # n.learn(X)
#     GraphDAG(n.causal_matrix, true_dag)
#     met = MetricsDAG(n.causal_matrix, true_dag)
#     print(met.metrics)
#
# print('----------------------------------------------sim2结束-------------------------------------------------')

#
# import os
# import sys
#
# # make a copy of original stdout route
# stdout_backup = sys.stdout
#
# log_file = open('result/output.txt', "w")
#
# # redirect print output to log file
# sys.stdout = log_file # 将系统输出切换至log_file
#
# print ("Now all print info will be written to message.log")
# # any command line that you will execute
#
# true_dag = np.loadtxt('data/sim3_target.csv', delimiter=',')
# data = scio.loadmat('data/sim3.mat')
# M_data = scio.loadmat('data/sim3.mat')
# dataset = np.array(data['ts'])
# M_dataset = np.array(M_data['ts'])
# # Meta_X1 = M_dataset[0:600, :]
# for i in range(0,42):
#     Meta_X1 = M_dataset[(0+i*200): (600+i*200)]
#     X = dataset[0:200]
#     n = MetaGae(beta=2e-3)
#     n.meta_learn(X,Meta_X1)
#     # n.learn(X)
#     GraphDAG(n.causal_matrix, true_dag)
#     met = MetricsDAG(n.causal_matrix, true_dag)
#     print(met.metrics)
# # for i in range(0,50):
# #     X = dataset[(0+i*200):(200+i*200),:]
# #     n = MetaGae(beta=2e-3)
# #     # n.meta_learn(X,Meta_X1)
# #     n.learn(X)
# #     # GraphDAG(n.causal_matrix, true_dag)
# #     met = MetricsDAG(n.causal_matrix, true_dag)
# #     print(met.metrics)
#
# print('----------------------------------------------sim3结束-------------------------------------------------')
#
#
# log_file.close()
# # restore the output to initial pattern
# sys.stdout = stdout_backup #将系统输出切换回console
#
# print ("Now this will be presented on screen")

#
# true_dag = np.loadtxt('data/sub1_target.csv', delimiter=',')
# data = scio.loadmat('data/sim1.mat')
# M_data = scio.loadmat('data/sim10.mat')
# dataset = np.array(data['ts'])
# M_dataset = np.array(data['ts'])
# Meta_X1 = M_dataset[0:600, :]
# # Meta_X2 = M_dataset[200,400 :]
# # Meta_X3 = M_dataset[400:600, :]
#
# for i in range(0,50):
#     X = dataset[(0+i*200):(200+i*200),:]
#     n = MetaGae(beta=2e-3)
#     # n.meta_learn(X,Meta_X1)
#     n.learn(X)
#     # GraphDAG(n.causal_matrix, true_dag)
#     met = MetricsDAG(n.causal_matrix, true_dag)
#     print(met.metrics)
#
# print('----------------------------------------------sim1结束-------------------------------------------------')
# true_dag = np.loadtxt('data/sub1_target.csv', delimiter=',')
# data = scio.loadmat('data/sim8.mat')
# M_data = scio.loadmat('data/sim8.mat')
# dataset = np.array(data['ts'])
# M_dataset = np.array(data['ts'])
# Meta_X1 = M_dataset[4600:5200, :]
# # Meta_X2 = M_dataset[200,400 :]
# # Meta_X3 = M_dataset[400:600, :]
#
# for i in range(0,50):
#     # Meta_X1 = M_dataset[(0 + i * 200): (600 + i * 200)]
#     # X = dataset[0:200 ,:]
#     X = dataset[(0+i*200):(200+i*200),:]
#     n = MetaGae(beta=2e-3)
#     # n.meta_learn(X,Meta_X1)
#     n.learn(X)
#     # GraphDAG(n.causal_matrix, true_dag)
#     met = MetricsDAG(n.causal_matrix, true_dag)
#     print(met.metrics)
#
# print('----------------------------------------------sim8结束-------------------------------------------------')
# true_dag = np.loadtxt('data/sub1_target.csv', delimiter=',')
# data = scio.loadmat('data/sim21.mat')
# M_data = scio.loadmat('data/sim10.mat')
# dataset = np.array(data['ts'])
# M_dataset = np.array(data['ts'])
# Meta_X1 = M_dataset[0:600, :]
# # Meta_X2 = M_dataset[200,400 :]
# # Meta_X3 = M_dataset[400:600, :]
#
# for i in range(0,50):
#     # Meta_X1 = M_dataset[(0 + i * 200): (600 + i * 200)]
#     # X = dataset[0:200, :]
#     X = dataset[(0+i*200):(200+i*200),:]
#     n = MetaGae(beta=2e-3)
#     # n.meta_learn(X,Meta_X1)
#     n.learn(X)
#     # GraphDAG(n.causal_matrix, true_dag)
#     met = MetricsDAG(n.causal_matrix, true_dag)
#     print(met.metrics)
#
# print('----------------------------------------------sim21结束-------------------------------------------------')
# true_dag = np.loadtxt('data/sub1_target.csv', delimiter=',')
# data = scio.loadmat('data/sim22.mat')
# M_data = scio.loadmat('data/sim22.mat')
# dataset = np.array(data['ts'])
# M_dataset = np.array(data['ts'])
# Meta_X1 = M_dataset[4200:4800, :]
# # Meta_X2 = M_dataset[200,400 :]
# # Meta_X3 = M_dataset[400:600, :]
#
# for i in range(0,50):
#     # Meta_X1 = M_dataset[(0 + i * 200): (600 + i * 200)]
#     # X = dataset[0:200, :]
#     X = dataset[(0+i*200):(200+i*200),:]
#     n = MetaGae(beta=2e-3)
#     # n.meta_learn(X,Meta_X1)
#     n.learn(X)
#     # GraphDAG(n.causal_matrix, true_dag)
#     met = MetricsDAG(n.causal_matrix, true_dag)
#     print(met.metrics)
#
# print('----------------------------------------------sim22结束-------------------------------------------------')
# true_dag = np.loadtxt('data/sub1_target.csv', delimiter=',')
# data = scio.loadmat('data/sim23.mat')
# M_data = scio.loadmat('data/sim23.mat')
# dataset = np.array(data['ts'])
# M_dataset = np.array(data['ts'])
# Meta_X1 = M_dataset[6400:7000, :]
# # Meta_X2 = M_dataset[200,400 :]
# # Meta_X3 = M_dataset[400:600, :]
#
# for i in range(0,50):
#     # Meta_X1 = M_dataset[(0 + i * 200): (600 + i * 200)]
#     # X = dataset[0:200, :]
#     X = dataset[(0+i*200):(200+i*200),:]
#     n = MetaGae(beta=2e-3)
#     n.meta_learn(X,Meta_X1)
#     # n.learn(X)
#     # GraphDAG(n.causal_matrix, true_dag)
#     met = MetricsDAG(n.causal_matrix, true_dag)
#     print(met.metrics)
#
# print('----------------------------------------------sim23结束-------------------------------------------------')
# true_dag = np.loadtxt('data/sub1_target.csv', delimiter=',')
# data = scio.loadmat('data/sim24.mat')
# M_data = scio.loadmat('data/sim10.mat')
# dataset = np.array(data['ts'])
# M_dataset = np.array(data['ts'])
# Meta_X1 = M_dataset[600:1200, :]
# # Meta_X2 = M_dataset[200,400 :]
# # Meta_X3 = M_dataset[400:600, :]
#
# for i in range(0,50):
#     # Meta_X1 = M_dataset[(0 + i * 200): (600 + i * 200)]
#     # X = dataset[0:200, :]
#     X = dataset[(0+i*200):(200+i*200),:]
#     n = MetaGae(beta=2e-3)
#     n.meta_learn(X,Meta_X1)
#     # n.learn(X)
#     # GraphDAG(n.causal_matrix, true_dag)
#     met = MetricsDAG(n.causal_matrix, true_dag)
#     print(met.metrics)
#
# print('----------------------------------------------sim24结束-------------------------------------------------')
# true_dag = np.loadtxt('data/sub13_target.csv', delimiter=',')
# data = scio.loadmat('data/sim13.mat')
# M_data = scio.loadmat('data/sim13.mat')
# dataset = np.array(data['ts'])
# M_dataset = np.array(data['ts'])
# Meta_X1 = M_dataset[2800:3400, :]
# # Meta_X2 = M_dataset[200,400 :]
# # Meta_X3 = M_dataset[400:600, :]
#
# for i in range(0,50):
#     # Meta_X1 = M_dataset[(0 + i * 200): (600 + i * 200)]
#     # X = dataset[0:200, :]
#     X = dataset[(0+i*200):(200+i*200),:]
#     n = MetaGae(beta=2e-3)
#     n.meta_learn(X,Meta_X1)
#     # n.learn(X)
#     # GraphDAG(n.causal_matrix, true_dag)
#     met = MetricsDAG(n.causal_matrix, true_dag)
#     print(met.metrics)
#
# print('----------------------------------------------sim13结束-------------------------------------------------')
#
# true_dag = np.loadtxt('data/sub14_target.csv', delimiter=',')
# data = scio.loadmat('data/sim14.mat')
# M_data = scio.loadmat('data/sim14.mat')
# dataset = np.array(data['ts'])
# M_dataset = np.array(data['ts'])
# Meta_X1 = M_dataset[0:600, :]
# # Meta_X2 = M_dataset[200,400 :]
# # Meta_X3 = M_dataset[400:600, :]
#
# for i in range(0,50):
#     # Meta_X1 = M_dataset[(0 + i * 200): (600 + i * 200)]
#     # X = dataset[0:200, :]
#     X = dataset[(0+i*200):(200+i*200),:]
#     n = MetaGae(beta=2e-3)
#     # n.meta_learn(X,Meta_X1)
#     n.learn(X)
#     # GraphDAG(n.causal_matrix, true_dag)
#     met = MetricsDAG(n.causal_matrix, true_dag)
#     print(met.metrics)
#
# print('----------------------------------------------sim14结束-------------------------------------------------')
# true_dag = np.loadtxt('data/sub1_target.csv', delimiter=',')
# data = scio.loadmat('data/sim15.mat')
# M_data = scio.loadmat('data/sim15.mat')
# dataset = np.array(data['ts'])
# M_dataset = np.array(data['ts'])
# Meta_X1 = M_dataset[4400:5000, :]
# # Meta_X2 = M_dataset[200,400 :]
# # Meta_X3 = M_dataset[400:600, :]
#
# for i in range(0,50):
#     # Meta_X1 = M_dataset[(0 + i * 200): (600 + i * 200)]
#     # X = dataset[0:200, :]
#     X = dataset[(0+i*200):(200+i*200),:]
#     n = MetaGae(beta=2e-3)
#     # n.meta_learn(X,Meta_X1)
#     n.learn(X)
#     # GraphDAG(n.causal_matrix, true_dag)
#     met = MetricsDAG(n.causal_matrix, true_dag)
#     print(met.metrics)
#
# print('----------------------------------------------sim15结束-------------------------------------------------')
# true_dag = np.loadtxt('data/sub16_target.csv', delimiter=',')
# data = scio.loadmat('data/sim16.mat')
# M_data = scio.loadmat('data/sim16.mat')
# dataset = np.array(data['ts'])
# M_dataset = np.array(data['ts'])
# Meta_X1 = M_dataset[7000:7600, :]
# # Meta_X2 = M_dataset[200,400 :]
# # Meta_X3 = M_dataset[400:600, :]
#
# for i in range(0,50):
#     # Meta_X1 = M_dataset[(0 + i * 200): (600 + i * 200)]
#     # X = dataset[0:200, :]
#     X = dataset[(0+i*200):(200+i*200),:]
#     n = MetaGae(beta=2e-3)
#     # n.meta_learn(X,Meta_X1)
#     n.learn(X)
#     # GraphDAG(n.causal_matrix, true_dag)
#     met = MetricsDAG(n.causal_matrix, true_dag)
#     print(met.metrics)
#
# print('----------------------------------------------sim16结束-------------------------------------------------')


# log_file.close()
# # restore the output to initial pattern
# sys.stdout = stdout_backup #将系统输出切换回console
#
# print ("Now this will be presented on screen")