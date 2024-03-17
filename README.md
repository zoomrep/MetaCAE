## Dependencies
```
Python 3

numpy
scipy
networkx
PyTorch 1.0.1 or later
matplotlib
pandas
```

## Usage
See `MGtest.py` for a simple script demonstrating usage.

In general, using this package looks like:
```python 
from MCAERun import MetaCAE


true_dag = np.loadtxt()
data = np.loadtxt()
M_data = np.loadtxt()

n = MetaCAE(beta=2e-3)
# data are the original input
# M_data are the learning task
n.meta_learn(data,M_data)
GraphDAG(n.causal_matrix, true_dag)
met = MetricsDAG(n.causal_matrix, true_dag)
```



## Citation

Our code is developed based on gCastle in [trustworthyAI](https://github.com/huawei-noah/trustworthyAI/tree/master), which is a causal structure learning toolchain developed by [Huawei Noah's Ark Lab](https://www.noahlab.com.hk/#/home). We use the evaluation metrics to evaluate our method and baseline methods.

```
@misc{zhang2021gcastle,
  title={gCastle: A Python Toolbox for Causal Discovery}, 
  author={Keli Zhang and Shengyu Zhu and Marcus Kalander and Ignavier Ng and Junjian Ye and Zhitang Chen and Lujia Pan},
  year={2021},
  eprint={2111.15155},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```