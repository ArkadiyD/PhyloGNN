Basic commands to run the approximate tree containment problem solvers:

**Combine-GNN**:
    ```python tree_containment_CombineGNN.py --config=configs/CombineGNN.yaml```

Baselines:
- python ```baselines/tree_containment_baseline_xgboost.py --config=configs/baseline_XGBoost.yaml```
- python ```baselines/tree_containment_baseline_gnn.py --config=configs/baseline_GNN.yaml``` 

All results are saved in JSON format in the corresponding folder

---

Parallel experiments (using ray) can be launched using scripts in the `run` folder

---
Exact algorithm for tree tree containment problem (BOTCH) is located in `TreeWidthTreeContainment` folder

---
Citation:
```
@article{
dushatskiy2024solving,
title={Solving the Tree Containment Problem Using Graph Neural Networks},
author={Arkadiy Dushatskiy and Esther Julien and Leen Stougie and Leo van Iersel},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2024},
url={https://openreview.net/forum?id=nK5MazeIpn},
note={}
}
```
