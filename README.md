Basic commands to run the approximate tree containment problem sovlers:

**Combine-GNN**:
    ```python tree_containment_CombineGNN.py --config=configs/CombineGNN.py```

Baselines:
- python ```baselines/tree_containment_baseline_xgboost.py --config=configs/baseline_XGBoost.yaml```
- python ```baselines/tree_containment_baseline_gnn.py --config=configs/baseline_GNN.yaml``` 

All results are saved in JSON format in the corresponding folder

---

Parallel experiments (using ray) can be launched using scripts in the `run` folder

---
Exact algorithm for tree tree containment problem (BOTCH) is located in `TreeWidthTreeContainment` folder