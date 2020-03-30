# sklearn-ray
Some benchmarks for using Ray's implementation of joblib backend to distribute scikit-learn training.

**How to run:**
1) run `ray up -y blog<#nodes>.yaml`.
2) run `ray attach blog<#nodes>.yaml`
3) copy the relevant \<benchmark\>.py to the remote cluster.
4) run `RAY_ADDRESS=auto python <benchmark>.py`
