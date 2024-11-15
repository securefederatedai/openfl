# PyTorch tutorial for FedCurv Federated Learning method on Histology dataset

To show results on non-iid data distribution, this tutorial contains shard descriptor with custom data splitter where data is split log-normally. Federation consists of 8 envoys.

Your Python environment must have OpenFL installed.

1. Run Director instance:
```
cd director
bash start_director.sh
```

2. In a separate terminal, execute:
```
cd envoy
bash populate_envoys.sh # This creates all envoys folders in current directory
bash start_envoys.sh # This launches all envoy instances 
```

3. In a separate terminal, launch a Jupyter Lab:
```
cd workspace
jupyter lab
```

4. Open your browser at corresponding port and open `pytorch_histology.ipynb` from Jupyter web interface. 

5. Execute all cells in order.
