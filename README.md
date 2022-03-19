# Correct and Smooth (C&S) OGB submissions

Paper: https://arxiv.org/abs/2010.13993

This directory contains OGB submissions. All hyperparameters were tuned on the validation set with optuna, except for products, which was hand tuned. All experiments were run with a RTX 2080 TI with 11GB.

## Setup
Several Python packages are dependencies:

```
pip install julia
pip install h5py
pip install optuna
pip install torch-scatter
pip install torch-sparse
pip install torch-geometric
pip install pyg
pip install dg
pip install ogb
pip install scipy
pip install networkx
```
**NOTE**: We have not set up Julia yet, which will cause failures when the spectral embedding is required (e.g., in `python gen_models.py --dataset arxiv --model mlp --use_embeddings`).

This one is admittedly dirty, but for `ogb.nodeproppred.Evaluator` to recognise Cora, one has to modify `master.csv` found inside `site-packages/ogb/nodeproppred/` (mine is at `/Users/samdatta/miniconda/envs/gmlpipe/lib/python3.9/site-packages/ogb/nodeproppred/` - your's may be different). I did it with:

```sh
cp ogbn-cora-submission/master.csv /Users/samdatta/miniconda/envs/gmlpipe/lib/python3.9/site-packages/ogb/nodeproppred/
```

## Some Tips 
- In general, the more complex and "smooth" your GNN is, the less likely it'll be that applying the "Correct" portion helps performance. In those cases, you may consider just applying the "smooth" portion, like we do on the GAT. In almost all cases, applying the "smoothing" component will improve performance. For Linear/MLP models, applying the "Correct" portion is almost always essential for obtaining good performance.

- In a similar vein, an improvement of performance of your model may not correspond to an improvement after applying C&S. Considering that C&S learns no parameters over your data, our intuition is that C&S "levels" the playing field, allowing models that learn interesting features to shine (as opposed to learning how to be smooth).
     - Even though GAT (73.57) is outperformed by GAT + labels (73.65), when we apply C&S, we see that GAT + C&S (73.86) performs better than GAT + labels + C&S (~73.70) , 
     - Even though a 6 layer GCN performs on par with a 2 layer GCN with Node2Vec features, C&S improves performance of the 2 layer GCN with Node2Vec features substantially more.
     - Even though MLP + Node2Vec outperforms MLP + Spectral in both arxiv and products, the performance ordering flips after we apply C&S.
     - On Products, the MLP (74%) is substantially outperformed by ClusterGCN (80%). However, MLP + C&S (84.1%) substantially outperforms ClusterGCN + C&S (82.4%).

- In general, autoscale works more reliably than fixedscale, even though fixedscale may make more sense...

## Cora
**NOTE**: The current version of the code does not aim to reproduce the paper exactly - while we use (inside `adapter.py`, which downloads Cora from DGL and packages it for a submission to OGB) the standard train-val-test split of 140:500:1000, the C&S paper reportedly used a 60:20:20 _random_ split (see Sec. 3).

### Experiments
- `python run_experiments.py --dataset cora --method lp` gives val/test accuracies of 0.698/0.707.		->D
- `python gen_models.py --dataset arxiv --model mlp --use_embeddings` gives val/test accuracies of 76.6600 ±	->D 0.8796/76.9600 ± 1.0384.

**NOTE** Other models (e.g., GAT+C&S) have not been tested yet. Most likely, those pathways will throw errors (the original code has quite a bit of code-duplication - not all pathways were modified to accept Cora). However, fixing them should be routine.

## Arxiv

### Label Propagation (0 params):
```
python run_experiments.py --dataset arxiv --method lp
Valid acc:  0.7018356320681902
Test acc: 0.683496903483324


Valid acc: 0.7013658176448874
Test acc: 0.6832294302820814
```
```
python run_experiments.py --dataset cora --method lp		->DONE

Valid acc: 0.968
Test acc: 0.707
```
### Plain Linear + C&S (5160 params, 52.5% base accuracy)
```
python gen_models.py --dataset arxiv --model plain --epochs 1000    
python run_experiments.py --dataset arxiv --method plain

Valid acc -> Test acc
Args []: 73.00 ± 0.01 -> 71.26 ± 0.01
```

```
python gen_models.py --dataset cora --model plain --epochs 1000     D
python run_experiments.py --dataset cora --method plain		D

All runs:
Highest Train: 100.0000 ± 0.0000
Highest Valid: 51.2600 ± 0.3893
  Final Train: 100.0000 ± 0.0000
   Final Test: 47.4400 ± 0.5641
   
   
Valid acc -> Test acc
Args []: 74.34 ± 0.21 -> 75.17 ± 0.09



_________________________________________________________________________

python run_experiments.py --dataset cora --method plain_gen_bound		D

Valid acc -> Test acc
Args []: 71.84 ± 0.67 -> 74.18 ± 0.62

```

### Linear + C&S (15400 params, 70.11% base accuracy)
```
python gen_models.py --dataset arxiv --model linear --use_embeddings --epochs 1000 
python run_experiments.py --dataset arxiv --method linear

Valid acc -> Test acc
Args []: 73.68 ± 0.04 -> 72.22 ± 0.02;
```
```
python gen_models.py --dataset cora --model linear --use_embeddings --epochs 1000 	D
python run_experiments.py --dataset cora --method linear				D



All runs:
Highest Train: 100.0000 ± 0.0000
Highest Valid: 72.9600 ± 0.4695
  Final Train: 99.7857 ± 0.3450
   Final Test: 71.7000 ± 0.3055
   
Valid acc -> Test acc
Args []: 75.78 ± 0.06 -> 75.36 ± 0.08

_________________________________________________________________________

python run_experiments.py --dataset cora --method linear_gen_bound		D

Valid acc -> Test acc
Args []: 74.18 ± 0.82 -> 75.29 ± 0.40


```

### MLP + C&S (175656 params, 71.44% base accuracy)
```
python gen_models.py --dataset arxiv --model mlp --use_embeddings
python run_experiments.py --dataset arxiv --method mlp

Valid acc -> Test acc
Args []: 73.91 ± 0.15 -> 73.12 ± 0.12
```

```
python gen_models.py --dataset cora --model mlp --use_embeddings	D
python run_experiments.py --dataset cora --method mlp	


All runs:
Highest Train: 100.0000 ± 0.0000
Highest Valid: 76.8800 ± 1.2300
  Final Train: 100.0000 ± 0.0000
   Final Test: 76.9000 ± 1.0965

Valid acc -> Test acc
Args []: 78.50 ± 1.33 -> 79.10 ± 1.28

_________________________________________________________________________

python run_experiments.py --dataset cora --method mlp_gen_bound 		D

Valid acc -> Test acc
Args []: 78.50 ± 1.33 -> 79.10 ± 1.28



```

### GAT + C&S (1567000 params, 73.56% base accuracy)
```
cd gat && python gat.py --use-norm
Average epoch time: 21.718203401565553, Test acc: 0.6538896775919182
Runned 10 times
Val Accs: [0.6605255209906372, 0.6424376656934796, 0.6559951676230746, 0.6449880868485519, 0.6551562132957481, 0.6409275479042921, 0.6187120373166884, 0.6712976945535085, 0.6283432329943958, 0.6312292358803987]
Test Accs: [0.667160463345884, 0.6027817212929243, 0.6717075077670103, 0.6384791062280106, 0.6247145237948275, 0.6592597164784066, 0.578585684011275, 0.6555356665226426, 0.6484579141205276, 0.6538896775919182]
Average val accuracy: 0.6449612403100775 ± 0.015290282129320602
Average test accuracy: 0.6400571981153427 ± 0.02830663270099123
[768, 768, 98304, 768, 768, 589824, 120, 120, 92160, 98304, 589824, 92160, 768, 768, 768, 768, 40]
Number of params: 1567000



cd .. && python run_experiments.py --dataset arxiv --method gat
Valid acc -> Test acc
Args []: 67.27 ± 1.39 -> 66.25 ± 2.90


Valid acc -> Test acc
Args []: 74.84 ± 0.07 -> 73.86 ± 0.14
```

```
cd gat && python gat.py --use-norm --dataset cora
cd .. && python run_experiments.py --dataset cora --method gat


Average epoch time: 0.3995665550231934, Test acc: 0.262
Runned 10 times
**************************************************
Average epoch time: 0.39290672254562375, Test acc: 0.791
Runned 10 times
Val Accs: [0.772, 0.77, 0.768, 0.788, 0.78, 0.78, 0.79, 0.746, 0.774, 0.76]
Test Accs: [0.788, 0.775, 0.771, 0.804, 0.794, 0.802, 0.797, 0.772, 0.772, 0.791]
Average val accuracy: 0.7727999999999999 ± 0.012432216214336054
Average test accuracy: 0.7866000000000002 ± 0.012362847568420484
[768, 768, 1100544, 768, 768, 589824, 21, 21, 16128, 1100544, 589824, 16128, 768, 768, 768, 768, 7]
Number of params: 3419185

Valid acc -> Test acc
Args []: 78.18 ± 1.14 -> 80.38 ± 1.15
```

### Notes
As opposed to the paper's results, which only use spectral embeddings, here we use spectral *and* diffusion embeddings, which we find improves Arxiv performance.

## Products

### Label Propagation (0 params):
```
python run_experiments.py --dataset products --method lp 

Valid acc:  0.9090608549703736
Test acc: 0.7434145274640762
```

### Plain Linear + C&S (4747 params, 47.73% base accuracy)
```
python gen_models.py --dataset products --model plain --epochs 1000 --lr 0.1
python run_experiments.py --dataset products --method plain

Valid acc -> Test acc
Args []: 91.03 ± 0.01 -> 82.54 ± 0.03
```

### Linear + C&S (10763 params, 50.05% base accuracy)
```
python gen_models.py --dataset products --model linear --use_embeddings --epochs 1000 --lr 0.1
python run_experiments.py --dataset products --method linear

Valid acc -> Test acc
Args []: 91.34 ± 0.01 -> 83.01 ± 0.01
```

### MLP + C&S (96247 params, 63.41% base accuracy)
```
python gen_models.py --dataset products --model mlp --hidden_channels 200 --use_embeddings
python run_experiments.py --dataset products --method mlp

Valid acc -> Test acc
Args []: 91.47 ± 0.09 -> 84.18 ± 0.07
```
