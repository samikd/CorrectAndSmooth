
import pickle
import numpy as np
import pandas as pd




train_idx_weights = [0]*2708
split_idx = {}
split_idx['train'] = np.load("cora_splits/cora_train_split.npy")

for i in range(len(split_idx['train'])):
    index = split_idx['train'][i]
    train_idx_weights[index] = 1



# file_path = "coraCSbounds-2.pkl"
file_path = "/media/harsh/forUbuntu/Common_Drive/IIT_KGP/Subgroup_generalization/c and s/samik/CorrectAndSmooth/coraCSbounds-2.pkl"
gen_bound_error_dict = pickle.load(open(file_path, "rb"))

gen_bound_error_dict = {k: v for k, v in sorted(gen_bound_error_dict.items(), key=lambda item: item[1])}
print(gen_bound_error_dict)
print("___________________________________________________________")
min_gen_key = res = list(gen_bound_error_dict.keys())[0]
min_gen_value = res = gen_bound_error_dict[min_gen_key]

for key in gen_bound_error_dict.keys():
    new_val = gen_bound_error_dict[key]
    new_val = (1/new_val)*min_gen_value
    gen_bound_error_dict[key] = new_val
    train_idx_weights[key] = new_val
    
print(gen_bound_error_dict)



print(train_idx_weights)