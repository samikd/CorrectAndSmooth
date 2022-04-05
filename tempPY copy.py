
import pickle
import numpy as np
import pandas as pd

# file_path = "coraCSbounds-2.pkl"
file_path = "/media/harsh/forUbuntu/Common_Drive/IIT_KGP/Subgroup_generalization/c and s/samik/CorrectAndSmooth/coraCSbounds-2.pkl"
gen_bound_error_dict = pickle.load(open(file_path, "rb"))

gen_bound_error_dict = {k: v for k, v in sorted(gen_bound_error_dict.items(), key=lambda item: item[1])}
print(gen_bound_error_dict)