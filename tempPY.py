
import numpy as np
import pandas as pd

list_of_nodes = [x for x in range(0,2708)]
list_of_nodes = pd.DataFrame(list_of_nodes)

train, validate, test = np.split(list_of_nodes.sample(frac=1, random_state=42), [int(.6*len(list_of_nodes)), int(.8*len(list_of_nodes))])

train_list = []
for i in range(len(train)):
    train_list.append(train.iloc[i][0])    
train_list = np.array(train_list)

print("train = ",train_list)
np.save('cora_splits/cora_train_split.npy', train_list)

validate_list = []
for i in range(len(validate)):
    validate_list.append(validate.iloc[i][0])    
validate_list = np.array(validate_list)
np.save('cora_splits/cora_validate_split.npy', validate_list)


test_list = []
for i in range(len(test)):
    test_list.append(test.iloc[i][0])    
test_list = np.array(test_list)
np.save('cora_splits/cora_test_split.npy', test_list)





print("train len = ",len(train_list))
print("train type = ",type(train_list))
print("validate len = ",len(validate))
print("test len = ",len(test))

print("train list = ",train)
train
