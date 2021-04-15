#%%
import numpy as np
import pandas as pd
# %%
with open('data.txt', 'r') as f:
    with open('output_data.txt', 'w') as outfile:
        for line in f:
            if not (line[0].isalpha() or line[0] == '#'):
                outfile.writelines(line)

            
# %%
data = pd.read_csv('output_data.txt', delim_whitespace= True, header= None, index_col= 0)

# %%
data.head()
# %%
data.columns =  [f'A{i}' for i in range(1,16)] + ['B']

# %%
data.columns
# %%

X = data.iloc[:,:-1]
Y = data.iloc[:, -1]

# %%
X_prime = (X - X.min())/ (X.max() - X.min())

# %%
X_prime.head()

# %%
X_prime.insert(0,'A0',1)

# %%
np.array(X).shape


# %%
row_ids = np.array(range(10))

valid_ids = np.split(row_ids[:len(row_ids) - len(row_ids) % 5], 5)
valid_ids[-1] = np.append(valid_ids[-1], row_ids[len(row_ids) - len(row_ids) % 5:])
