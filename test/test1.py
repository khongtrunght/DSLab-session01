# %% 
import numpy as np


a = np.array([1,2,3,4])
l = [1,2,3,4]
def test(b):
    global a
    arr = [0,1,2,3]
    np.random.shuffle(arr)
    b = b[arr]
    print(b)

def test1(l):
    l = [3,4,5,6]




print(a)
test(a)
print(a)
# %%
print(l)
test1(l)
print(l)