import numpy as np

t =  np.array([[x * i for x in range(3)] for i in range(1, 4)])

t2 = np.array([[10 * i] for i in range(3)])

print(t)
print(t2)

print(t * t2)