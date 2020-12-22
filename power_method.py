"""
Created on Wed Oct 28 12:22:54 2020

@author: ashwath
"""

import numpy as np

def eigenvalue(A, v):
    Av = A.dot(v)
    return v.dot(Av)

def power_iteration(A):
    n, d = A.shape

    v = np.ones(d)
    ev = eigenvalue(A, v)

    while True:
        Av = A.dot(v)
        v_new = Av / np.linalg.norm(Av)

        ev_new = eigenvalue(A, v_new)
        if np.abs(ev - ev_new) < 0.00001:
            break
        
        v = v_new
        ev = ev_new
    return ev_new, v_new

A = np.array([[4,2],[5,7]])
A_inv = np.linalg.inv(A)

largest_val1, eigen_vector1 = power_iteration(A)  

print("\nPower Method")
print(largest_val1, eigen_vector1)

largest_val2, eigen_vector2 = power_iteration(A_inv)

#print("\nInverse Power Method")
#print(1/largest_val2)   
#
#p = 3
#I = np.identity(3, dtype = int)
#B = np.linalg.inv(A-p*I)
#largest_val3, eigen_vector3 = power_iteration(B)
#print("\nShifted Inverse Power Method")
#print(1/largest_val3 + p)






