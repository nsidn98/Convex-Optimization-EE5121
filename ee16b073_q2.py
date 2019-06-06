import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
k=80
A = np.array([[1,2,0,1],[0,0,3,1],[0,3,1,1],[2,1,2,5],[1,0,3,2]])
c_max = np.ones(5)*100
p = np.array([3,2,7,6])
p_disc = np.array([2,1,4,2])
q = np.array([4,10,5,10])
# one = np.ones(4)
# objective = cp.Maximize(p.T@x + p_disc.T@(cp.minimum(0,x-q)))
print('-'*k)
print('Question 2 (Activity Level Problems)')
print('The convex optimization problem is as follows:\n')
print('maximize 1T.u')
print('subject to:\n')
print('x >= 0')
print('Ax <= c_max')
print('px >= u')
print('pq + p_disc(x-q) >= u')
print('-'*k)

x = cp.Variable(4)
constraints = [A@x<=c_max,x>=0]
# objective = cp.Maximize(cp.sum(cp.minimum(p*x,p*q+p_disc*(x-q))))
objective = cp.Maximize(p.T@x + p_disc.T@(cp.minimum(0,x-q)))
prob = cp.Problem(objective,constraints)
prob.solve()
x = np.array(x.value)
r = np.minimum(p*x,p*q+p_disc*(x-q))
print("The maximum revenue is", np.sum(r))
print("The optimal activity levels are:")
print(x)
print('The optimal revenues from each activity are:')
print(r)
print('The optimal average price per unit for each activity level are:')
print(r/x)
print('-'*k)
print()
print('#'*k)
print()


