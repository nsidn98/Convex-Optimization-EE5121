import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

k=80
mat = scipy.io.loadmat('piecewise_constant_data.mat')
y = mat['y']
y = y.reshape(len(y))
x = cp.Variable(len(y))
A = np.zeros((len(y)-1,len(y)))
for i in range(A.shape[0]):
     for j in range(A.shape[1]):
         if i==j:
             A[i][j]=-1
             A[i][j+1]=1

print('-'*k)
print('Question 1 (Signal recovery problem)')
print('The convex optimization problem is as follows:\n')
print('Epigraph for of the original form')
print('minimize t')
print('subject to: w1*||y-x||_2 - w1*t + w2*|Ax|_1 <= w2*20\n')
# print('|Ax|1<=20')
print('Here |Ax|1 is an approximation to cardinality(Ax)')
print('Cardinality of Ax is the number of jumps')
print('Here the matrix A is as follows:')
print(A)
print("-"*k)
#
# constraints = [cp.norm(A@x,1)<=10]
# objective = cp.Minimize(cp.norm(y-x,2))
# prob = cp.Problem(objective,constraints)
# prob.solve()
# print("\nThe optimal value is", prob.value)
# # print("A solution x is")
# # print(x.value)
# print('-'*k)
# plt.plot(x.value)
# plt.grid()
# plt.xlabel('time t')
# plt.ylabel('x')
# plt.title('Reconstructed Signal')
# plt.show()
#
# print()
# print('#'*k)
# print()
######################################################################################################
# SOCP
w1 = 0.5
w2 = 1 - w1
k=80
mat = scipy.io.loadmat('piecewise_constant_data.mat')
y = mat['y']
y = y.reshape(len(y))
x = cp.Variable(len(y))
t = cp.Variable(1)
A = np.zeros((len(y)-1,len(y)))
for i in range(A.shape[0]):
     for j in range(A.shape[1]):
         if i==j:
             A[i][j]=-1
             A[i][j+1]=1
             
# print('-'*k)
# print('Question 1 (Signal recovery problem)')
# print('The convex optimization problem is as follows:\n')
# print('minimize ||y-x||2')
# print('subject to: |Ax|1<=20\n')
# # print('|Ax|1<=20')
# print('Here |Ax|1 is an approximation to cardinality(Ax)')
# print('Cardinality of Ax is the number of jumps')
# print('Here the matrix A is as follows:')
# print(A)
# print("-"*k)
             
# constraints = [cp.norm(A@x,1)<=10]
constraints = [w1*cp.norm(y-x,2)- w1*t + w2*cp.norm(A@x,1)<=w2*5]
# objective = cp.Minimize(cp.norm(y-x,2))
objective = cp.Minimize(t)
prob = cp.Problem(objective,constraints)
prob.solve()
print("MSE:", np.linalg.norm(y-x.value,2))
# print("A solution x is")
# print(x.value)
print('-'*k)
plt.plot(x.value)
plt.grid()
plt.xlabel('time t')
plt.ylabel('x')
plt.title('Reconstructed Signal')
plt.show()

print()
print('#'*k)
print()