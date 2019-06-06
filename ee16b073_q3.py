import cvxpy as cp
import numpy as np
import scipy.io
mat = scipy.io.loadmat('Ratings.mat')
M = mat['X']
k=80
print('-'*k)
print('Question 3 (Netflix Problem)')
print('Ratings matrix is of the shape:',M.shape)
print('-'*k)

print('The relaxed convex optimisation problem is as follows:\n')
print('minimize: r')
print('subject to:')
print('Xij = Mij for all i,j in W(set of known elements in the Netflix data)')
print('trace(Y)+trace(Z) <= 2*r')
print('[Y   X]\n[X.T Z] > 0')
print('-'*k)

m = 50
n = 38
X = cp.Variable((m,n))
# Y = cp.Variable((m,m),symmetric=True)
# Z = cp.Variable((n,n),symmetric=True)
Y = cp.Variable((m,m),PSD=True)
Z = cp.Variable((n,n),PSD=True)

constraints = [cp.vstack([cp.hstack([Y,X]),cp.hstack([X.T,Z])]) >>0 ,X>=0]
# constraints = [cp.vstack([cp.hstack([Y,X]),cp.hstack([X.T,Z])]) >>0]
# constraints+= [X >> 0]
index = np.where(M!=0)
constraints += [X[index] == M[index]]
#
# for i in range(M.shape[0]):
#     for j in range(M.shape[1]):
#         if M[i][j]!=0:
#             constraints+=[X[i][j] == M[i][j]]

prob = cp.Problem(cp.Minimize((cp.trace(Y)+cp.trace(Z))),constraints)
print('Solving the problem... ')
prob.solve()

# Print result.
print('-'*k)
print("The optimal solution X is")
print(X.value)
print('\nRank of the optimal matrix of X is:',np.linalg.matrix_rank(X.value))
print('-'*k)

######################################################################################################
