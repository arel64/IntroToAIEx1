
import matplotlib.pyplot as plt
import numpy as np
n_rows = 200000
n_cols = 20
p = 0.5
samples = 50


matrix = np.ndarray([n_rows,n_cols]);
for cell in np.nditer(matrix,op_flags=['writeonly']):
    cell[...] = np.random.binomial(1,p)
    
emp_mean_vector = matrix.sum(axis=1) / n_cols
emp_mean_corrected = np.abs(emp_mean_vector - p)


# emperical prob as a function of epsilon
eps_possiblities = np.linspace(0,1,samples)
emperical_prob = []
hoffding_ineq = []
for eps in eps_possiblities:
    emperical_prob += [np.sum(emp_mean_corrected>eps)/n_rows]
    hoffding_ineq += [2*np.power(np.e,(-2*(eps**2)*n_cols))]


plt.plot(eps_possiblities,emperical_prob,'r')
plt.plot(eps_possiblities,hoffding_ineq,'b')
plt.xlabel("Epsilon")
plt.ylabel("Empirical Probability")
plt.show()