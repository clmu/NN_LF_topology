import numpy as np

data = np.asarray([[0, 1, 2, 3], [2.5111, 2.49999, 2.666661, 2.11111]])

np.savetxt('array.csv', data, delimiter=',')

imported= np.loadtxt('array.csv', delimiter=',')