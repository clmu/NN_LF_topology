import numpy as np

error_mat = np.load('../Tests/percentage_accuracy_matrix.npy')
arg_indices = np.load('../Tests/args_over_threshold.npy')

values, indices = np.shape(arg_indices)

indices_values_large = np.zeros((values, indices + 1), dtype=float)

indices_values_large[:, :2] = arg_indices

percentage_deviation = np.zeros((values,), dtype=float)

for i in range(values):
    l, r = int(indices_values_large[i][0]), int(indices_values_large[i][1])
    indices_values_large[i][2], percentage_deviation[i] = error_mat[l][r], error_mat[l][r]

rows_over_threshold = np.argwhere(percentage_deviation > 20)

indices_values_small = np.zeros((len(rows_over_threshold), indices + 1), dtype=float)

counter = 0
for i in rows_over_threshold:
    i = i[0]
    indices_values_small[counter] = indices_values_large[i]
    counter += 1

set_large = set()
set_small = set()
for i in indices_values_large:
    set_large.add(i[0])

for i in indices_values_small:
    set_small.add(i[0])

voltage_pred_errors = 0
for i in indices_values_large:
    if i[1] < 3:
        voltage_pred_errors += 1

v_error_large = np.argwhere(indices_values_large < 3)
v_error_small = np.argwhere(indices_values_small < 3)