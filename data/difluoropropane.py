import numpy as np

# Initialize the Zeeman scalar values as a numpy array
inter = {
    'zeeman': {
        'scalar': np.array([4.6075, 4.6075, -223.5314, 2.1011, 2.1011, -223.5314, 4.6075, 4.6075])
    }
}

# Initialize the scalar coupling matrix (8x8) with zeros
inter['coupling'] = {'scalar': np.zeros((8, 8))}


params_4 = 47.0060
params_5 = 5.7871
params_6 = 25.7705
# Scalar couplings (set specific matrix entries)
inter['coupling']['scalar'][0, 2] = params_4
inter['coupling']['scalar'][1, 2] = params_4
inter['coupling']['scalar'][5, 6] = params_4
inter['coupling']['scalar'][5, 7] = params_4
inter['coupling']['scalar'][0, 3] = params_5
inter['coupling']['scalar'][0, 4] = params_5
inter['coupling']['scalar'][1, 3] = params_5
inter['coupling']['scalar'][1, 4] = params_5
inter['coupling']['scalar'][3, 6] = params_5
inter['coupling']['scalar'][3, 7] = params_5
inter['coupling']['scalar'][4, 6] = params_5
inter['coupling']['scalar'][4, 7] = params_5
inter['coupling']['scalar'][2, 3] = params_6
inter['coupling']['scalar'][2, 4] = params_6
inter['coupling']['scalar'][3, 5] = params_6
inter['coupling']['scalar'][4, 5] = params_6

arr = inter['coupling']['scalar'].copy()

np.fill_diagonal(arr, inter['zeeman']['scalar'])

upper_triangle = np.triu(arr)
symmetric_matrix = upper_triangle + upper_triangle.T - np.diag(np.diag(upper_triangle))

np.savetxt('data/Difluoropropane.csv', symmetric_matrix, delimiter=',', fmt='%.6f')
