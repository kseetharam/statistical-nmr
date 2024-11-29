import numpy as np

# Initialize the Zeeman scalar values as a numpy array
inter = {'zeeman': {'scalar': np.array([1.32375] * 6 + [4.61040] * 2 + [0.0] * 2)}}

# Initialize the scalar coupling matrix (10x10) with zeros
inter['coupling'] = {'scalar': np.zeros((10, 10))}

j_f_ch3_near = 24.0756
j_f_ch3_far = 6.4933
j_h_ch3 = 1.4429
j3_h_h = 3.5942
j3_f_h = 15.7298
j2_f_h = 47.7592
j3_f_f = -13.5792

# Scalar couplings (set specific matrix entries)
inter['coupling']['scalar'][0, 8] = j_f_ch3_near
inter['coupling']['scalar'][1, 8] = j_f_ch3_near
inter['coupling']['scalar'][2, 8] = j_f_ch3_near
inter['coupling']['scalar'][3, 9] = j_f_ch3_near
inter['coupling']['scalar'][4, 9] = j_f_ch3_near
inter['coupling']['scalar'][5, 9] = j_f_ch3_near
inter['coupling']['scalar'][0, 9] = j_f_ch3_far
inter['coupling']['scalar'][1, 9] = j_f_ch3_far
inter['coupling']['scalar'][2, 9] = j_f_ch3_far
inter['coupling']['scalar'][3, 8] = j_f_ch3_far
inter['coupling']['scalar'][4, 8] = j_f_ch3_far
inter['coupling']['scalar'][5, 8] = j_f_ch3_far
inter['coupling']['scalar'][0, 6] = j_h_ch3
inter['coupling']['scalar'][1, 6] = j_h_ch3
inter['coupling']['scalar'][2, 6] = j_h_ch3
inter['coupling']['scalar'][3, 7] = j_h_ch3
inter['coupling']['scalar'][4, 7] = j_h_ch3
inter['coupling']['scalar'][5, 7] = j_h_ch3
inter['coupling']['scalar'][6, 7] = j3_h_h
inter['coupling']['scalar'][7, 8] = j3_f_h
inter['coupling']['scalar'][6, 9] = j3_f_h
inter['coupling']['scalar'][6, 8] = j2_f_h
inter['coupling']['scalar'][7, 9] = j2_f_h
inter['coupling']['scalar'][8, 9] = j3_f_f


arr = inter['coupling']['scalar'].copy()

np.fill_diagonal(arr, inter['zeeman']['scalar'])

upper_triangle = np.triu(arr)
symmetric_matrix = upper_triangle + upper_triangle.T - np.diag(np.diag(upper_triangle))

np.savetxt('data/anti-difluorobutane.csv', symmetric_matrix, delimiter=',', fmt='%.6f')
