import numpy as np

data = np.genfromtxt('./testset063024_1.txt', delimiter=',')
start_index = len(data) - 50 * 1000
last_50_units = data[start_index:]
for i in range(50):
    unit = last_50_units[i * 500:(i + 2) * 500]
    np.savetxt(f'./testset063024_1/testset063024_{i+1}.txt', unit, delimiter=',')