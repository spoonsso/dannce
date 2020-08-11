import os
from pprint import pprint
import numpy as np
import scipy.io

m = 0
y = list()
for i in range(0, 6):
    os.chdir("D:\\20191030\\mouse11\\raw\\Camera" + str(i + 1))
    x = np.load("f1.npy")
    # x = np.load('test_array_Cam' + str(i+1) + '.npy')
    y.append(x)

    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    np.set_printoptions(threshold=np.inf)
    # print(x)

    xp = np.diff(x[1, :])
    xps = np.sort(xp)

    print(max(xp))
    print(x[0, -1])
    print(x[1, -1])
    scipy.io.savemat("f.mat", dict(x=x))
