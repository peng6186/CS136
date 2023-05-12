import numpy as np

P = np.array(
    [[0, 1, 0, 0, 0, 0, 0],
     [0.5, 0, 0.5, 0, 0,0,0],
     [0, 0.5, 0, 0.5, 0, 0, 0],
     [0, 0, 0.5, 0, 0.5, 0, 0],
     [0, 0, 0, 0.5, 0, 0, 0.5],
     [0, 0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0.5, 0.5, 0],
    ]
)

K = 3000
RES = np.array([1, 0, 0, 0, 0, 0, 0])

def calc_stationary_distribution():
    for i in range(K):
        # print(i, " iteration:")
        global RES
        RES = RES @ P
    
    print("Final result:")
    print(RES)
calc_stationary_distribution()