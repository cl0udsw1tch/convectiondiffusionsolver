import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import  gmres, spilu, LinearOperator
import math

def solve(A: csc_matrix, b, random_upper=None):
    
    # print("Solving...")
    # print("Generating preconditioner...")
    # ILU preconditioner
    ilu = None
    
    if random_upper != None:
        i = 1
        while i < A.shape[0]:
            try:
                ilu = spilu(A)
                break   
            except Exception as e:
                k = (-1)**(i + 1) * math.floor(i/2)
                print(f"Singular, setting diagonal {k} from main.")
                A.setdiag(2 * random_upper, k)
                i+=1
    else:
        ilu = spilu(A)
        
    # print("Preconditioning done.")
    # print("Starting gmres...")
                
    x = None

    try:
        x, _ = gmres(A, b, M=LinearOperator(A.shape, ilu.solve) )
        # print(f"Solution residual: {np.linalg.norm(residual)}")
        # print(f"Solution norm: {np.linalg.norm(x)}")
    except Exception as e:
        print("Error: ", e)

    return x

