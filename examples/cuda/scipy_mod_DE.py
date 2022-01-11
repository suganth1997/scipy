import os
from multiprocessing import Pool
import time
import numpy as np
import matplotlib.pyplot as plt
from common.function_helper import cudaFunction
from scipy.optimize import differential_evolution

code = '''extern "C" 
__global__ void objFunGPU(const float *A, float *B, int N_pop, int N_params, float *x_ref, float h)
{
    int t_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_i < N_pop)
    {
        // float val = (A[i] * A[i] * A[i] - 2);
        // B[i] = val * val;

        float x = 0.0f;
        float a = A[t_i*N_params];
        float b = A[t_i*N_params + 1];
        float c = A[t_i*N_params + 2];
        float t = 0.0f;
        float error = 0.0f;

        for(int i=0; i<1000; i++)
        {
            error += fabs(x - x_ref[i]);
            x += h * (a * powf(x, 2) + c * t * x + sinf(b * t));
            t += h;
        }

        B[t_i] = sqrtf(error);
    }
    __syncthreads();
}
'''

a = 3.0  * np.random.rand()
b = 20.0 * np.random.rand()
c = 2.0  * np.random.rand()
p = [a, b, c]
# p =  [2.2997538565871825, 11.886552570552489, 1.851583608302054]
print("p = ", p)

f = lambda t, x, p : p[0] * x**2 + p[2]*t*x + np.sin(p[1] * t)

N = 1000
h = 1.0/N
t_ref = np.linspace(0, 1, N).astype(np.float32)
x_ref = np.zeros(N).astype(np.float32)

for i, t_ in enumerate(t_ref[:-1]):
    x_ref[i + 1] = x_ref[i] + h * f(t_, x_ref[i], p)

x_ref += 3e-3 * (-1 + 2 * np.random.rand(len(x_ref)))

# import matplotlib.pyplot as plt
# plt.plot(t_ref, x_ref)
# plt.show()

# exit()

N_params = 3

N_population = 25

h_A = np.linspace(-0.5, 1.6, N_population * N_params * N_params).astype(dtype=np.float32)
h_B = np.zeros(N_population * N_params).astype(dtype=np.float32)

objFun = cudaFunction(code, 'objFunGPU', {'h_A':('arr', N_population * N_params * N_params, h_A.dtype, -1), 'h_B':('arr', N_population * N_params, h_B.dtype, 1), 'N_pop':('var', 1, np.int32, -1), 'N_params':('var', 1, np.int32, -1), 'x_ref':('arr', N, x_ref.dtype, -1), 'h':('var', 1, np.float32, -1)}, ('h_A', 'h_B', 'N_pop', 'N_params', 'x_ref', 'h'))

# args = {'h_A':h_A, 'h_B':h_B, 'N':N_population}

threadsPerBlock = 1024
blocksPerGrid   = 4

def f_(p):
    x = np.zeros(N)
    for i, t_ in enumerate(t_ref[:-1]):
        x[i + 1] = x[i] + h * f(t_, x[i], p)

    return np.linalg.norm(x - x_ref)

def f_CPU(X, n_process = 8):
    if n_process == 1:
        val = []
        # print(len(X))
        for x in X:
            val.append(f_(x))
    else:
        p = Pool(n_process)
        val = p.map(f_, X)

    return val

def f_GPU(X):
    global h_A
    for i in range(len(X)):
        for j in range(len(X[i])):
            h_A[i * N_params + j] = X[i][j]
    
    args = {'h_A':h_A, 'h_B':h_B, 'N_pop':len(X), 'N_params':N_params, 'x_ref':x_ref, 'h':h}

    objFun(args, blocksPerGrid, threadsPerBlock)

    # print("Error = ", np.linalg.norm(h_B - np.array(val)))
    
    return h_B

def f_wrapper(X):
    val = []
    # print("X = ", [x[0] for x in X])
    islist = False
    try:
        X[0][0]
        islist = True
    except IndexError:
        islist = False

    # if not islist:
    #     print("Scalar detected")
    
    if islist:
        # return f_CPU(np.array([x[0] for x in X]))
        # return f_CPU(X, 8)
        return f_GPU(X)
    else:
        return f_(X)


start = time.time()
result = differential_evolution(f_wrapper, [(0, 3), (0, 20), (0, 2)], popsize=N_population, polish=False, pop_eval_by_list=True)
end = time.time()
print(result)
print("Time Taken = ", end - start)
