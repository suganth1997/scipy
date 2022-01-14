## Offloading objective function evaluation to GPU

The scipy library is modified to call the objective function with the whole parameter list. This allows for parallelization the way the user intends to. 

Here the objective function evaluation is offloaded to GPU with the cuda package in python. If the objective function evaluation takes time this might give good speedups. Here a ODE problem is considered and is solved with Euler's method. 

Scipy Differential evolution is used to optimize the parameters of the ODE.

Runtime values taken on,

CPU: Intel(R) Core(TM) i7-4710HQ

GPU: NVIDIA GeForce GTX 860M

|       | Time taken (s) |
| ----------- | ----------- |
| Single threaded | ~ 6.2       |
| Multi threaded | ~ 3.7        |
| GPU | ~ 0.15 |
