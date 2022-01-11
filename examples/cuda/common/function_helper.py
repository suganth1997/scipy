import numpy as np
from cuda import cuda
from common import common
from common.helper_cuda import checkCudaErrors, findCudaDeviceDRV


class cudaFunctionHelper:
   def __init__(self, code, fname):
      self.code = code
      self.buffer = []

      # Initialize
      checkCudaErrors(cuda.cuInit(0))

      self.cuDevice = findCudaDeviceDRV()
      # Create context
      self.cuContext = checkCudaErrors(cuda.cuCtxCreate(0, self.cuDevice))

      uvaSupported = checkCudaErrors(cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, self.cuDevice))
      if not uvaSupported:
         print("Accessing pageable memory directly requires UVA")
         return

      self.kernelHelper = common.KernelHelper(self.code, int(self.cuDevice))
      self.cudaFun = self.kernelHelper.getFunction(fname)

   def allocateMemory_(self, *args):
      for arg in args:
         if type(arg) == np.ndarray:
            size = len(arg) * arg.dtype.itemsize
            self.buffer.append(checkCudaErrors(cuda.cuMemAlloc(size)))

   def allocateMemory(self, nBytesMem):
      for nBytes in nBytesMem:
         self.buffer.append(checkCudaErrors(cuda.cuMemAlloc(nBytes)))

   def __del__(self):
      checkCudaErrors(cuda.cuCtxDestroy(self.cuContext))
      for buff in self.buffer:
         checkCudaErrors(cuda.cuMemFree(buff))


class cudaFunction:
   # argsAttr is a dictionary with C argument names as keys and the values are a tuple (var/arr, N_size, np.dtype, input/output/both)
   def __init__(self, code, fname, argsAttr, argNamesInOrder):
      self.argsAttr = argsAttr
      self.argNames = argNamesInOrder
      self.argNameInd = dict(zip(self.argNames, range(len(self.argNames))))
      self.memBuffer = []

      self.funGPU = cudaFunctionHelper(code, bytes(fname, 'utf-8'))

      for k in self.argNames:
         if self.argsAttr[k][0] == 'var':
            self.memBuffer.append(0)
         elif self.argsAttr[k][0] == 'arr':
            self.memBuffer.append(checkCudaErrors(cuda.cuMemAlloc(self.argsAttr[k][1] * self.argsAttr[k][2].itemsize)))

   def __call__(self, args, blocksPerGrid, threadsPerBlock):
      for k in args.keys():
         if self.argsAttr[k][3] <= 0 and self.argsAttr[k][0] == 'arr':
            checkCudaErrors(cuda.cuMemcpyHtoD(self.memBuffer[self.argNameInd[k]], args[k], self.argsAttr[k][1] * self.argsAttr[k][2].itemsize))

      c_attr = []
      c_attr_type = []
      for k in self.argNames:
         if self.argsAttr[k][0] == 'arr':
            c_attr.append(self.memBuffer[self.argNameInd[k]])
            c_attr_type.append(None)
         elif self.argsAttr[k][0] == 'var':
            c_attr.append(args[k])
            c_attr_type.append(np.ctypeslib.as_ctypes_type(self.argsAttr[k][2]))

      kernelArgs = (tuple(c_attr), tuple(c_attr_type))

      checkCudaErrors(cuda.cuLaunchKernel(self.funGPU.cudaFun,
                                          blocksPerGrid, 1, 1,
                                          threadsPerBlock, 1, 1,
                                          0, cuda.CUstream(0),
                                          kernelArgs, 0))

      for k in args.keys():
         if self.argsAttr[k][3] >= 0 and self.argsAttr[k][0] == 'arr':
            checkCudaErrors(cuda.cuMemcpyDtoH(args[k], self.memBuffer[self.argNameInd[k]], self.argsAttr[k][1] * self.argsAttr[k][2].itemsize))
