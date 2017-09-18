/*
It is instructive to compare zero-copy and unified memory. For the former, the memory is allocated in
page-locked fashion on the host. A device thread has to reach out to get the data. No guarantee of
coherence is provided as, for instance, the host could change the content of the pinned memory while the
device reads its content. For UM, the memory is allocated on the device and transparently made available
where needed. Specifically, upon a call to
cudaError_t cudaMallocManaged(void** devPtr, size_t size, unsigned int flag);
the user has, in devPtr, a pointer to an address of a chunk of device memory. This address can be
equally well manipulated on the device and the host (although, as illustrated below, not simultaneously).
Note that cudaMallocManaged and cudaMalloc are semantically identical; in fact, the former can be
used anywhere the latter is used.
UM enables a “single-pointer-to-data” memory model. For instance, the same pointer can be used on the
host in a memcpy operation to copy a set of integers to an array mA, and then on the device to alter, just
like in the code snippet above, the value of each entry in mA. The data in mA will be coherent as long as
the host does not touch entries in mA when the GPU executes a kernel. The host can safely operate
with/on mA only after a cudaDeviceSynchronize call. Failure to obey this rule will lead to a segfault,
as illustrated in the following example lifted from the CUDA Programming guide.
*/

#include <iostream>
#include <cmath>
const int ARRAY_SIZE = 1000;
__global__ void increment(double* aArray, double val, unsigned int sz) {
 unsigned int indx = blockIdx.x * blockDim.x + threadIdx.x;
 if (indx < sz)
 aArray[indx] += val;
}
int main() {
 double* mA;
 cudaMallocManaged(&mA, ARRAY_SIZE * sizeof(double));
 for (int i = 0; i < ARRAY_SIZE; i++)
 mA[i] = 1.*i;
 double inc_val = 2.0;
 increment<<<2, 512>>>(mA, inc_val, ARRAY_SIZE);
 cudaDeviceSynchronize();
 double error = 0.;
 for (int i = 0; i < ARRAY_SIZE; i++)
 error += std::fabs(mA[i] - (i + inc_val));

for (int i=0;i<ARRAY_SIZE;++i){
    std::cout<<mA[i]<<' ';
}
std::cout<<'\n';

 std::cout << "Test: " << (error < 1.E-9 ? "Passed" : "Failed") << std::endl;
 cudaFree(mA);
 return 0;
}