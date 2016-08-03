#define B 2

__global__ void cudaAcc_GetPowerSpectrum_kernel( int NumDataPoints, float2* FreqData, float* PowerSpectrum) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;    
   
//	if (i < NumDataPoints) {
		float ax = FreqData[i].x;		
		float ay = FreqData[i].y;		
//		PowerSpectrum[i] = freqData.x * freqData.x + freqData.y * freqData.y;
		PowerSpectrum[i] = __fadd_rn( __fmul_rn(ax,ax),__fmul_rn(ay,ay)); 
//	}
}

/*
__global__ void cudaAcc_GetPowerSpectrum_kernel2( int NumDataPoints, float2* FreqData, float* PowerSpectrum) 
{
	const int i = blockIdx.x * blockDim.x*B + threadIdx.x;    

	float ax[B];
	float ay[B];
   
	#pragma unroll
	for (int k=0;k<B;k++)
	{
		ax[k] = FreqData[i+k*blockDim.x].x;		
		ay[k] = FreqData[i+k*blockDim.x].y;		
	}
//		PowerSpectrum[i] = freqData.x * freqData.x + freqData.y * freqData.y;

	#pragma unroll
	for (int k=0;k<B;k++)
	{
		PowerSpectrum[i+k*blockDim.x] = __fadd_rn( __fmul_rn(ax[k],ax[k]),__fmul_rn(ay[k],ay[k])); 
	}
}
*/

void cudaAcc_GetPowerSpectrum(int blksize, float2* dev_WorkData, float* dev_PowerSpectrum) {
    const int cudaAcc_NumDataPoints = 1024*1024;
	dim3 block(blksize, 1, 1);
	dim3 grid((cudaAcc_NumDataPoints + block.x - 1) / block.x, 1, 1);

	cudaAcc_GetPowerSpectrum_kernel<<<grid, block>>>(cudaAcc_NumDataPoints, dev_WorkData, dev_PowerSpectrum);
}

/*
void cudaAcc_GetPowerSpectrum2(int blksize, float2* dev_WorkData, float* dev_PowerSpectrum) {
    const int cudaAcc_NumDataPoints = 1024*1024;
	dim3 block(blksize, 1, 1);
	dim3 grid((cudaAcc_NumDataPoints + block.x*B - 1) / (block.x*B), 1, 1);

	cudaAcc_GetPowerSpectrum_kernel2<<<grid, block>>>(cudaAcc_NumDataPoints, dev_WorkData, dev_PowerSpectrum);
}
*/