/*
 * Copyright 1993-2008 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.  This source code is a "commercial item" as 
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer software" and "commercial computer software 
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */

#include "cudaAcceleration.h"
#ifdef USE_CUDA

#include "cudaAcc_data.h"
#include "cudaAcc_utilities.h"

#define BLOCK_DIM 32

// Transpose function from sample in NVIDIA CUDA SDK
__global__ void cudaAcc_transpose(float *odata, float *idata, int width, int height)
{
	__shared__ float block[BLOCK_DIM][BLOCK_DIM+1];

	// read the matrix tile into shared memory
	unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
	unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
	if((xIndex < width) && (yIndex < height))
	{
		unsigned int index_in = yIndex * width + xIndex;
		block[threadIdx.y][threadIdx.x] = idata[index_in];
	}

	__syncthreads();

	// write the transposed matrix tile to global memory
	xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
	yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
	if((xIndex < height) && (yIndex < width))
	{
		unsigned int index_out = yIndex * height + xIndex;
		odata[index_out] = block[threadIdx.x][threadIdx.y];
	}
}


template <int width, int height>
__global__ void cudaAcc_transposeOriginal(float *odata, float *idata)
{
  __shared__ float block[BLOCK_DIM][BLOCK_DIM+1];
  
  // read the matrix tile into shared memory
  unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
  unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
  //  if((xIndex < width) && (yIndex < height))
  //  {
  unsigned int index_in = yIndex * width + xIndex;
  block[threadIdx.y][threadIdx.x] = idata[index_in];
  //    }
  
  __syncthreads();
  
  // write the transposed matrix tile to global memory
  xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
  yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
  //if((xIndex < height) && (yIndex < width))
  //  {
  unsigned int index_out = yIndex * height + xIndex;
  odata[index_out] = block[threadIdx.x][threadIdx.y];
  //  }
}


// Transpose that effectively reorders execution of thread blocks along diagonals of the
// matrix (also coalesced and has no bank conflicts)
//
// Here blockIdx.x is interpreted as the distance along a diagonal and blockIdx.y as
// corresponding to different diagonals
//
// blockIdx_x and blockIdx_y expressions map the diagonal coordinates to the more commonly
// used cartesian coordinates so that the only changes to the code from the coalesced version
// are the calculation of the blockIdx_x and blockIdx_y and replacement of blockIdx.x and
// bloclIdx.y with the subscripted versions in the remaining code
#define TILE_DIM 32
#define BLOCK_ROWS 8
/*
    dim3 grid(size_x/TILE_DIM, size_y/TILE_DIM), threads(TILE_DIM,BLOCK_ROWS);
 */

template <int width, int height>
__global__ void cudaAcc_transpose(float *odata, float *idata)
{
  __shared__ float tile[TILE_DIM][TILE_DIM+1];

  int blockIdx_x, blockIdx_y;
  
  // do diagonal reordering
  if (width == height)
    {
      blockIdx_y = blockIdx.x;
      blockIdx_x = (blockIdx.x+blockIdx.y) & (width/TILE_DIM - 1);//%gridDim.x;
    }
  else
    {
      int bid = blockIdx.x + width/TILE_DIM*blockIdx.y;
      blockIdx_y = bid & (height/TILE_DIM - 1); //%gridDim.y;
      blockIdx_x = ((bid/(height/TILE_DIM))+blockIdx_y) & (width/TILE_DIM - 1); //%gridDim.x; // height/TILE_DIM = gridDim.y
    }
  
  // from here on the code is same as previous kernel except blockIdx_x replaces blockIdx.x
    // and similarly for y
  
  int xIndex = blockIdx_x * TILE_DIM + threadIdx.x;
  int yIndex = blockIdx_y * TILE_DIM + threadIdx.y;
  int index_in = xIndex + (yIndex)*width;

  xIndex = blockIdx_y * TILE_DIM + threadIdx.x;
  yIndex = blockIdx_x * TILE_DIM + threadIdx.y;
  int index_out = xIndex + (yIndex)*height;

  for(int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
    {
      tile[threadIdx.y + i][threadIdx.x] = idata[index_in + i*width];
    }
  
  __syncthreads();
  
  for(int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
    {
      odata[index_out + i*height] = tile[threadIdx.x][threadIdx.y + i];
    }
}


void cudaAcc_transposeGPU(float *odata, float *idata, int width, int height) 
{
	if (!cudaAcc_initialized()) return;

    dim3 grid(width/TILE_DIM, height/TILE_DIM);
    dim3 block(TILE_DIM, BLOCK_ROWS);

//	dim3 block(BLOCK_DIM, BLOCK_DIM, 1);
//	dim3 grid((width + BLOCK_DIM - 1) / BLOCK_DIM, (height + BLOCK_DIM - 1) / BLOCK_DIM, 1);

	// no need to copy data from host, this data is already on device thanks to cudaAcc_execute_dfts
	switch(width)
	  {
	  case 32:
	    if(height == 32768)
	      cudaAcc_transpose<32, 32768><<<grid, block>>>(odata, idata);
	    else
	      cudaAcc_transpose<<<grid, block>>>(odata, idata, width, height);
	    break;
	    
	  case 64:
	    if(height == 16384)
	      cudaAcc_transpose<64, 16384><<<grid, block>>>(odata, idata);
	    else
	      cudaAcc_transpose<<<grid, block>>>(odata, idata, width, height);
	    break;
	    
	  case 128:
	    if(height == 8192)
	      cudaAcc_transpose<128, 8192><<<grid, block>>>(odata, idata);
	    else
	      cudaAcc_transpose<<<grid, block>>>(odata, idata, width, height);
	    break;
	    
	  case 256:
	    if(height == 4096)
	      cudaAcc_transpose<256, 4096><<<grid, block>>>(odata, idata);
	    else
	      cudaAcc_transpose<<<grid, block>>>(odata, idata, width, height);
	    break;
	      
	  case 512:
	    if(height == 2048)
	      cudaAcc_transpose<512, 2048><<<grid, block>>>(odata, idata);
	    else
	      cudaAcc_transpose<<<grid, block>>>(odata, idata, width, height);
	    break;
	    
	    
	  case 1024:
	    if(height == 1024)
	      cudaAcc_transpose<1024, 1024><<<grid, block>>>(odata, idata);
	    else
	      cudaAcc_transpose<<<grid, block>>>(odata, idata, width, height);
	    break;
	  }


	CUDA_ACC_SAFE_CALL_NO_SYNC("cudaAcc_transpose");
}


void cudaAcc_transposeGPU(float *odata, float *idata, int width, int height, cudaStream_t stream) 
{
	if (!cudaAcc_initialized()) return;

    dim3 grid(width/TILE_DIM, height/TILE_DIM);
    dim3 block(TILE_DIM, BLOCK_ROWS);

//	dim3 block(BLOCK_DIM, BLOCK_DIM, 1);
//	dim3 grid((width + BLOCK_DIM - 1) / BLOCK_DIM, (height + BLOCK_DIM - 1) / BLOCK_DIM, 1);

	// no need to copy data from host, this data is already on device thanks to cudaAcc_execute_dfts
	switch(width)
	  {
	  case 32:
	    if(height == 32768)
	      cudaAcc_transpose<32, 32768><<<grid, block, 0, stream>>>(odata, idata);
	    else
	      cudaAcc_transpose<<<grid, block, 0, stream>>>(odata, idata, width, height);
	    break;
	    
	  case 64:
	    if(height == 16384)
	      cudaAcc_transpose<64, 16384><<<grid, block, 0, stream>>>(odata, idata);
	    else
	      cudaAcc_transpose<<<grid, block, 0, stream>>>(odata, idata, width, height);
	    break;
	    
	  case 128:
	    if(height == 8192)
	      cudaAcc_transpose<128, 8192><<<grid, block, 0, stream>>>(odata, idata);
	    else
	      cudaAcc_transpose<<<grid, block, 0, stream>>>(odata, idata, width, height);
	    break;
	    
	  case 256:
	    if(height == 4096)
	      cudaAcc_transpose<256, 4096><<<grid, block, 0, stream>>>(odata, idata);
	    else
	      cudaAcc_transpose<<<grid, block, 0, stream>>>(odata, idata, width, height);
	    break;
	      
	  case 512:
	    if(height == 2048)
	      cudaAcc_transpose<512, 2048><<<grid, block, 0, stream>>>(odata, idata);
	    else
	      cudaAcc_transpose<<<grid, block, 0, stream>>>(odata, idata, width, height);
	    break;
	    
	    
	  case 1024:
	    if(height == 1024)
	      cudaAcc_transpose<1024, 1024><<<grid, block, 0, stream>>>(odata, idata);
	    else
	      cudaAcc_transpose<<<grid, block, 0, stream>>>(odata, idata, width, height);
	    break;
	  }


	CUDA_ACC_SAFE_CALL_NO_SYNC("cudaAcc_transpose");
}

/*
void cudaAcc_transposeGPU(float *odata, float *idata, int width, int height, cudaStream stream) {
	if (!cudaAcc_initialized()) return;

	dim3 block(BLOCK_DIM, BLOCK_DIM, 1);
	dim3 grid((width + BLOCK_DIM - 1) / BLOCK_DIM, (height + BLOCK_DIM - 1) / BLOCK_DIM, 1);

	// no need to copy data from host, this data is already on device thanks to cudaAcc_execute_dfts
	switch(width)
	  {
	  case 32:
	    if(height == 32768)
	      cudaAcc_transpose<32, 32768><<<grid, block, 0, stream>>>(odata, idata);
	    else
	      cudaAcc_transpose<<<grid, block, 0, stream>>>(odata, idata, width, height);
	    break;
	    
	  case 64:
	    if(height == 16384)
	      cudaAcc_transpose<64, 16384><<<grid, block, 0, stream>>>(odata, idata);
	    else
	      cudaAcc_transpose<<<grid, block, 0, stream>>>(odata, idata, width, height);
	    break;
	    
	  case 128:
	    if(height == 8192)
	      cudaAcc_transpose<128, 8192><<<grid, block, 0, stream>>>(odata, idata);
	    else
	      cudaAcc_transpose<<<grid, block, 0, stream>>>(odata, idata, width, height);
	    break;
	    
	  case 256:
	    if(height == 4096)
	      cudaAcc_transpose<256, 4096><<<grid, block, 0, stream>>>(odata, idata);
	    else
	      cudaAcc_transpose<<<grid, block, 0, stream>>>(odata, idata, width, height);
	    break;
	      
	  case 512:
	    if(height == 2048)
	      cudaAcc_transpose<512, 2048><<<grid, block, 0, stream>>>(odata, idata);
	    else
	      cudaAcc_transpose<<<grid, block, 0, stream>>>(odata, idata, width, height);
	    break;
	    
	    
	  case 1024:
	    if(height == 1024)
	      cudaAcc_transpose<1024, 1024><<<grid, block, 0, stream>>>(odata, idata);
	    else
	      cudaAcc_transpose<<<grid, block, 0, stream>>>(odata, idata, width, height);
	    break;
	  }


	CUDA_ACC_SAFE_CALL_NO_SYNC("cudaAcc_transpose");
}

*/
#endif //USE_CUDA
