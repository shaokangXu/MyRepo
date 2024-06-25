/**
* Fabio D'Isidoro, ETH Zurich, 08.08.2017
*
* Implementation of a CUDA-based Cpp library for fast DRR generation with GPU acceleration
*
* Based both on the description found in the 锟絀mproved Algorithm?section in Jacob锟絪 paper (1998)
* https://www.researchgate.net/publication/2344985_A_Fast_Algorithm_to_Calculate_the_Exact_Radiological_Path_Through_a_Pixel_Or_Voxel_Space
* and on the implementation suggested in Greef et al 2009
* https://www.ncbi.nlm.nih.gov/pubmed/19810482
*
* Source file for the Class Siddon (see header for more information)
*/

#include "siddon_class.cuh"
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef MAX
#define MAX(a,b) (a)>(b)?(a):(b)
#endif // !MAX
#ifndef MIN
#define MIN(a,b) (a)<(b)?(a):(b)
#endif // !MIN


// auxiliary functions

__device__ void get_dest(int idx, float* dest_array, float* dest) {

	dest[0] = dest_array[0 + 3 * idx];
	dest[1] = dest_array[1 + 3 * idx];
	dest[2] = dest_array[2 + 3 * idx];

}


__global__ void cuda_kernel(float* DRRarray,	// DRR output
	float* source,								// focal(source) point
	float* DestArray,							// spatial positions for all drr pixels
	int DRRsize0,								// DRR image width and height
	float* movImgArray,							// CT data array why float? short?
	int* MovSize,								// CT size
	float* MovSpacing,							// CT spacing
	float X0, float Y0, float Z0,				// CT Origin
	float  threshold)
{

	// DRR image indeces
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (row >= DRRsize0 || col >= DRRsize0)
	{
		return;
	}

	//auto PrintMe = [=]()->bool {return (row == 10 && col == 10) ? true : false; };
	auto PrintMe = [=]()->bool {return false; };

	if (PrintMe())
	{
		printf("row = %d, col = %d\n", row, col);
	}

	// DRR array index
	int DRRidx = col + DRRsize0 * row;


	float rayVector[3];
	int cIndex[3];


	float firstIntersection[3];
	float alphaX1, alphaXN, alphaXmin, alphaXmax;
	float alphaY1, alphaYN, alphaYmin, alphaYmax;
	float alphaZ1, alphaZN, alphaZmin, alphaZmax;
	float alphaMin, alphaMax;
	float alphaX, alphaY, alphaZ, alphaCmin, alphaCminPrev;
	float alphaUx, alphaUy, alphaUz;
	float alphaIntersectionUp[3], alphaIntersectionDown[3];
	float d12, value, rayLength;
	float firstIntersectionIndex[3];
	int   firstIntersectionIndexUp[3], firstIntersectionIndexDown[3];
	int   iU, jU, kU;


	// --- define destination point based on DRR array index --- 
	float dest[3]; // spatial position for current(idx) drr pixel
	get_dest(DRRidx, DestArray, dest);

	rayVector[0] = dest[0] - source[0];
	rayVector[1] = dest[1] - source[1];
	rayVector[2] = dest[2] - source[2];

	// --- source-to-destination distance --- 
	rayLength = sqrtf(powf(rayVector[0], 2) + powf(rayVector[1], 2) + powf(rayVector[2], 2));

	auto IsEqual = [](float a, float b) { return fabsf(a - b) < 1e-5 ? true : false; };


	// X
	if (!IsEqual(rayVector[0], 0))
	{
		alphaX1 = (0.0 + X0 - source[0]) / rayVector[0];
		alphaXN = (MovSize[0] * MovSpacing[0] + X0 - source[0]) / rayVector[0];
		alphaXmin = MIN(alphaX1, alphaXN);
		alphaXmax = MAX(alphaX1, alphaXN);
	}
	else
	{
		alphaXmin = -2;
		alphaXmax = 2;
	}

	if (PrintMe())
	{
		printf("alpha X min = %f, alpha X max = %f\n", alphaXmin, alphaXmax);
	}

	// Y
	if (!IsEqual(rayVector[1], 0))
	{
		alphaY1 = (0.0 + Y0 - source[1]) / rayVector[1];
		alphaYN = (MovSize[1] * MovSpacing[1] + Y0 - source[1]) / rayVector[1];
		alphaYmin = MIN(alphaY1, alphaYN);
		alphaYmax = MAX(alphaY1, alphaYN);

	}
	else
	{
		alphaYmin = -2;
		alphaYmax = 2;
	}

	if (PrintMe())
	{
		printf("alpha Y min = %f, alpha Y max = %f\n", alphaYmin, alphaYmax);
	}

	// Z
	if (!IsEqual(rayVector[2], 0))
	{
		alphaZ1 = (0.0 + Z0 - source[2]) / rayVector[2];
		alphaZN = (MovSize[2] * MovSpacing[2] + Z0 - source[2]) / rayVector[2];
		alphaZmin = MIN(alphaZ1, alphaZN);
		alphaZmax = MAX(alphaZ1, alphaZN);
	}
	else
	{
		alphaZmin = -2;
		alphaZmax = 2;
	}

	if (PrintMe())
	{
		printf("alpha Z min = %f, alpha Z max = %f\n", alphaZmin, alphaZmax);
	}

	/* Get the very first and the last alpha values when the ray
			intersects with the CT volume. */
	alphaMin = MAX(MAX(alphaXmin, alphaYmin), alphaZmin);
	alphaMax = MIN(MIN(alphaXmax, alphaYmax), alphaZmax);


	/* Calculate the parametric values of the first intersection point
		of the ray with the X, Y, and Z-planes after the ray entered the
		CT volume. */

	firstIntersection[0] = source[0] + alphaMin * rayVector[0];
	firstIntersection[1] = source[1] + alphaMin * rayVector[1];
	firstIntersection[2] = source[2] + alphaMin * rayVector[2];

	/* Transform world coordinate to the continuous index of the CT volume*/
	firstIntersectionIndex[0] = (firstIntersection[0] - X0) / MovSpacing[0];
	firstIntersectionIndex[1] = (firstIntersection[1] - Y0) / MovSpacing[1];
	firstIntersectionIndex[2] = (firstIntersection[2] - Z0) / MovSpacing[2];

	firstIntersectionIndexUp[0] = static_cast<int>(ceilf(firstIntersectionIndex[0]));
	firstIntersectionIndexUp[1] = static_cast<int>(ceilf(firstIntersectionIndex[1]));
	firstIntersectionIndexUp[2] = static_cast<int>(ceilf(firstIntersectionIndex[2]));

	firstIntersectionIndexDown[0] = static_cast<int>(floorf(firstIntersectionIndex[0]));
	firstIntersectionIndexDown[1] = static_cast<int>(floorf(firstIntersectionIndex[1]));
	firstIntersectionIndexDown[2] = static_cast<int>(floorf(firstIntersectionIndex[2]));

	if (IsEqual(rayVector[0], 0))
	{
		alphaX = 2;
	}
	else
	{
		alphaIntersectionUp[0] = (firstIntersectionIndexUp[0] * MovSpacing[0] + X0 - source[0]) / rayVector[0];
		alphaIntersectionDown[0] = (firstIntersectionIndexDown[0] * MovSpacing[0] + X0 - source[0]) / rayVector[0];
		alphaX = MAX(alphaIntersectionUp[0], alphaIntersectionDown[0]);
	}

	if (IsEqual(rayVector[1], 0))
	{
		alphaY = 2;
	}
	else
	{
		alphaIntersectionUp[1] = (firstIntersectionIndexUp[1] * MovSpacing[1] + Y0 - source[1]) / rayVector[1];
		alphaIntersectionDown[1] = (firstIntersectionIndexDown[1] * MovSpacing[1] + Y0 - source[1]) / rayVector[1];
		alphaY = MAX(alphaIntersectionUp[1], alphaIntersectionDown[1]);
	}

	if (IsEqual(rayVector[2], 0))
	{
		alphaZ = 2;
	}
	else
	{
		alphaIntersectionUp[2] = (firstIntersectionIndexUp[2] * MovSpacing[2] + Z0 - source[2]) / rayVector[2];
		alphaIntersectionDown[2] = (firstIntersectionIndexDown[2] * MovSpacing[2] + Z0 - source[2]) / rayVector[2];
		alphaZ = MAX(alphaIntersectionUp[2], alphaIntersectionDown[2]);
	}

	/* Calculate alpha incremental values when the ray intercepts with x, y, and z-planes */
	if (!IsEqual(rayVector[0], 0))
	{
		alphaUx = MovSpacing[0] / fabsf(rayVector[0]);
	}
	else
	{
		alphaUx = 999;
	}
	if (!IsEqual(rayVector[1], 0))
	{
		alphaUy = MovSpacing[1] / fabsf(rayVector[1]);
	}
	else
	{
		alphaUy = 999;
	}
	if (!IsEqual(rayVector[2], 0))
	{
		alphaUz = MovSpacing[2] / fabsf(rayVector[2]);
	}
	else
	{
		alphaUz = 999;
	}

	/* Calculate voxel index incremental values along the ray path. */
	if (source[0] < dest[0])
	{
		iU = 1;
	}
	else
	{
		iU = -1;
	}
	if (source[1] < dest[1])
	{
		jU = 1;
	}
	else
	{
		jU = -1;
	}

	if (source[2] < dest[2])
	{
		kU = 1;
	}
	else
	{
		kU = -1;
	}


	d12 = 0.0F; /* Initialize the sum of the voxel intensities along the ray path to zero. */

	/* Initialize the current ray position. */
	alphaCmin = MIN(MIN(alphaX, alphaY), alphaZ);

	/* Initialize the current voxel index. */
	cIndex[0] = firstIntersectionIndexDown[0];
	cIndex[1] = firstIntersectionIndexDown[1];
	cIndex[2] = firstIntersectionIndexDown[2];

	if (PrintMe())
	{
		printf("alphaCmin = %f,alphaMax = %f\n", alphaCmin, alphaMax);
		printf("dest = [%f,%f,%f]\n", dest[0], dest[1], dest[2]);
		printf("source = [%f,%f,%f]\n", source[0], source[1], source[2]);
	}

	while (alphaCmin < alphaMax) /* Check if the ray is still in the CT volume */
	{
		if (PrintMe())
		{
			printf("alphaCmin = %f,alphaMax = %f\n", alphaCmin, alphaMax);
		}

		/* Store the current ray position */
		alphaCminPrev = alphaCmin;

		if ((alphaX <= alphaY) && (alphaX <= alphaZ))
		{
			/* Current ray front intercepts with x-plane. Update alphaX. */
			alphaCmin = alphaX;
			cIndex[0] = cIndex[0] + iU;
			alphaX = alphaX + alphaUx;
		}
		else if ((alphaY <= alphaX) && (alphaY <= alphaZ))
		{
			/* Current ray front intercepts with y-plane. Update alphaY. */
			alphaCmin = alphaY;
			cIndex[1] = cIndex[1] + jU;
			alphaY = alphaY + alphaUy;
		}
		else
		{
			/* Current ray front intercepts with z-plane. Update alphaZ. */
			alphaCmin = alphaZ;
			cIndex[2] = cIndex[2] + kU;
			alphaZ = alphaZ + alphaUz;
		}

		if ((cIndex[0] >= 0) && (cIndex[0] < MovSize[0]) &&
			(cIndex[1] >= 0) && (cIndex[1] < MovSize[1]) &&
			(cIndex[2] >= 0) && (cIndex[2] < MovSize[2]))
		{
			/* If it is a valid index, get the voxel intensity. */
			int idx = cIndex[2] * MovSize[0] * MovSize[1] + cIndex[1] * MovSize[0] + cIndex[0];
			value = static_cast<float>(movImgArray[idx]);
			if (value > threshold) /* Ignore voxels whose intensities are below the threshold. */
			{
				d12 += (alphaCmin - alphaCminPrev) * (value - threshold);
			}
		}
	}
	d12 *= rayLength;
	DRRarray[DRRidx] = d12;
}


/**
*
* Deafult constructor
*
**/
SiddonGpu::SiddonGpu() { }

/**
*
* Overloaded constructor loads the CT scan (together with size and spacing) onto GPU memory
*
**/
SiddonGpu::SiddonGpu(int* NumThreadsPerBlock,	// for launch parameter: block size
	float* movImgArray,							// CT data array
	int* MovSize,								// CT size
	float* MovSpacing,							// CT spacing
	float X0, float Y0, float Z0,				// CT origin
	int* DRRSize								// DRR size
) 
{

	// ---- Allocate variable members ---- 
	m_NumThreadsPerBlock[0] = NumThreadsPerBlock[0];
	m_NumThreadsPerBlock[1] = NumThreadsPerBlock[1];
	m_NumThreadsPerBlock[2] = NumThreadsPerBlock[2];

	m_X0 = X0;
	m_Y0 = Y0;
	m_Z0 = Z0;

	m_DRRsize[0] = DRRSize[0];
	m_DRRsize[1] = DRRSize[1];
	m_DRRsize[2] = DRRSize[2];

	m_DRRsize0 = DRRSize[0];

	m_movImgMemSize = MovSize[0] * MovSize[1] * MovSize[2] * sizeof(float);
	m_DestMemSize = (DRRSize[0] * DRRSize[1] * DRRSize[2] * 3) * sizeof(float);
	m_DrrMemSize = (DRRSize[0] * DRRSize[1] * DRRSize[2]) * sizeof(float); // memory for each output drr

	// allocate space for device copies
	cudaMalloc((void**)&m_d_movImgArray, m_movImgMemSize);
	cudaMalloc((void**)&m_d_MovSize, 3 * sizeof(int));
	cudaMalloc((void**)&m_d_MovSpacing, 3 * sizeof(float));


	cudaMalloc(&m_d_source, 3 * sizeof(float));
	cudaMalloc(&m_d_destarray, m_DestMemSize);
	cudaMalloc(&m_d_drrarray, m_DrrMemSize);

	// Copy arrays related to the moving image onto device array
	cudaMemcpy(m_d_movImgArray, movImgArray, m_movImgMemSize, cudaMemcpyHostToDevice);
	cudaMemcpy(m_d_MovSize, MovSize, 3 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(m_d_MovSpacing, MovSpacing, 3 * sizeof(float), cudaMemcpyHostToDevice);
}

/**
*
* Destructor clears everything left from the GPU memory
*
**/
SiddonGpu::~SiddonGpu() 
{
	cudaFree(m_d_movImgArray);
	cudaFree(m_d_MovSize);
	cudaFree(m_d_MovSpacing);
	cudaFree(m_d_drrarray);
	cudaFree(m_d_destarray);
	cudaFree(m_d_source);

}

/**
*-The function generate DRR must be called with the following variables :
*
* @param source : array of(transformed) source physical coordinates
* @param DestArray : C - ordered 1D array of physical coordinates relative to the(transformed) output DRR image.
* @param drrArray : output, 1D array for output values of projected CT densities
*
**/
void SiddonGpu::generateDRR(float* source,		// focal point
	float* DestArray,							// the spatial positions for all drr pixels in (x,y,z)
	float* drrArray,							// drr output
	float threshold
)
{
	// Copy source and destination to device
	cudaMemcpy(m_d_destarray, DestArray, m_DestMemSize, cudaMemcpyHostToDevice);
	cudaMemcpy(m_d_source, source, 3 * sizeof(float), cudaMemcpyHostToDevice);

	// determine number of required blocks
	dim3 threads_per_block(m_NumThreadsPerBlock[0], m_NumThreadsPerBlock[1]);
	dim3 number_of_blocks(
		(m_DRRsize[0] + threads_per_block.x - 1) / threads_per_block.x,
		(m_DRRsize[1] + threads_per_block.y - 1) / threads_per_block.y
	);

	// launch kernel
	cuda_kernel << <number_of_blocks, threads_per_block >> > (
		m_d_drrarray,
		m_d_source,
		m_d_destarray,
		m_DRRsize0,
		m_d_movImgArray,
		m_d_MovSize,
		m_d_MovSpacing,
		m_X0, m_Y0, m_Z0, threshold);


	// Check for errors in Kernel launch
	cudaError_t status = cudaGetLastError();
	if (status != cudaSuccess)
	{
		printf("Cuda Launch Error: %s\n", cudaGetErrorString(status));
	}
	status = cudaDeviceSynchronize();
	if (status != cudaSuccess)
	{
		printf("Cuda DRR Kernel Error: %s\n", cudaGetErrorString(status));
	}
	// Copy result to host array
	cudaMemcpy(drrArray, m_d_drrarray, m_DrrMemSize, cudaMemcpyDeviceToHost);

	return;

}