#include "cudaLib.cuh"
#include <algorithm>
#define GEMM_TILE_SIZE 32

void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	//	Insert GPU SAXPY kernel code here
}

int runGpuSaxpy(int vectorSize) {

	std::cout << "Hello GPU Saxpy!\n";

	//	Insert code here
	std::cout << "Lazy, you are!\n";
	std::cout << "Write code, you must\n";

	return 0;
}

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	//	Insert code here
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	//	Insert code here
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	double approxPi = 3.14159f;

	std::cout << "Sneaky, you are ...\n";
	std::cout << "Compute pi, you must!\n";
	return approxPi;
}

int runGpuMedianFilter (std::string imgPath, std::string outPath, MedianFilterArgs args) {
	
	std::cout << "Lazy, you are! ... ";
	std::cout << "Filter pixels, you must! ... ";

	return 0;
}

int medianFilter_gpu (uint8_t inPixels, ImageDim imgDim, 
	uint8_t outPixels, MedianFilterArgs args) {

	return 0;
}


int runGpuConv (int argc, char ** argv) {

	TensorShape iShape = AlexL1_InShape;
	TensorShape fShape = AlexL1_FilterShape;
	ConvLayerArgs convArgs = AlexL1_ConvArgs;
	
	
	int maxSharedMemPerBlock;
	cudaDeviceGetAttribute(&maxSharedMemPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, 0);
	std::cout << "Max shared memory per block: " << maxSharedMemPerBlock << "\n";


	int choice; 

	std::cout << "Which Layer do you want, choose 1-6!\n";
	std::cin >> choice;

	std::cout << "\n";
	std::cout << "Choice selected - " << choice << "\n\n";
	switch (choice) {

		case 1:
			std::cout << "Running Alex Net L1!\n";
			iShape = AlexL1_InShape;
			fShape = AlexL1_FilterShape;
			convArgs = AlexL1_ConvArgs;
			break;
		case 2:
			std::cout << "Running Alex Net L2!\n";
			iShape = AlexL2_InShape;
			fShape = AlexL2_FilterShape;
			convArgs = AlexL2_ConvArgs;
			break;
		case 3:
			std::cout << "Running Alex Net L3!\n";
			iShape = AlexL3_InShape;
			fShape = AlexL3_FilterShape;
			convArgs = AlexL3_ConvArgs;
			break;
		case 4:
			std::cout << "Running Alex Net L4!\n";
			iShape = AlexL4_InShape;
			fShape = AlexL4_FilterShape;
			convArgs = AlexL4_ConvArgs;
			break;
		case 5:
			std::cout << "Running Alex Net L5!\n";
			iShape = AlexL5_InShape;
			fShape = AlexL5_FilterShape;
			convArgs = AlexL5_ConvArgs;
			break;
		case 6:
			std::cout << "Running Alex Net L6!\n";
			iShape = AlexL6_InShape;
			fShape = AlexL6_FilterShape;
			convArgs = AlexL6_ConvArgs;
			break;
		default:
			std::cout << "Defaulting to running Alex Net L1!\n";
			iShape = AlexL1_InShape;
			fShape = AlexL1_FilterShape;
			convArgs = AlexL1_ConvArgs;
			break;

	}

	std::cout << "Evaluate convolution : \n";
	std::cout << "Input : " << iShape << " \n";
	std::cout << "Filter : " << fShape << " \n";

	TensorShape oShape;

	/*
	
	float* gpu_out = executeGpuConv(iShape, fShape, oShape, convArgs);
	float* cpu_out = executeCpuConv(iShape, fShape, oShape, convArgs);
	verifyVector_convLayer(cpu_out, gpu_out, )
	
	*/
	evaluateGpuConv(iShape, fShape, oShape, convArgs);
	
	//uint64_t errorCount = evaluateGpuConv(iShape, fShape, oShape, convArgs);
	//std::cout << "Found " << errorCount << " / " << tensorSize(oShape) << " errors \n";
	return 0;
}

uint64_t executeGpuConv(TensorShape iShape, TensorShape fShape,
	TensorShape& oShape, ConvLayerArgs args) {

	oShape.height = (iShape.height + 2 * args.padH - fShape.height) / args.strideH + 1;
	oShape.width = (iShape.width + 2 * args.padW - fShape.width) / args.strideW + 1;
	oShape.channels = (fShape.count);
	oShape.count = 1;

	float* h_in = nullptr;
	float* h_filter = nullptr;
	float* h_bias = nullptr;
	float* h_out = nullptr;

	int retVal;
	retVal = makeTensor(&h_in, iShape);
	if (retVal != 0) {
		std::cout << "Unable to make tensor \n";
		return 99;
	}
	retVal = makeTensor(&h_filter, fShape);
	if (retVal != 0) {
		std::cout << "Unable to make tensor \n";
		return 99;
	}
	retVal = makeVector(&h_bias, oShape.channels);
	if (retVal != 0) {
		std::cout << "Unable to make vector \n";
		return 99;
	}

	std::cout << "OutShape : " << oShape << " \n";
	h_out = (float*)malloc(tensorSize(oShape) * sizeof(float));

	/*CUDA Malloc for in, out, filter and bias*/

	float* d_in, * d_filter, * d_bias, * d_out;
	cudaMalloc(&d_in, tensorSize(iShape) * sizeof(float));
	cudaMalloc(&d_filter, tensorSize(fShape) * sizeof(float));
	cudaMalloc(&d_bias, (oShape.channels) * sizeof(float));
	cudaMalloc(&d_out, tensorSize(oShape) * sizeof(float));

	/*CUDA Memcpy for in, filter and bias*/
	cudaMemcpy(d_in, h_in, tensorSize(iShape) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_filter, h_filter, tensorSize(fShape) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_bias, h_bias, (oShape.channels) * sizeof(float), cudaMemcpyHostToDevice);

	/*Tile Sizing Dynamically based on my 4070 (49KB/SM)*/
	const int max_floats = 49152 / (4 * iShape.channels); //Sized for my 4070/v100
	const int max_tile_dim = std::floor(std::sqrt(max_floats)) - fShape.height + 1;
	const int oShape_dims = std::min(oShape.height, oShape.width);
	std::cout << "Sizing options: Max_tile_dim: " << max_tile_dim << " | oShape_dims: " << oShape_dims << "\n";
	int TILE_SIZE = std::min(max_tile_dim, oShape_dims);
	
	if (TILE_SIZE > 15) {
		TILE_SIZE = 6;
	}

	/*Block and Grid Dims*/
	dim3 blockDim(TILE_SIZE, TILE_SIZE, 1);
	dim3 gridDim((oShape.width + TILE_SIZE - 1) / TILE_SIZE, (oShape.height + TILE_SIZE - 1) / TILE_SIZE, oShape.channels);

	int shared_window_height = ((TILE_SIZE - 1) * args.strideH) + fShape.height; // 31
	int shared_window_width = ((TILE_SIZE - 1) * args.strideW) + fShape.width;  // 31

	size_t sharedMemSize = ((iShape.channels * (shared_window_height) * (shared_window_width))) * sizeof(float); // 4B*(3*31*31) = 11,5KB


	/*ConvLayer Kernel Call*/
	//convLayer_gpu << <gridDim, blockDim >> > (d_in, iShape, d_filter, fShape, d_bias, d_out, oShape, args);
	std::cout << "\n\n GPU Starting!\n\n\n";
	std::cout << "Bias[0] = " << h_bias[0] << "\n";
	std::cout << "Memory Size: " << sharedMemSize << " Bytes! \n";
	//convLayer_gpu << <gridDim, blockDim >> > (d_in, iShape, d_filter, fShape, d_bias, d_out, oShape, args);
	convLayer_gpu_SM_DM_v3 << <gridDim, blockDim, sharedMemSize >> > (d_in, iShape, d_filter, fShape, d_bias, d_out, oShape, args, TILE_SIZE);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cout << "Kernel Launch Error: " << cudaGetErrorString(err) << "\n";
	}
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		std::cout << "CUDA Error: " << cudaGetErrorString(err) << "\n";
	}

	/*CUDA memcpy for d_out to d_in*/
	cudaMemcpy(h_out, d_out, tensorSize(oShape) * sizeof(float), cudaMemcpyDeviceToHost);

	std::cout << "\n Comparing CPU and GPU now...\n";

	float* cpu_out = executeCpuConv2(iShape, fShape, oShape, args);
	uint64_t verify_errors = verifyVector_convLayer(cpu_out, h_out, (oShape.height * oShape.width * oShape.channels));
	std::cout << "\nFound " << verify_errors << " Errors...\n";

	/* cudaFree() functions */
	cudaFree(d_in);
	cudaFree(d_filter);
	cudaFree(d_bias);
	cudaFree(d_out);

	/*CPU Free*/
	free(h_in);
	free(h_filter);
	free(h_bias);
	free(h_out);
	return verify_errors;
}

uint64_t evaluateGpuConv (TensorShape iShape, TensorShape fShape, 
	TensorShape & oShape, ConvLayerArgs args) {

	uint64_t errorCount = 0;

	errorCount = executeGpuConv(iShape, fShape, oShape, args);

	#ifndef CONV_CHECK_DISABLE
		//	STUDENT: Verify number of errors in ouput matrix generated by convLayer_gpu
		//	STUDENT: Compare results with CPU output
		//	STUDENT: Return error count


	#endif

	return errorCount;
}

__global__ void convLayer_gpu(float* input, TensorShape iShape, float* filter,
	TensorShape fShape, float* bias, float* output, TensorShape oShape, ConvLayerArgs args) {

	/*Coordinates*/
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int channels = blockIdx.z;

	if (col < oShape.width && row < oShape.height) {
		uint32_t out_idx = ((channels * oShape.height + row) * oShape.width + col);
		output[out_idx] = bias[channels];
		//CPU Code For-loop repeat for the filter window//
		for (uint32_t i = 0; i < fShape.height; ++i) {
			for (uint32_t j = 0; j < fShape.width; ++j) {
				for (uint32_t k = 0; k < fShape.channels; ++k) {
					uint32_t in_h = args.strideH * row + i;
					uint32_t in_w = args.strideW * col + j;

					if (in_h < iShape.height && in_w < iShape.width) {
						uint32_t in_idx = (k * iShape.height + in_h) * iShape.width + in_w;
						uint32_t filter_idx = ((channels * fShape.channels + k) * fShape.height + i) * fShape.width + j;

						output[out_idx] += input[in_idx] * filter[filter_idx];
					}
				}
			}
		}
		if (args.activation) {
			output[out_idx] = fmaxf(0.0f, output[out_idx]);
		}
	}
}

__global__ void convLayer_gpu_SM_DM_v3(float* input, TensorShape iShape, float* filter,
	TensorShape fShape, float* bias, float* output, TensorShape oShape, ConvLayerArgs args, int TILE_SIZE) {

	extern __shared__ float tile[];
	
	const int tile_height = ((TILE_SIZE - 1) * args.strideH) + fShape.height;  // 6-1*4+11 = 31 
	const int tile_width  = ((TILE_SIZE - 1) * args.strideW) + fShape.width;   // " "		 = 31
	const int tile_depth = iShape.channels;									   //			 = 3
	
	/*Shared Memory Sizing: 31x31x3 = 2,883 elements, 961 elements per channel*/

	int tidx = threadIdx.x;
	int tidy = threadIdx.y;

	int out_x = blockIdx.x * blockDim.x + tidx;
	int out_y = blockIdx.y * blockDim.y + tidy;
	int out_z = blockIdx.z;

	int threadblock_thread_id = tidy * blockDim.x + tidx; // 0:35

	int base_x = blockIdx.x * blockDim.x * args.strideW; // 0:9*6*4 = 0 --> 216
	int base_y = blockIdx.y * blockDim.y * args.strideH; // 0:9*6*4 = 0 --> 216

	/*Load full 31x31 * 3 tile cooperatively*/
	/*First 12 threads load 80 elements per*/

	//int coop_tid = tidy * TILE_SIZE + tidx;
	int total_threads = TILE_SIZE * TILE_SIZE;			  // 6*6  = 36
	//int threads_per_channel = total_threads / tile_depth; // 36/3 = 12
	int elements_per_channel = (tile_height * tile_width); // 961+12-1/12 = 81, 972 elements will be loaded 12*81 but the last 11 are counted
	int total_elements = tile_depth * elements_per_channel;

	if (threadblock_thread_id == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
		printf("Block (0, 0, 0): Expected shared memory = %d bytes\n", total_elements * 4);
	}

	/*Cooperative Loading*/
	for (int idx = threadblock_thread_id; idx < total_elements; idx += total_threads) {
		int c = idx / elements_per_channel; //0:2
		int coop_idx = idx % elements_per_channel;
		int coop_y = coop_idx / tile_width; // row in shared tile (0 to tile_height-1)
		int coop_x = coop_idx % tile_width; // col in shared tile (0 to tile_width-1)

		// Compute the corresponding global input coordinates.
		// Incorporate any padding in the index calculation if needed.
		int input_x = base_x + coop_x; // PAD ?
		int input_y = base_y + coop_y; // PAD ?

		int shared_idx = (c * tile_height * tile_width) + coop_y * tile_width + coop_x;
		//printf("Channel: %d | Input_x: %d | Input_y: %d\n", c, input_x, input_y);
		if (input_x >= 0 && input_x < iShape.width &&
			input_y >= 0 && input_y < iShape.height) {
			int global_idx = (c * iShape.height + input_y) * iShape.width + input_x;
			tile[shared_idx] = input[global_idx];
			//tile[shared_idx] = 1.0f;
		}
		else {
			tile[shared_idx] = 0.0f;  // Handle boundaries (zero padding)
			
		}

	}

	__syncthreads();
	// Debug: Confirm execution
	if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadblock_thread_id == 0) {
		printf("Block (0, 0, 0): Loading completed\n");
	}

	// Compute output
	if (out_x < oShape.width && out_y < oShape.height) {
		uint32_t out_idx = (out_z * oShape.height + out_y) * oShape.width + out_x;
		float shared_sum = bias[out_z];

		for (uint32_t k = 0; k < fShape.channels; ++k) {
			float channel_sum = 0.0f;
			for (uint32_t i = 0; i < fShape.height; ++i) {
				for (uint32_t j = 0; j < fShape.width; ++j) {
					int in_x = (out_x)*args.strideW + j;
					int in_y = out_y * args.strideH + i;
					int tile_x = in_x - base_x;
					int tile_y = in_y - base_y;

					if (in_x < iShape.width && in_y < iShape.height) {
						uint32_t shared_idx = k * tile_height * tile_width + tile_y * tile_width + tile_x;
						uint32_t filter_idx = (out_z * fShape.channels + k) * fShape.height * fShape.width + i * fShape.width + j;
						channel_sum += tile[shared_idx] * filter[filter_idx];
						/*
						
						if (out_x == 3 && out_y == 0 && out_z == 0) {
							printf("Channel %d contribution: %f, Running sum: %f | Filter_H: %u | Filter_W: %u | Shared_IDX: %u | Filter_idx: %u\n",
								k, channel_sum, shared_sum, i, j, shared_idx, filter_idx);
							printf("\tBounds: in_x = %d | base_x = %d | in_y = %d | base_y = %d  && iS.w = %u | iS.h = %u ### tile_x = %d | tile_y = %d && tile_width = %d | tile_height = %d\n Filter_H: %u | Filter_W: %u\n ",
								in_x, base_x, in_y, base_y, iShape.width, iShape.height, tile_x, tile_y, (in_x + tile_width), (in_y + tile_height), i, j);
						}
						
						*/
					}
				}
			}
			shared_sum += channel_sum;
		}
		output[out_idx] = shared_sum;
	}

}

int runGpuGemm (int argc, char ** argv) {

	
	uint32_t BatchSize = 1;
	uint32_t fcc = 8192;
	//TensorShape aShape = { BatchSize, 1, 6, 4 };
	//TensorShape bShape = { 1, 1, 4, 8 };
	TensorShape aShape = { BatchSize, 1, 1, fcc };
	TensorShape bShape = { 1, 1, fcc, fcc };
	TensorShape cShape;
	GemmLayerArgs args = { 2, 2, 1 };

	evaluateGpuGemm_copy_speed(aShape, bShape, cShape, args, BatchSize);
	evaluateGpuGemm_copy(aShape, bShape, cShape, args, BatchSize);
	evaluateGpuGemm_uvm(aShape, bShape, cShape, args, BatchSize);
	executeCpuGemm_v1(aShape, bShape, cShape, args, BatchSize);
	return 0;
}


__global__ void gemmLayer_gpu_speed(float* a, TensorShape aShape,	float* b, TensorShape bShape,
	float* c, TensorShape cShape) {
	extern __shared__ float shared[];
	float* Mds = shared;
	float* Nds = shared + GEMM_TILE_SIZE * GEMM_TILE_SIZE;

	int bx = blockIdx.x;
	//int by = blockIdx.y;
	int bz = blockIdx.z;

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int Row		= bz;
	int Col		= bx * GEMM_TILE_SIZE + tx;

	float pVal = 0;
	int Width = aShape.width; //Inner Mat Dim

	//Preloading Nds (filter)
	/*
	
	if (ty == 0 && tx < GEMM_TILE_SIZE && Col < bShape.width) {
		for (int k = 0; k < GEMM_TILE_SIZE; k++) {
			Nds[k * GEMM_TILE_SIZE + tx] = b[k * bShape.width + Col];
		}
	}
	__syncthreads();
	
	*/

	if (Row < aShape.count) {
		for (int p_idx = 0; p_idx < (Width + GEMM_TILE_SIZE - 1) / GEMM_TILE_SIZE; ++p_idx) {

			if (ty < GEMM_TILE_SIZE && tx < GEMM_TILE_SIZE && Col < bShape.width) {
				int b_row = p_idx * GEMM_TILE_SIZE + ty;
				if (b_row < bShape.height) {
					Nds[ty * GEMM_TILE_SIZE + tx] = b[b_row * bShape.width + Col];
				}
				else {
					Nds[ty * GEMM_TILE_SIZE + tx] = 0.0f;
				}
			}

			//Loading A into Mds
			if (ty < GEMM_TILE_SIZE && (p_idx * GEMM_TILE_SIZE + tx) < aShape.width) {
				Mds[ty*GEMM_TILE_SIZE+tx] = a[Row * Width + p_idx * GEMM_TILE_SIZE + tx];
			}
			else {
				Mds[ty * GEMM_TILE_SIZE + tx] = 0.0f;
			}

			__syncthreads();

			for (int k = 0; k < GEMM_TILE_SIZE; ++k) {
				pVal += Mds[ty * GEMM_TILE_SIZE + k] * Nds[k*GEMM_TILE_SIZE+tx];
			}
			__syncthreads();
		}
		if (ty == 0 && Col < cShape.width) {
			c[Row * cShape.width + Col] = pVal;
		}
	}
	
}

__global__ void gemmLayer_gpu_v2(float* a, TensorShape aShape, float* b, TensorShape bShape,
	float* c, TensorShape cShape){
	__shared__ float Mds[GEMM_TILE_SIZE][GEMM_TILE_SIZE];
	__shared__ float Nds[GEMM_TILE_SIZE][GEMM_TILE_SIZE];

	int bx = blockIdx.x;
	//int by = blockIdx.y;
	int bz = blockIdx.z; //Batching idx 0:2


	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int Row = bz;
	int Col = bx * GEMM_TILE_SIZE + tx;

	float pVal = 0;
	int Width = aShape.width; //Inner Mat Dim
	if (Row < aShape.count) {
		for (int p_idx = 0; p_idx < (Width + GEMM_TILE_SIZE - 1) / GEMM_TILE_SIZE; ++p_idx) {

			//Loading A into Mds
			if (ty < GEMM_TILE_SIZE && (p_idx * GEMM_TILE_SIZE + tx) < aShape.width) {
				Mds[ty][tx] = a[Row * Width + p_idx * GEMM_TILE_SIZE + tx];
			}
			else {
				Mds[ty][tx] = 0.0f;
			}

			//Loading B into Nds
			if (Col < bShape.width && (p_idx * GEMM_TILE_SIZE + ty) < bShape.height) {
				Nds[ty][tx] = b[(p_idx * GEMM_TILE_SIZE + ty) * bShape.width + Col];
			}
			else {
				Nds[ty][tx] = 0.0f;
			}

			__syncthreads();

			for (int k = 0; k < GEMM_TILE_SIZE; ++k) {
				pVal += Mds[ty][k] * Nds[k][tx];
			}
			__syncthreads();

		}
		if (ty == 0 && Col < cShape.width) {
			c[Row * cShape.width + Col] = pVal;
		}
	}
}

__global__ void gemmLayer_gpu(float* a, TensorShape aShape, float* b, TensorShape bShape,
	float* c, TensorShape cShape) {
	__shared__ float Mds[GEMM_TILE_SIZE][GEMM_TILE_SIZE];
	__shared__ float Nds[GEMM_TILE_SIZE][GEMM_TILE_SIZE];

	int bx = blockIdx.x;
	//int by = blockIdx.y;
	int bz = blockIdx.z; //Batching idx 0:2


	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int Row = bz;
	int Col = bx * GEMM_TILE_SIZE + tx;

	float pVal = 0;
	int Width = aShape.width; //Inner Mat Dim
	if (Row < aShape.count) {
		for (int p_idx = 0; p_idx < (Width + GEMM_TILE_SIZE - 1) / GEMM_TILE_SIZE; ++p_idx) {

			//Loading A into Mds
			if (ty < GEMM_TILE_SIZE && (p_idx * GEMM_TILE_SIZE + tx) < aShape.width) {
				Mds[ty][tx] = a[Row * Width + p_idx * GEMM_TILE_SIZE + tx];
			}
			else {
				Mds[ty][tx] = 0.0f;
			}

			//Loading B into Nds
			if (Col < bShape.width && (p_idx * GEMM_TILE_SIZE + ty) < bShape.height) {
				Nds[ty][tx] = b[(p_idx * GEMM_TILE_SIZE + ty) * bShape.width + Col];
			}
			else {
				Nds[ty][tx] = 0.0f;
			}

			__syncthreads();

			for (int k = 0; k < GEMM_TILE_SIZE; ++k) {
				pVal += Mds[ty][k] * Nds[k][tx];
			}
			__syncthreads();

		}
		if (ty == 0 && Col < cShape.width) {
			c[Row * cShape.width + Col] = pVal;
		}
	}
}

int evaluateGpuGemm_copy_speed(TensorShape aShape, TensorShape bShape,
	TensorShape& cShape, GemmLayerArgs args, uint32_t BatchSize) {

	//cShape = { BatchSize, 1, 1, 4096 };

	cShape.height = aShape.height;
	cShape.width = bShape.width;
	cShape.channels = aShape.channels;
	cShape.count = BatchSize;

	float* h_a = nullptr;
	float* h_b = nullptr;

	int retVal;
	retVal = makeTensor(&h_a, aShape);
	if (retVal != 0) {
		std::cout << "Unable to make tensor \n";
		return -1;
	}
	retVal = makeTensor(&h_b, bShape);
	if (retVal != 0) {
		std::cout << "Unable to make tensor \n";
		return -1;
	}

	/*C (out) Array initialization*/
	float* h_c = (float*)malloc(tensorSize(cShape) * sizeof(float));

	/*CUDA Malloc for in, out, filter and bias*/

	float* d_a, * d_b, * d_c;
	cudaMalloc(&d_a, tensorSize(aShape) * sizeof(float));
	cudaMalloc(&d_b, tensorSize(bShape) * sizeof(float));
	cudaMalloc(&d_c, tensorSize(cShape) * sizeof(float));

	/*CUDA Memcpy for in, filter and bias*/
	cudaMemcpy(d_a, h_a, tensorSize(aShape) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, tensorSize(bShape) * sizeof(float), cudaMemcpyHostToDevice);


	/*Block and Grid Dims*/
	dim3 blockDim(GEMM_TILE_SIZE, GEMM_TILE_SIZE);
	dim3 gridDim((cShape.width + GEMM_TILE_SIZE - 1) / GEMM_TILE_SIZE, 1, cShape.count); // 128x1x3
	size_t sharedMemSize = 2 * GEMM_TILE_SIZE * GEMM_TILE_SIZE * sizeof(float);

	std::cout << "SPEED(?) COPY GPU Starting!\n";
	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	cudaEventRecord(start);
	std::cout << "\tMemory Size: " << sharedMemSize << " Bytes! \n";
	gemmLayer_gpu_speed << <gridDim, blockDim, sharedMemSize>> > (d_a, aShape, d_b, bShape, d_c, cShape);
	//gemmLayer_gpu_v2 << <gridDim, blockDim >> > (d_a, aShape, d_b, bShape, d_c, cShape);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float ms; cudaEventElapsedTime(&ms, start, stop);
	std::cout << "Copy Batch " << BatchSize << " Time: " << ms << " ms\n";
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cout << "Kernel Launch Error: " << cudaGetErrorString(err) << "\n";
	}
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		std::cout << "CUDA Error: " << cudaGetErrorString(err) << "\n";
	}

	cudaMemcpy(h_c, d_c, tensorSize(cShape) * sizeof(float), cudaMemcpyDeviceToHost);

	/* cudaFree() functions */
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	/*CPU Free*/
	free(h_a);
	free(h_b);
	free(h_c);

	return 0;
}

int evaluateGpuGemm_copy (TensorShape aShape, TensorShape bShape,
	TensorShape& cShape, GemmLayerArgs args, uint32_t BatchSize) {

	//cShape = { BatchSize, 1, 1, 4096 };

	cShape.height = aShape.height;
	cShape.width = bShape.width;
	cShape.channels = aShape.channels;
	cShape.count = BatchSize;

	float* h_a = nullptr;
	float* h_b = nullptr;

	int retVal;
	retVal = makeTensor(&h_a, aShape);
	if (retVal != 0) {
		std::cout << "Unable to make tensor \n";
		return -1;
	}
	retVal = makeTensor(&h_b, bShape);
	if (retVal != 0) {
		std::cout << "Unable to make tensor \n";
		return -1;
	}

	/*C (out) Array initialization*/
	float* h_c = (float*)malloc(tensorSize(cShape) * sizeof(float));
	std::cout << "OutShape : " << cShape << " \n";
	
	/*CUDA Malloc for in, out, filter and bias*/

	float* d_a, * d_b, * d_c;
	cudaMalloc(&d_a, tensorSize(aShape) * sizeof(float));
	cudaMalloc(&d_b, tensorSize(bShape) * sizeof(float));
	cudaMalloc(&d_c, tensorSize(cShape) * sizeof(float));

	/*CUDA Memcpy for in, filter and bias*/
	cudaMemcpy(d_a, h_a, tensorSize(aShape) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b , tensorSize(bShape) * sizeof(float), cudaMemcpyHostToDevice);


	/*Block and Grid Dims*/
	dim3 blockDim(GEMM_TILE_SIZE,GEMM_TILE_SIZE);
	dim3 gridDim((cShape.width + GEMM_TILE_SIZE - 1) / GEMM_TILE_SIZE, (cShape.height + GEMM_TILE_SIZE - 1) / GEMM_TILE_SIZE);
	
	std::cout << "Copy GPU Starting!\n";
	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	cudaEventRecord(start);
	//std::cout << "Memory Size: " << sharedMemSize << " Bytes! \n";
	gemmLayer_gpu << <gridDim, blockDim >> > (d_a, aShape, d_b, bShape, d_c, cShape);
	//gemmLayer_gpu_v2 << <gridDim, blockDim >> > (d_a, aShape, d_b, bShape, d_c, cShape);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float ms; cudaEventElapsedTime(&ms, start, stop);
	std::cout << "Copy Batch " << BatchSize << " Time: " << ms << " ms\n";
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cout << "Kernel Launch Error: " << cudaGetErrorString(err) << "\n";
	}
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		std::cout << "CUDA Error: " << cudaGetErrorString(err) << "\n";
	}
	
	cudaMemcpy(h_c, d_c, tensorSize(cShape) * sizeof(float), cudaMemcpyDeviceToHost);

	/* cudaFree() functions */
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	/*CPU Free*/
	free(h_a);
	free(h_b);
	free(h_c);

	return 0;
}

int evaluateGpuGemm_uvm(TensorShape aShape, TensorShape bShape,
	TensorShape& cShape, GemmLayerArgs args, uint32_t BatchSize) {

	//cShape = { BatchSize, 1, 1, 4096 };

	cShape.height = aShape.height;
	cShape.width = bShape.width;
	cShape.channels = aShape.channels;
	cShape.count = BatchSize;

	float* uvm_a = nullptr;
	float* uvm_b = nullptr;
	float* uvm_c = nullptr;

	/*CUDA UVM Babyyy*/
	cudaMallocManaged(&uvm_a, aShape.count * aShape.width * sizeof(float));
	cudaMallocManaged(&uvm_b, bShape.height * bShape.width * sizeof(float));
	cudaMallocManaged(&uvm_c, cShape.count * cShape.width * sizeof(float));

	int retVal;
	retVal = makeTensor_uvm(&uvm_a, aShape);
	if (retVal != 0) {
		std::cout << "Unable to make tensor \n";
		return -1;
	}
	retVal = makeTensor_uvm(&uvm_b, bShape);
	if (retVal != 0) {
		std::cout << "Unable to make tensor \n";
		return -1;
	}

	/*Block and Grid Dims*/
	dim3 blockDim(GEMM_TILE_SIZE, GEMM_TILE_SIZE);
	dim3 gridDim((cShape.width + GEMM_TILE_SIZE - 1) / GEMM_TILE_SIZE, 1, cShape.count);
	

	/*gpuGemm Kernel Call*/
	std::cout << "UVM GPU Starting!\n";
	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	cudaEventRecord(start);
	gemmLayer_gpu_v2 << <gridDim, blockDim >> > (uvm_a, aShape, uvm_b, bShape, uvm_c, cShape);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cout << "Kernel Launch Error: " << cudaGetErrorString(err) << "\n";
	}
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		std::cout << "CUDA Error: " << cudaGetErrorString(err) << "\n";
	}


	float ms; cudaEventElapsedTime(&ms, start, stop);
	std::cout << "UVM Batch " << BatchSize << " Time: " << ms << " ms\n";

	cudaFree(uvm_a);
	cudaFree(uvm_b);
	cudaFree(uvm_c);
	return 0;
}


//	STUDENT: Add functions here (No)
