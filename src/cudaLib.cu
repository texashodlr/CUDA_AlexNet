#include "cudaLib.cuh"
#define TILE_SIZE 3

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
	executeGpuConv(iShape, fShape, oShape, convArgs);
	
	//uint64_t errorCount = evaluateGpuConv(iShape, fShape, oShape, convArgs);
	//std::cout << "Found " << errorCount << " / " << tensorSize(oShape) << " errors \n";
	return 0;
}

int executeGpuConv(TensorShape iShape, TensorShape fShape,
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
		return -1;
	}
	retVal = makeTensor(&h_filter, fShape);
	if (retVal != 0) {
		std::cout << "Unable to make tensor \n";
		return -1;
	}
	retVal = makeVector(&h_bias, oShape.channels);
	if (retVal != 0) {
		std::cout << "Unable to make vector \n";
		return -1;
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

	/*Block and Grid Dims*/
	dim3 blockDim(TILE_SIZE, TILE_SIZE, 1);
	dim3 gridDim((oShape.width + TILE_SIZE-1) / TILE_SIZE, (oShape.height + TILE_SIZE-1) / TILE_SIZE, oShape.channels);

	int shared_window_height = ((TILE_SIZE - 1) * args.strideH) + fShape.height; // 31
	int shared_window_width = ((TILE_SIZE - 1) * args.strideW) + fShape.width;  // 31

	size_t sharedMemSize = (iShape.channels * (shared_window_height) * (shared_window_width)) * sizeof(float); // 4B*(3*31*31) = 11,5KB

	

	/*ConvLayer Kernel Call*/
	//convLayer_gpu << <gridDim, blockDim >> > (d_in, iShape, d_filter, fShape, d_bias, d_out, oShape, args);
	std::cout << "\n\n GPU Starting!\n\n\n";
	std::cout << "Bias[0] = " << h_bias[0] << "\n";
	std::cout << "Memory Size: " << ((iShape.channels * (shared_window_height) * (shared_window_width)) * 4) << " Bytes! \n";
	convLayer_gpu_SM_DM_v3 << <gridDim, blockDim, sharedMemSize >> > (d_in, iShape, d_filter, fShape, d_bias, d_out, oShape, args);
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
	int verify_errors = verifyVector_convLayer(cpu_out, h_out, (oShape.height * oShape.width * oShape.channels));
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
	return 0;
}

uint64_t evaluateGpuConv (TensorShape iShape, TensorShape fShape, 
	TensorShape & oShape, ConvLayerArgs args) {

	uint64_t errorCount = 0;

	//	STUDENT: Add code here

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

__global__ void convLayer_gpu_SM_DM(float* input, TensorShape iShape, float* filter,
	TensorShape fShape, float* bias, float* output, TensorShape oShape, ConvLayerArgs args) {

	extern __shared__ float tile[]; // 26x26 for 2.7KB

	/*Local TB Coords*/
	int shared_x = threadIdx.x;
	int shared_y = threadIdx.y;

	/*Sizing*/
	int shared_dim = TILE_SIZE + fShape.height - 1;

	/*Using my local TB Coords I'm orienting myself globally to then load into shMem*/

	/*Input Coords to load into tile[]*/
	int input_x = blockIdx.x * blockDim.x * args.strideW + shared_x - (fShape.width / 2);
	int input_y = blockIdx.y * blockDim.y * args.strideH + shared_y - (fShape.height / 2);

	//printf("Input_x: %d | Input_y: %d\n", input_x, input_y);

	if ((input_x >= 0 && input_x < iShape.width) && (input_y >= 0 && input_y < iShape.height)) {
		for (uint32_t k = 0; k < iShape.channels;++k) {
			tile[shared_y * shared_dim + shared_x] = input[((k*iShape.height+input_y) * iShape.width) + input_x];
		}		
	}
	else {
		tile[shared_y * shared_dim + shared_x] = 0.0f;
		//tile[shared_y * shared_dim + shared_x] = input[(0 * iShape.height + input_y) * iShape.width + input_x];

		//printf("OOB Baby!");
	}

	__syncthreads();

	/*Coordinates*/
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int channels = blockIdx.z;

	if (col < oShape.width && row < oShape.height) {
		/*Output Matrix Index*/
		uint32_t out_idx = ((channels * oShape.height + row) * oShape.width + col);
		float shared_sum = bias[channels];

		//CPU Code For-loop repeat for the filter window//
		for (uint32_t i = 0; i < fShape.height; ++i) {
			for (uint32_t j = 0; j < fShape.width; ++j) {
				for (uint32_t k = 0; k < fShape.channels; ++k) {

					/*Index+offset of the shMem location*/
					uint32_t sharedMem_x = shared_x + j;
					uint32_t sharedMem_y = shared_y + i;

					/*Filter Index*/
					uint32_t filter_idx = ((channels * fShape.channels + k) * fShape.height + i) * fShape.width + j;
					
					if (sharedMem_x < shared_dim && sharedMem_y < shared_dim) {
						shared_sum += tile[sharedMem_y * shared_dim + sharedMem_x] * filter[filter_idx];
					}
				}
			}
		}
		if (args.activation) {
			output[out_idx] = fmaxf(0.0f, shared_sum);
		}
		else {
			output[out_idx] = shared_sum;
		}
	}
}

__global__ void convLayer_gpu_SM_DM_v2(float* input, TensorShape iShape, float* filter,
	TensorShape fShape, float* bias, float* output, TensorShape oShape, ConvLayerArgs args) {

	extern __shared__ float tile[]; // 26x26 for 2.7KB
	const int tile_height = TILE_SIZE + fShape.height - 1; //26
	const int tile_width = TILE_SIZE + fShape.width - 1; //26
	
	/*Local TB Coords*/
	int shared_x = threadIdx.x;
	int shared_y = threadIdx.y;

	/*Output Coods*/
	int out_x = blockIdx.x * TILE_SIZE + shared_x;
	int out_y = blockIdx.y * TILE_SIZE + shared_y;
	int out_z = blockIdx.z; //0-95 for AlexNet (96)

	for (uint32_t k = 0; k < iShape.channels;++k) {
		int input_x = blockIdx.x * TILE_SIZE * args.strideW + shared_x;
		int input_y = blockIdx.y * TILE_SIZE * args.strideH + shared_y;
		int shared_idx = k * tile_height * tile_width + shared_y * tile_width + shared_x;
	
		if ((input_x >= 0 && input_x < iShape.width) && (input_y >= 0 && input_y < iShape.height)) {
			int in_idx = (k * iShape.height + input_y) * iShape.width + input_x;
			tile[shared_idx] = input[in_idx];
		}
		else {
			tile[shared_idx] = 0.0f;
		}
	}

	__syncthreads();

	/*Coordinates*/
	if (out_x < oShape.width && out_y < oShape.height) {
		/*Output Matrix Index*/
		uint32_t out_idx = ((out_z * oShape.height + out_y) * oShape.width + out_x);
		float shared_sum = bias[out_z];

		//CPU Code For-loop repeat for the filter window//
		for (uint32_t k = 0; k < fShape.channels; ++k) {
			float channel_sum = 0.0f;
			for (uint32_t i = 0; i < fShape.height; ++i) {
				for (uint32_t j = 0; j < fShape.width; ++j) {
					int in_x = out_x * args.strideW + j;  // Absolute input coord
					int in_y = out_y * args.strideH + i;
					int tile_x = in_x - (blockIdx.x * TILE_SIZE * args.strideW);  // Offset within tile
					int tile_y = in_y - (blockIdx.y * TILE_SIZE * args.strideH);

					if (in_x < iShape.width && in_y < iShape.height && tile_x>= 0 && tile_x < tile_width && tile_y >= 0 && tile_y < tile_height) {
						uint32_t shared_idx = k * tile_height * tile_width + tile_y * tile_width + tile_x;
						uint32_t filter_idx = (out_z * fShape.channels + k) * fShape.height * fShape.width + i * fShape.width + j;
						channel_sum += tile[shared_idx] * filter[filter_idx];
					}
					if (out_x == 2 && out_y == 0 && out_z == 0 && k == 0 && i == 0 && j == 0) {
						printf("tile_x = %d, tile_y = %d\n", tile_x, tile_y);
					}
				}
			}
			shared_sum += channel_sum;
			if (out_x == 2 && out_y == 0 && out_z == 0) {
				printf("Channel %d contribution: %f, Running sum: %f\n", k, channel_sum, shared_sum);
			}
		}
		if (args.activation) {
			output[out_idx] = fmaxf(0.0f, shared_sum);
		}
		else {
			output[out_idx] = shared_sum;
		}
	}
}

__global__ void convLayer_gpu_SM_DM_v3(float* input, TensorShape iShape, float* filter,
	TensorShape fShape, float* bias, float* output, TensorShape oShape, ConvLayerArgs args) {

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

	evaluateGpuGemm();
	return 0;
}

int evaluateGpuGemm () {

	return 0;
}

//	STUDENT: Add functions here
