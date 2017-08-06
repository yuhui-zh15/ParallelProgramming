#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#define NUM_GPUS 2

const int INF = 10000000;
int *host_D;
int *dev_D[NUM_GPUS];
int n, m;

void Input(char *inFileName) {
	FILE *infile = fopen(inFileName, "r");
	setvbuf(infile, new char[1 << 20], _IOFBF, 1 << 20);
	fscanf(infile, "%d %d", &n, &m);
    host_D = (int*)malloc(n * n * sizeof(int));
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			if (i == j) host_D[i * n + j] = 0;
			else host_D[i * n + j] = INF;
		}
	}
	while (--m >= 0) {
		int a, b, v;
		fscanf(infile, "%d %d %d", &a, &b, &v);
		host_D[(a - 1) * n + (b - 1)] = v;
	}
	fclose(infile);
}

void Output(char *outFileName) {
	FILE *outfile = fopen(outFileName, "w");
	setvbuf(outfile, new char[1 << 20], _IOFBF, 1 << 20);
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			if (host_D[i * n + j] >= INF) fprintf(outfile, "INF ");
			else fprintf(outfile, "%d ", host_D[i * n + j]);
		}
		fprintf(outfile, "\n");
	}        
	fclose(outfile);
}

__global__ void func1(int n, int B, int k, int* arr) {
	extern __shared__ int shared_memory[];
    int* dBlock = shared_memory;
    int i = threadIdx.x / B;
    int j = threadIdx.x % B;
	int x = i + k * B;
	int y = j + k * B;
    dBlock[threadIdx.x] = (x < n && y < n)? arr[x * n + y]: INF;
    for (int l = 0; l < B; l++) {
        __syncthreads();
        int temp = dBlock[(i * B) + l] + dBlock[(l * B) + j];
        if (dBlock[threadIdx.x] > temp) {
            dBlock[threadIdx.x] = temp;
        }
    }
	if (x < n && y < n) arr[x * n + y] = dBlock[threadIdx.x];
}

__global__ void func2(int n, int B, int k, int* arr) {
	if (blockIdx.x == k) return;
	extern __shared__ int shared_memory[];
	int* dBlock = shared_memory;
    int* cBlock = &shared_memory[B * B];
    int i = threadIdx.x / B;
    int j = threadIdx.x % B;
	int x = i + k * B;
	int y = j + k * B;
    dBlock[threadIdx.x] = (x < n && y < n)? arr[x * n + y]: INF;
	if (blockIdx.y != 0) x = i + blockIdx.x * B;
	if (blockIdx.y == 0) y = j + blockIdx.x * B;
    cBlock[threadIdx.x] = (x < n && y < n)? arr[x * n + y]: INF;
    for (int l = 0; l < B; l++) {
        __syncthreads();
        int temp = (blockIdx.y == 0)? dBlock[i * B + l] + cBlock[l * B + j]: cBlock[i * B + l] + dBlock[l * B + j];
        if (cBlock[threadIdx.x] > temp) {
            cBlock[threadIdx.x] = temp;
        }
    }
    if (x < n && y < n) arr[x * n + y] = cBlock[threadIdx.x];
}

__global__ void func3(int n, int B, int k, int* arr, int offset) {
	if ((blockIdx.x + offset == k) || blockIdx.y == k) return;
	extern __shared__ int shared_memory[];
    int* dyBlock = shared_memory;
    int* dxBlock = &shared_memory[B * B];
    int i = threadIdx.x / B;
    int j = threadIdx.x % B;
	int x = i + k * B;
	int y = j + blockIdx.y * B;
    dxBlock[threadIdx.x] = (x < n && y < n)? arr[x * n + y]: INF;
	x = i + (blockIdx.x + offset) * B;
	y = j + k * B;
    dyBlock[threadIdx.x] = (x < n && y < n)? arr[x * n + y]: INF;
	x = i + (blockIdx.x + offset) * B;
	y = j + blockIdx.y * B;
    int dist = (x < n && y < n)? arr[x * n + y]: INF;
    __syncthreads();
    for (int l = 0; l < B; l++) {
        int temp = dyBlock[i * B + l] + dxBlock[l * B + j];
        if (dist > temp) {
            dist = temp;
        }
    }
	if (x < n && y < n) arr[x * n + y] = dist;
}

void Block(int B) {
	omp_set_num_threads(NUM_GPUS);
	int round = (n + B - 1) / B;
	int offset = (round + NUM_GPUS - 1) / NUM_GPUS;
	int num[2] = { offset * B * n, n * n - offset * B * n };
	dim3 bk1(1, 1);
    dim3 bk2(round, 2);
    dim3 bk3(offset, round);
    int gputhreads = B * B;
	#pragma omp parallel 
	{
		unsigned int thread_id = omp_get_thread_num();
		cudaSetDevice(thread_id);
		cudaDeviceEnablePeerAccess(!thread_id, 0);
		cudaMalloc(&dev_D[thread_id], n * n * sizeof(int));
		cudaMemcpy(dev_D[thread_id], host_D, n * n * sizeof(int), cudaMemcpyHostToDevice);
		for (int k = 0; k < round; k++) {
			func1<<<bk1, gputhreads, gputhreads * sizeof(int)>>>(n, B, k, dev_D[thread_id]);
			func2<<<bk2, gputhreads, 2 * gputhreads * sizeof(int)>>>(n, B, k, dev_D[thread_id]);
			func3<<<bk3, gputhreads, 2 * gputhreads * sizeof(int)>>>(n, B, k, dev_D[thread_id], thread_id * offset);
			#pragma omp barrier
			if (thread_id == 0) cudaMemcpyPeer(dev_D[1], 1, dev_D[0], 0, num[0] * sizeof(int));
			else cudaMemcpyPeer(dev_D[0] + num[0], 0, dev_D[1] + num[0], 1, num[1] * sizeof(int));
			#pragma omp barrier
		}
	}
    cudaMemcpy(host_D, dev_D[0], n * n * sizeof(int), cudaMemcpyDeviceToHost);
}

int main(int argc, char **argv) {
	Input(argv[1]);
	int B = atoi(argv[3]);
	Block(B);
	Output(argv[2]);
    return 0;
}
