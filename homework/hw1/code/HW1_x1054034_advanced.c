/*************************************************************************
    > File Name: ParallelOddEvenSort.c
    > Author: Zhang Yuhui
    > Mail: yuhui-zh15@mails.tsinghua.edu.cn 
    > Created Time: Wed 05 Jul 2017 10:45:32 AM CST
 ************************************************************************/
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <memory.h>

// Compare function
int cmp(const void* a, const void* b) {
	if (*(float*)a > *(float*)b) return 1;
	else return -1;
}

// Merge items ensure a is not bigger than b
int Merge(float* a, float* b, int len_a, int len_b, int mode) {
	int ret = 1;	
	float* c = (float*)malloc(sizeof(float) * len_a);
	if (mode == 0) {
		int i = 0, j = 0, k = 0;
		while (i != len_a && j != len_b && k != len_a) {
			if (a[i] <= b[j]) c[k++] = a[i++];
			else { c[k++] = b[j++]; ret = 0; }
		}
		while (i != len_a && k != len_a) c[k++] = a[i++];
		while (j != len_b && k != len_a) c[k++] = b[j++];
	}
	else {
		int i = len_a - 1, j = len_b - 1, k = len_a - 1;
		while (i != -1 && j != -1 && k != -1) {
			if (a[i] >= b[j]) c[k--] = a[i--];
			else { c[k--] = b[j--]; ret = 0; }
		}
		while (i != -1 && k != -1) c[k--] = a[i--];
		while (j != -1 && k != -1) c[k--] = b[j--];
	}
	memcpy(a, c, sizeof(float) * len_a);
	free(c);
	return ret;
}

int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);
	int rank, size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int n = atoi(argv[1]);
	float* array;
	float* arr;
	int len_arr;

	// Input
	if (rank == 0) {
		FILE* fin = fopen(argv[2], "rb");
		if (fin == NULL) { perror("Error: file input error.\n"); return -1; }
		array = (float*)malloc(sizeof(float) * n);
		fread(array, sizeof(float), n, fin);
		fclose(fin);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	
	// Initialize
	len_arr = n / size;
	if (rank == size - 1) len_arr = n - (size - 1) * len_arr;
	arr = (float*)malloc(sizeof(float) * len_arr);
	
	// Scatter
	if (rank == 0) {
		memcpy(arr, array, sizeof(float) * len_arr);
		int i;
		for (i = 1; i < size - 1; i++) {
			MPI_Send(array + i * len_arr, len_arr, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
		}
		if (size > 1) {
			MPI_Send(array + (size - 1) * len_arr, n - (size - 1) * len_arr, MPI_FLOAT, size - 1, 0, MPI_COMM_WORLD);
		}
	}
	else {
		MPI_Status status;
		MPI_Recv(arr, len_arr, MPI_FLOAT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
	}
	qsort(arr, len_arr, sizeof(float), cmp);
	MPI_Barrier(MPI_COMM_WORLD);

	// Odd Even Sort
	int sorted = 0;
	int tmpsorted = 0;
	while (!sorted) {
		sorted = 1;
		tmpsorted = 1;
		if (rank % 2 == 1) { // odd	
			int len_arr2 = n / size;
			float* arr2 = (float*)malloc(sizeof(float) * len_arr2);
			MPI_Status status;
			MPI_Send(arr, len_arr, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD);
			MPI_Recv(arr2, len_arr2, MPI_FLOAT, rank - 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			if (!Merge(arr, arr2, len_arr, len_arr2, 1)) tmpsorted = 0;
			free(arr2);
		}
		else { // even
			if (rank != size - 1) {
				int len_arr2 = (rank == size - 2)? (n - (size - 1) * len_arr): len_arr;
				float* arr2 = (float*)malloc(sizeof(float) * len_arr2);
				MPI_Status status;
				MPI_Recv(arr2, len_arr2, MPI_FLOAT, rank + 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
				MPI_Send(arr, len_arr, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
				if (!Merge(arr, arr2, len_arr, len_arr2, 0)) tmpsorted = 0;
				free(arr2);
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);

		if (rank % 2 == 0) {
			if (rank != 0) {
				int len_arr2 = n / size;
				float* arr2 = (float*)malloc(sizeof(float) * len_arr2);
				MPI_Status status;
				MPI_Send(arr, len_arr, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD);
				MPI_Recv(arr2, len_arr2, MPI_FLOAT, rank - 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
				if (!Merge(arr, arr2, len_arr, len_arr2, 1)) tmpsorted = 0;
				free(arr2);
			}
		}
		else {
			if (rank != size - 1) {
				int len_arr2 = (rank == size - 2)? (n - (size - 1) * len_arr): len_arr;
				float* arr2 = (float*)malloc(sizeof(float) * len_arr2);
				MPI_Status status;
				MPI_Recv(arr2, len_arr2, MPI_FLOAT, rank + 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
				MPI_Send(arr, len_arr, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
				if (!Merge(arr, arr2, len_arr, len_arr2, 0)) tmpsorted = 0;
				free(arr2);
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);

		MPI_Allreduce(&tmpsorted, &sorted, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
	}
	
	// Output
	MPI_Barrier(MPI_COMM_WORLD);
	if (rank == 0) {
		memcpy(array, arr, sizeof(float) * len_arr);
		int i;
		MPI_Status status;
		for (i = 1; i < size - 1; i++) {
			MPI_Recv(array + i * len_arr, len_arr, MPI_FLOAT, i, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		}
		if (size > 1) {
			MPI_Recv(array + (size - 1) * len_arr, n - (size - 1) * len_arr, MPI_FLOAT, size - 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		}
		
		FILE* fout = fopen(argv[3], "wb");
		if (fout == NULL) { perror("Error: file output error.\n"); return -1; }
		fwrite(array, sizeof(float), n, fout);	
		fclose(fout);
		free(arr);
		free(array);
	}
	else {
		MPI_Send(arr, len_arr, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
		free(arr);
	}

	MPI_Finalize();
	return 0;
}
