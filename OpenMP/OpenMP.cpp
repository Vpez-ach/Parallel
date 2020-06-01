#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <random>
#include <chrono>
#include <iostream>

void gemm(long* A, long* B, long* C, int n)
{
	int i, j, k;
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			long dot = 0;
			for (k = 0; k < n; k++) {
				dot += A[i * n + k] * B[k * n + j];
			}
			C[i * n + j] = dot;
		}
	}
}

void gemm_omp(long* A, long* B, long* C, int n)
{
#pragma omp parallel
	{
		int i, j, k;
#pragma omp for
		for (i = 0; i < n; i++) {
			for (j = 0; j < n; j++) {
				long dot = 0;
				for (k = 0; k < n; k++) {
					dot += A[i * n + k] * B[k * n + j];
				}
				C[i * n + j] = dot;
			}
		}

	}
}
 
int main() {
	int i, n;
	long* A, * B, * C;

	n = 1200;
	A = (long*)malloc(sizeof(long) * n * n);
	B = (long*)malloc(sizeof(long) * n * n);
	C = (long*)malloc(sizeof(long) * n * n);

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<long> dis(0, 1000000);
	for (i = 0; i < n * n; i++)
	{
		A[i] = dis(gen);
		B[i] = dis(gen);
	}
	std::cout << "Processing...\n";

	auto start1 = std::chrono::steady_clock::now();
	gemm(A, B, C, n);
	auto end1 = std::chrono::steady_clock::now();
	std::cout << "Sequenced -> " << std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count() << " ms\n";

	auto start2 = std::chrono::steady_clock::now();
	gemm_omp(A, B, C, n);
	auto end2 = std::chrono::steady_clock::now();
	std::cout << "Parallel -> " << std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count() << " ms\n";

	return 0;

}