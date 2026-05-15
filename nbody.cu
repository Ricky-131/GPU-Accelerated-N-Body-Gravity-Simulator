#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>

#define SOFTENING 1e-9f

__global__ void interactBodies(float4 *p, float3 *v, float dt, int n) {
    extern __shared__ float4 shPos[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float3 accel = {0.0f, 0.0f, 0.0f};
    float4 iPos;
    if (i < n) iPos = p[i];

    for (int tile = 0; tile < gridDim.x; tile++) {
        int idx = tile * blockDim.x + threadIdx.x;
        if (idx < n) shPos[threadIdx.x] = p[idx];
        else shPos[threadIdx.x] = {0.0f, 0.0f, 0.0f, 0.0f};

        __syncthreads();

        for (int j = 0; j < blockDim.x; j++) {
            float4 jPos = shPos[j];
            float3 r = {jPos.x - iPos.x, jPos.y - iPos.y, jPos.z - iPos.z};
            float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + SOFTENING;
            float invDist = rsqrtf(distSqr);
            float s = jPos.w * (invDist * invDist * invDist);
            accel.x += r.x * s;
            accel.y += r.y * s;
            accel.z += r.z * s;
        }

        __syncthreads();
    }

    if (i < n) {
        v[i].x += accel.x * dt;
        v[i].y += accel.y * dt;
        v[i].z += accel.z * dt;
        p[i].x += v[i].x * dt;
        p[i].y += v[i].y * dt;
        p[i].z += v[i].z * dt;
    }
}

int main() {
    int n = 4096;
    float dt = 0.01f;
    int bytes_p = n * sizeof(float4);
    int bytes_v = n * sizeof(float3);

    float4 *h_p = (float4*)malloc(bytes_p);
    float3 *h_v = (float3*)malloc(bytes_v);

    for (int i = 0; i < n; i++) {
        h_p[i] = { (float)rand()/RAND_MAX, (float)rand()/RAND_MAX, (float)rand()/RAND_MAX, 1.0f };
        h_v[i] = { 0.0f, 0.0f, 0.0f };
    }

    float4 *d_p;
    float3 *d_v;
    cudaMalloc(&d_p, bytes_p);
    cudaMalloc(&d_v, bytes_v);

    cudaMemcpy(d_p, h_p, bytes_p, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, bytes_v, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    interactBodies<<<gridSize, blockSize>>>(d_p, d_v, dt, n);

    cudaMemcpy(h_p, d_p, bytes_p, cudaMemcpyDeviceToHost);
    int sharedMemSize = blockSize * sizeof(float4);
    interactBodies<<<gridSize, blockSize, sharedMemSize>>>(d_p, d_v, dt, n);

    std::cout << "Success: Particles updated on GPU." << std::endl;

    cudaFree(d_p);
    cudaFree(d_v);
    free(h_p);
    free(h_v);

    return 0;
}