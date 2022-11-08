#include "kernel.cuh"

__global__ void step(int *src, int *dst, int width, int height){
    //トーラス型ライフゲームの実装
    //TODO: 共有メモリに256 * interval分コピー
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    const int idy = threadIdx.y + blockDim.y * blockIdx.y;
    if(width <= idx || height <= idy) return;
    int neighbor = 0;
    for(int i = -1;i < 2;i++){
        for(int j = -1;j < 2;j++){
            neighbor += src[(idx + i + width) % width + ((idy + j + height) % height) * width];
        }
    }
    if(src[idx + width * idy] >= 1){
        dst[idx + width * idy] = neighbor == 3 || neighbor == 4 ? src[idx + width * idy] + 1 : 0;
    }else{
        dst[idx + width * idy] = neighbor == 3 ? 1 : 0;
    }
}

__global__ void render(unsigned char *dst, int *data, int width, int height, int pitch, int size){
//画素単位でスレッドを立ててレンダリング
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    const int idy = threadIdx.y + blockDim.y * blockIdx.y;
    const int p = pitch / width;
    if(width <= idx)return;
    if(height <= idy)return;
    for(int i = 0;i < p;i++){
        dst[idx * p + i + idy * pitch] = data[idx / size + idy / size * width] * 255;
    }
}

int game(unsigned char * data, int state, int *init, int width, int height, int pitch){
    static int frame_counter = 0;
    static int *buf1 = nullptr;
    static int *buf2 = nullptr;
    static unsigned char *buf3 = nullptr;
    const dim3 block(256, 1);
    const dim3 grid(width / 256, height);
    if(buf1 == nullptr || buf2 == nullptr){
        cudaMalloc((void **)&buf1, sizeof(int) * width * height);
        cudaMalloc((void **)&buf2, sizeof(int) * width * height);
        cudaMalloc((void **)&buf3, sizeof(unsigned char) * pitch * height);
    }
    if(state == -1){
        cudaFree(buf1);
        cudaFree(buf2);
        cudaFree(buf3);
        return state;
    }
    if(init != nullptr){
        cudaMemcpy(buf1, init, sizeof(int) * width * height, cudaMemcpyHostToDevice);
    }
    if(frame_counter % 2 == 0){
        step<<<grid, block>>>(buf1, buf2, width, height);
        render<<<grid, block>>>(buf3, buf2, width, height, pitch, 1);
    }else{
        step<<<grid, block>>>(buf2, buf1, width, height);
        render<<<grid, block>>>(buf3, buf1, width, height, pitch, 1);
    }
    cudaDeviceSynchronize();
    if(data != nullptr){
    cudaMemcpy(data, buf3, sizeof(unsigned char) * pitch * height, cudaMemcpyDeviceToHost);
    }
    return frame_counter++;
}