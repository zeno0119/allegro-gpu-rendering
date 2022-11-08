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
            neighbor += src[(idx + i + width) % width + ((idy + j + height) % height) * width] >= 1 ? 1 : 0;
        }
    }
    if(src[idx + width * idy] >= 1){
        dst[idx + width * idy] = neighbor == 3 || neighbor == 4 ? src[idx + width * idy] + 1 : 0;
    }else{
        dst[idx + width * idy] = neighbor == 3 ? 1 : 0;
    }
}

__global__ void render(unsigned int *dst, int *data, int width, int height, int pitch, int size){
//画素単位でスレッドを立ててレンダリング
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    const int idy = threadIdx.y + blockDim.y * blockIdx.y;
    const int p = pitch / width;
    if(width <= idx)return;
    if(height <= idy)return;
    for(int i = 0;i < p;i++){
        dst[idx + idy * width] = 0;
    }
    if(data[idx / size + idy / size * width] == 0)return;
    float h = data[idx / size + idy / size * width] % 360;
    unsigned int d = 0;
    if(0 <= h && h < 60){
        d = (255 << 16) + ((unsigned int)(h / 60 * 255) << 8);
    }else if(60 <= h && h < 120){
        d = ((unsigned int)((120 - h) / 60 * 255) << 16) + (255 << 8);
    }else if(120 <= h && h < 180){
        d = (255 << 8) + ((unsigned int)((h - 120) / 60 * 255));
    }else if(180 <= h && h < 240){
        d = ((unsigned int)((240 - h) / 60 * 255) << 8) + (255);
    }else if(240 <= h && h < 300){
        d = ((unsigned int)((h - 240) / 60 * 255) << 16) + (255);
    }else if(300 <= h && h < 360){
        d = (255 << 16) + (((unsigned int)(360 - h) / 60 * 255));
    }
    dst[idx + idy * width] = d << 8;
    // for(int i = 0;i < p;i++){
    //     dst[idx * p + i + idy * pitch] = 255;
    // }
}

int game(unsigned int * data, int state, int *init, int width, int height, int pitch){
    static int frame_counter = 0;
    static int *buf1 = nullptr;
    static int *buf2 = nullptr;
    static unsigned int *buf3 = nullptr;
    const dim3 block(256, 1);
    const dim3 grid(width / 256, height);
    if(buf1 == nullptr || buf2 == nullptr){
        cudaMalloc((void **)&buf1, sizeof(int) * width * height);
        cudaMalloc((void **)&buf2, sizeof(int) * width * height);
        cudaMalloc((void **)&buf3, sizeof(unsigned int) * width * height);
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
    cudaMemcpy(data, buf3, sizeof(unsigned int) * width * height, cudaMemcpyDeviceToHost);
    }
    return frame_counter++;
}