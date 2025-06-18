package com.erayt.cuda;

public interface GpuAlgorithm {
    int getId(); // 每个算法唯一标识

    float[] toArgs(); // 转换为 GPU 调用参数
}