package com.erayt.cuda;

public class GpuInterface {
    public static native void cudaInit();

    public static native double[] run(int algoId, double[] args);
}