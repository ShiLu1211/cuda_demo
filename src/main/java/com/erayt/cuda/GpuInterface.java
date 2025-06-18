package com.erayt.cuda;

public class GpuInterface {
    public static native void cudaInit();

    public static native float[] run(int algoId, float[] args);
}