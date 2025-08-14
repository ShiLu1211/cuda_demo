package com.erayt.cuda;

public class GpuInterface {
    public static native double[] run(int algoId, double[] args);

    public static native double[] runDouble(int algoId, double[] args);

    public static native double[] runCpu(int algoId, double[] args);
}