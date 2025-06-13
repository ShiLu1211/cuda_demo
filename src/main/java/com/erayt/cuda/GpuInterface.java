package com.erayt.cuda;

public class GpuInterface {
    public static native void quickInit();

    public static native float monteCarlo(int numPaths, float S0, float K, float r, float sigma, float T);

    public static void main(String[] args) {
        NativeLoader.sharedInstance().load();

        quickInit();

        float price = monteCarlo(1_000_000, 100f, 100f, 0.05f, 0.2f, 1f);
        System.out.println("GPU Monte Carlo price: " + price);
    }

}