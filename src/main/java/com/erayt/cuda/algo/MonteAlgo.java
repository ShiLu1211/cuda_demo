package com.erayt.cuda.algo;

import com.erayt.cuda.GpuAlgorithm;

public class MonteAlgo implements GpuAlgorithm {
    public int numPaths;
    public float S0, K, r, sigma, T;

    public MonteAlgo(int numPaths, float S0, float K, float r, float sigma, float T) {
        this.numPaths = numPaths;
        this.S0 = S0;
        this.K = K;
        this.r = r;
        this.sigma = sigma;
        this.T = T;
    }

    public int getId() {
        return 1;
    }

    public float[] toArgs() {
        return new float[] {
                numPaths, S0, K, r, sigma, T };
    }
}
