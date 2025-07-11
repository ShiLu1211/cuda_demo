package com.erayt.cuda.algo;

import com.erayt.cuda.GpuAlgorithm;

public class MonteAlgo implements GpuAlgorithm {
    public int numPaths;
    public double S0, K, r, sigma, T;

    public static MonteAlgo of(int numPaths, double S0, double K, double r, double sigma, double T) {
        MonteAlgo algo = new MonteAlgo();
        algo.numPaths = numPaths;
        algo.S0 = S0;
        algo.K = K;
        algo.r = r;
        algo.sigma = sigma;
        algo.T = T;
        return algo;
    }

    @Override
    public int getId() {
        return 1;
    }

    @Override
    public double[] toArgs() {
        return new double[] {
                numPaths, S0, K, r, sigma, T };
    }
}
