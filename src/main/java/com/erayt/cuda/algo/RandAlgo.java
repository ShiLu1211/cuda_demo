package com.erayt.cuda.algo;

import com.erayt.cuda.GpuAlgorithm;

public class RandAlgo implements GpuAlgorithm {

    int num_paths, num_steps, seed;

    @Override
    public int getId() {
        return 4;
    }

    @Override
    public double[] toArgs() {
        return new double[] { num_paths, num_steps, seed };
    }

    public static RandAlgo of(int num_paths2, int num_steps2, int seed2) {
        RandAlgo algo = new RandAlgo();
        algo.num_paths = num_paths2;
        algo.num_steps = num_steps2;
        algo.seed = seed2;
        return algo;
    }

}
