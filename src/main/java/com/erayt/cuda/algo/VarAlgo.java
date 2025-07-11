package com.erayt.cuda.algo;

import com.erayt.cuda.GpuAlgorithm;

public class VarAlgo implements GpuAlgorithm {
    int num;
    double[] returns;
    double confidence, holding;

    public static VarAlgo of(int num, double[] returns, double confidence, double holding) {
        VarAlgo algo = new VarAlgo();
        algo.num = num;
        algo.returns = returns;
        algo.confidence = confidence;
        algo.holding = holding;
        return algo;
    }

    @Override
    public int getId() {
        return 2;
    }

    @Override
    public double[] toArgs() {
        double[] args = new double[1 + returns.length + 2];
        args[0] = (double) num;
        System.arraycopy(returns, 0, args, 1, returns.length);
        args[1 + returns.length] = confidence;
        args[2 + returns.length] = holding;
        return args;
    }
}
