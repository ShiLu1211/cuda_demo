package com.erayt.cuda.algo;

import com.erayt.cuda.GpuAlgorithm;

public class VarAlgo implements GpuAlgorithm {
    int num;
    float[] returns;
    float confidence, holding;

    public static VarAlgo of(int num, float[] returns, float confidence, float holding) {
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
    public float[] toArgs() {
        float[] args = new float[1 + returns.length + 2];
        args[0] = (float) num;
        System.arraycopy(returns, 0, args, 1, returns.length);
        args[1 + returns.length] = confidence;
        args[2 + returns.length] = holding;
        return args;
    }
}
