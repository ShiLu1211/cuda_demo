package com.erayt.cuda.algo;

import com.erayt.cuda.GpuAlgorithm;

public class BSAlgo implements GpuAlgorithm {

    double[][] randMatrix;
    double S0, r, q, v, t;

    public static BSAlgo of(double S0, double r, double q, double v, double t, double[][] randMatrix) {
        BSAlgo algo = new BSAlgo();
        algo.S0 = S0;
        algo.r = r;
        algo.q = q;
        algo.v = v;
        algo.t = t;
        algo.randMatrix = randMatrix;
        return algo;
    }

    @Override
    public int getId() {
        return 5;
    }

    public int getId2() {
        return 6;
    }

    public double[] toArgs2() {
        return new double[] { S0, r, q, v, t};
    }

    @Override
    public double[] toArgs() {
        int paths = randMatrix.length;
        int steps = randMatrix[0].length;

        double[] args = new double[7 + paths * steps];
        int idx = 0;

        // 1. 基础参数
        args[idx++] = S0;
        args[idx++] = r;
        args[idx++] = q;
        args[idx++] = v;
        args[idx++] = t;
        args[idx++] = paths;
        args[idx++] = steps;

        // 2. 拍平随机矩阵（paths 在前，steps 在后）
        for (int i = 0; i < paths; i++) {
            System.arraycopy(randMatrix[i], 0, args, idx, steps);
            idx += steps;
        }

        return args;
    }

}
