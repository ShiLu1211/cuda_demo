package com.erayt.cuda.algo;

import com.erayt.cuda.GpuAlgorithm;

public class VolSurface implements GpuAlgorithm {
    int n;
    double[] S, K, T, r, P;

    public static VolSurface of(int n, double[] S, double[] K, double[] T, double[] r, double[] P) {
        VolSurface volSurface = new VolSurface();
        volSurface.n = n;
        volSurface.S = S;
        volSurface.K = K;
        volSurface.T = T;
        volSurface.r = r;
        volSurface.P = P;
        return volSurface;
    }

    @Override
    public int getId() {
        return 3;
    }

    @Override
    public double[] toArgs() {
        int totalLen = 1 + 5 * n; // n + S K T r P
        double[] args = new double[totalLen];
        args[0] = n;

        System.arraycopy(S, 0, args, 1, n);
        System.arraycopy(K, 0, args, 1 + n, n);
        System.arraycopy(T, 0, args, 1 + 2 * n, n);
        System.arraycopy(r, 0, args, 1 + 3 * n, n);
        System.arraycopy(P, 0, args, 1 + 4 * n, n);

        return args;
    }

}
