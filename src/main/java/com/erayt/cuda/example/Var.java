package com.erayt.cuda.example;

import java.security.SecureRandom;

import com.erayt.cuda.GpuInterface;
import com.erayt.cuda.algo.VarAlgo;

public class Var {

    private static final SecureRandom SECURE_RANDOM = new SecureRandom();

    public static void var() {
        System.out.println("Var");
        // === Var
        int n = 1_000_000;
        double[] returns = new double[n];

        for (int i = 0; i < n; i++) {
            // 正态分布模拟历史收益（均值0.001，标准差0.02）
            returns[i] = 0.001 + SECURE_RANDOM.nextGaussian() * 0.02;
        }

        double confidence = 0.99;
        double holding = 1.0;

        // === CPU Var
        long start = System.currentTimeMillis();
        double javaVaR = computeVaR(returns, confidence, holding);
        long end = System.currentTimeMillis();
        System.out.println("CPU Time: " + (end - start) + " ms");
        System.out.println("Java VaR = " + javaVaR);

        // === GPU Var
        start = System.currentTimeMillis();
        VarAlgo varAlgo = VarAlgo.of(n, returns, confidence, holding);
        double[] var = GpuInterface.run(varAlgo.getId(), varAlgo.toArgs());
        end = System.currentTimeMillis();
        System.out.println("GPU Time: " + (end - start) + " ms");
        System.out.println("GPU VaR = " + var[0]);

        System.out.println();
    }

    static double computeVaR(double[] returns, double confidence, double holding) {
        int n = returns.length;
        double[] losses = new double[n];

        double Z = SECURE_RANDOM.nextGaussian();

        for (int i = 0; i < n; i++) {
            losses[i] = holding - returns[i] * (1 + Z * 0.01);
        }

        java.util.Arrays.sort(losses);

        int index = (int) ((1.0 - confidence) * n);
        return losses[index];
    }
}
