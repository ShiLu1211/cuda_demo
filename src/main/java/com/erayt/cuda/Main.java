package com.erayt.cuda;

import java.util.Random;

import com.erayt.cuda.algo.MonteAlgo;
import com.erayt.cuda.algo.VarAlgo;

public class Main {
    static {
        NativeLoader.sharedInstance().load();
    }

    public static void main(String[] args) {
        GpuInterface.cudaInit();

        // === Monte Carlo 模拟
        int numPaths = 10_000_000;
        float S0 = 100f;
        float K = 100f;
        float r = 0.05f;
        float sigma = 0.2f;
        float T = 1f;

        // === CPU Monte Carlo
        long start = System.currentTimeMillis();
        float price = europeanCallPrice(numPaths, S0, K, r, sigma, T);
        long end = System.currentTimeMillis();
        System.out.println("Time: " + (end - start));
        System.out.println("CPU Monte Carlo price: " + price);

        // === GPU Monte Carlo
        start = System.currentTimeMillis();
        MonteAlgo algo = MonteAlgo.of(numPaths, S0, K, r, sigma, T);
        float[] discounted = GpuInterface.run(algo.getId(), algo.toArgs());
        end = System.currentTimeMillis();
        System.out.println("Time: " + (end - start));
        System.out.println("GPU Monte Carlo price: " + discounted[0]);

        // === Var
        int n = 1_000_000;
        float[] returns = new float[n];
        Random rand = new Random(42);

        for (int i = 0; i < n; i++) {
            // 正态分布模拟历史收益（均值0.001，标准差0.02）
            returns[i] = (float) (0.001 + rand.nextGaussian() * 0.02);
        }

        float confidence = 0.99f;
        float holding = 1.0f;

        // === CPU Var
        start = System.currentTimeMillis();
        float javaVaR = computeVaR(returns, confidence, holding);
        end = System.currentTimeMillis();
        System.out.println("Time: " + (end - start));
        System.out.println("Java VaR = " + javaVaR);

        // === GPU Var
        start = System.currentTimeMillis();
        VarAlgo varAlgo = VarAlgo.of(n, returns, confidence, holding);
        float[] var = GpuInterface.run(varAlgo.getId(), varAlgo.toArgs());
        end = System.currentTimeMillis();
        System.out.println("Time: " + (end - start));
        System.out.println("GPU VaR = " + var[0]);
    }

    public static float europeanCallPrice(
            int numPaths,
            float s0,
            float k,
            float r,
            float sigma,
            float t) {
        Random rand = new Random();
        float sumPayoff = 0.0f;

        for (int i = 0; i < numPaths; i++) {
            // 生成一个标准正态分布随机数 Z
            float z = (float) rand.nextGaussian();

            // 计算终止价格 S_T
            float st = (float) (s0 * Math.exp((r - 0.5 * sigma * sigma) * t + sigma * Math.sqrt(t) * z));

            // 看涨期权 payoff
            float payoff = Math.max(st - k, 0.0f);

            sumPayoff += payoff;
        }

        float meanPayoff = sumPayoff / numPaths;

        // 折现回当前价格
        return (float) (Math.exp(-r * t) * meanPayoff);
    }

    public static float computeVaR(float[] returns, float confidence, float holding) {
        int n = returns.length;
        float[] losses = new float[n];
        Random rand = new Random();

        float Z = (float) rand.nextGaussian();

        for (int i = 0; i < n; i++) {
            losses[i] = holding - returns[i] * (1 + Z * 0.01f);
        }

        java.util.Arrays.sort(losses);

        int index = (int) ((1.0f - confidence) * n);
        return losses[index];
    }
}
