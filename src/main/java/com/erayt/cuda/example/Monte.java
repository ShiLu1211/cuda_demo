package com.erayt.cuda.example;

import java.util.Random;

import com.erayt.cuda.GpuInterface;
import com.erayt.cuda.algo.MonteAlgo;

public class Monte {

    private static final Random RAND = new Random();
    public static void monto() {
        System.out.println("Monte Carlo");
        // === Monte Carlo 模拟
        int numPaths = 10_000_000;
        double S0 = 100;
        double K = 100;
        double r = 0.05;
        double sigma = 0.2;
        double T = 1;

        // === CPU Monte Carlo
        long start = System.currentTimeMillis();
        double price = europeanCallPrice(numPaths, S0, K, r, sigma, T);
        long end = System.currentTimeMillis();
        System.out.println("CPU Time: " + (end - start) + " ms");
        System.out.println("CPU Monte Carlo price: " + price);

        // === GPU Monte Carlo
        start = System.currentTimeMillis();
        MonteAlgo algo = MonteAlgo.of(numPaths, S0, K, r, sigma, T);
        double[] discounted = GpuInterface.run(algo.getId(), algo.toArgs());
        end = System.currentTimeMillis();
        System.out.println("GPU Time: " + (end - start) + " ms");
        System.out.println("GPU Monte Carlo price: " + discounted[0]);

        System.out.println();
    }

    static double europeanCallPrice(
            int numPaths,
            double s0,
            double k,
            double r,
            double sigma,
            double t) {
        double sumPayoff = 0.0;

        for (int i = 0; i < numPaths; i++) {
            // 生成一个标准正态分布随机数 Z
            double z = RAND.nextGaussian();

            // 计算终止价格 S_T
            double st = s0 * Math.exp((r - 0.5 * sigma * sigma) * t + sigma * Math.sqrt(t) * z);

            // 看涨期权 payoff
            double payoff = Math.max(st - k, 0.0);

            sumPayoff += payoff;
        }

        double meanPayoff = sumPayoff / numPaths;

        // 折现回当前价格
        return Math.exp(-r * t) * meanPayoff;
    }
}
