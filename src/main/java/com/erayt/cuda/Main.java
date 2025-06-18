package com.erayt.cuda;

import java.util.Random;

import com.erayt.cuda.algo.MonteAlgo;

public class Main {
    static {
        NativeLoader.sharedInstance().load();
    }

    public static void main(String[] args) {
        GpuInterface.cudaInit();

        MonteAlgo algo = new MonteAlgo(10_000_000, 100f, 100f, 0.05f, 0.2f, 1f);

        long start = System.currentTimeMillis();
        float[] discounted = GpuInterface.run(algo.getId(), algo.toArgs());
        long end = System.currentTimeMillis();
        System.out.println("Time: " + (end - start));
        System.out.println("GPU Monte Carlo price: " + discounted[0]);

        start = System.currentTimeMillis();
        float price = europeanCallPrice(10_000_000, 100f, 100f, 0.05f, 0.2f, 1f);
        end = System.currentTimeMillis();
        System.out.println("Time: " + (end - start));
        System.out.println("CPU European Call price: " + price);
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
}
