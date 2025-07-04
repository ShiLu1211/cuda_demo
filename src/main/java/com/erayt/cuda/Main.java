package com.erayt.cuda;

import java.util.Random;

import com.erayt.cuda.algo.MonteAlgo;
import com.erayt.cuda.algo.VarAlgo;
import com.erayt.cuda.algo.VolSurface;

public class Main {
    static {
        NativeLoader.sharedInstance().load();
    }

    public static void main(String[] args) {
        GpuInterface.cudaInit();

        monto();

        var();

        volSurface();
    }

    static void monto() {
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
    }

    static void var() {
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
        long start = System.currentTimeMillis();
        float javaVaR = computeVaR(returns, confidence, holding);
        long end = System.currentTimeMillis();
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

    static void volSurface() {
        int N = 1_000_000;
        float[] S = new float[N];
        float[] K = new float[N];
        float[] T = new float[N];
        float[] r = new float[N];
        float[] P = new float[N];

        Random rand = new Random(42);
        for (int i = 0; i < N; i++) {
            S[i] = 90 + rand.nextFloat() * 20; // spot price: 90 ~ 110
            K[i] = 90 + rand.nextFloat() * 20; // strike price: 90 ~ 110
            T[i] = 0.1f + rand.nextFloat(); // maturity: 0.1 ~ 1.1 years
            r[i] = 0.01f; // interest rate
        }

        // 用固定sigma算出市场价格
        float trueSigma = 0.2f;
        for (int i = 0; i < N; i++) {
            P[i] = blackScholesCall(S[i], K[i], T[i], r[i], 0.2f) + (float) (Math.random() - 0.5) * 0.5f;
        }

        // CPU计算隐含波动率
        long start = System.currentTimeMillis();
        float[] cpuVol = new float[N];
        for (int i = 0; i < N; i++) {
            cpuVol[i] = impliedVol(S[i], K[i], T[i], r[i], P[i]);
        }
        long end = System.currentTimeMillis();
        System.out.println("CPU total time: " + (end - start) + " ms");

        // 打印前几个结果
        for (int i = 0; i < 10; i++) {
            System.out.printf("S=%.2f K=%.2f T=%.2f -> IV=%.4f (true=%.4f)\n",
                    S[i], K[i], T[i], cpuVol[i], trueSigma);
        }

        VolSurface volSurface = VolSurface.of(N, S, K, T, r, P);
        // === GPU
        start = System.currentTimeMillis();
        float[] gpuVol = GpuInterface.run(volSurface.getId(), volSurface.toArgs());
        end = System.currentTimeMillis();
        System.out.println("GPU total time: " + (end - start) + " ms");

        for (int i = 0; i < 10; i++) {
            System.out.printf("S=%.2f K=%.2f T=%.2f -> IV=%.4f (true=%.4f)\n",
                    S[i], K[i], T[i], gpuVol[i], trueSigma);
        }
    }

    static float europeanCallPrice(
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

    static float computeVaR(float[] returns, float confidence, float holding) {
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

    static float impliedVol(float S, float K, float T, float r, float marketPrice) {
        float sigma = 0.2f;
        for (int i = 0; i < 20; i++) {
            float d1 = (float) ((Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T)));
            float d2 = d1 - sigma * (float) Math.sqrt(T);
            float call = (float) (S * normCdf(d1) - K * Math.exp(-r * T) * normCdf(d2));
            float vega = (float) (S * Math.sqrt(T) * normPdf(d1));
            float diff = call - marketPrice;
            if (Math.abs(diff) < 1e-6)
                break;
            sigma -= diff / vega;
        }
        return sigma;
    }

    static float blackScholesCall(float S, float K, float T, float r, float sigma) {
        float d1 = (float) ((Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T)));
        float d2 = d1 - sigma * (float) Math.sqrt(T);
        return S * normCdf(d1) - K * (float) Math.exp(-r * T) * normCdf(d2);
    }

    static float normPdf(float x) {
        return (float) (1.0 / Math.sqrt(2 * Math.PI) * Math.exp(-0.5 * x * x));
    }

    static float normCdf(float x) {
        return (float) (0.5 * (1 + erf(x / Math.sqrt(2))));
    }

    // 近似误差函数
    static double erf(double x) {
        // Abramowitz & Stegun formula 7.1.26
        double t = 1.0 / (1.0 + 0.5 * Math.abs(x));
        double tau = t * Math.exp(-x * x
                - 1.26551223 + t * (1.00002368 + t * (0.37409196 +
                        t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 +
                                t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 +
                                        t * 0.17087277)))))))));
        return x >= 0 ? 1.0 - tau : tau - 1.0;
    }
}
