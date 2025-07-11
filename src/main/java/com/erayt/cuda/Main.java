package com.erayt.cuda;

import java.security.SecureRandom;
import java.util.Random;
import java.util.stream.IntStream;

import com.erayt.cuda.algo.MonteAlgo;
import com.erayt.cuda.algo.VarAlgo;
import com.erayt.cuda.algo.VolSurface;

public class Main {
    static {
        NativeLoader.sharedInstance().load();
    }
    private static final SecureRandom SECURE_RANDOM = new SecureRandom();
    private static final Random RAND = new Random();

    public static void main(String[] args) {
        GpuInterface.cudaInit();

        monto();

        var();

        volSurface();
    }

    static void monto() {
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

    static void var() {
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

    static void volSurface() {
        System.out.println("Vol Surface");
        int N = 1_000_000;
        int count = 5;
        double[] S = new double[N];
        double[] K = new double[N];
        double[] T = new double[N];
        double[] r = new double[N];
        double[] P = new double[N];

        for (int i = 0; i < N; i++) {
            S[i] = 90 + SECURE_RANDOM.nextDouble() * 20; // spot price: 90 ~ 110
            K[i] = 90 + SECURE_RANDOM.nextDouble() * 20; // strike price: 90 ~ 110
            T[i] = 0.1 + SECURE_RANDOM.nextDouble(); // maturity: 0.1 ~ 1.1 years
            r[i] = 0.01; // interest rate
        }

        // 用固定sigma算出市场价格
        double trueSigma = 0.2;
        for (int i = 0; i < N; i++) {
            P[i] = blackScholesCall(S[i], K[i], T[i], r[i], 0.2) + (Math.random() - 0.5) * 0.5;
        }

        // CPU计算隐含波动率
        long start = System.currentTimeMillis();
        double[] cpuVol = new double[N];
        for (int i = 0; i < N; i++) {
            cpuVol[i] = impliedVol(S[i], K[i], T[i], r[i], P[i]);
        }
        long end = System.currentTimeMillis();
        System.out.println("CPU total time: " + (end - start) + " ms");

        // 打印前几个结果
        for (int i = 0; i < count; i++) {
            System.out.printf("S=%.2f K=%.2f T=%.2f -> IV=%.4f (true=%.4f)%n",
                    S[i], K[i], T[i], cpuVol[i], trueSigma);
        }

        start = System.currentTimeMillis();
        IntStream.range(0, N).parallel().forEach(i -> {
            cpuVol[i] = impliedVol(S[i], K[i], T[i], r[i], P[i]);
        });
        end = System.currentTimeMillis();
        System.out.println("CPU Parallel total time: " + (end - start) + " ms");

        // 打印前几个结果
        for (int i = 0; i < count; i++) {
            System.out.printf("S=%.2f K=%.2f T=%.2f -> IV=%.4f (true=%.4f)%n",
                    S[i], K[i], T[i], cpuVol[i], trueSigma);
        }

        VolSurface volSurface = VolSurface.of(N, S, K, T, r, P);
        // === GPU
        start = System.currentTimeMillis();
        double[] gpuVol = GpuInterface.run(volSurface.getId(), volSurface.toArgs());
        end = System.currentTimeMillis();
        System.out.println("GPU total time: " + (end - start) + " ms");

        for (int i = 0; i < count; i++) {
            System.out.printf("S=%.2f K=%.2f T=%.2f -> IV=%.4f (true=%.4f)%n",
                    S[i], K[i], T[i], gpuVol[i], trueSigma);
        }

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

    static double impliedVol(double S, double K, double T, double r, double marketPrice) {
        double sigma = 0.2;
        for (int i = 0; i < 20; i++) {
            double d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
            double d2 = d1 - sigma * Math.sqrt(T);
            double call = S * normCdf(d1) - K * Math.exp(-r * T) * normCdf(d2);
            double vega = S * Math.sqrt(T) * normPdf(d1);
            double diff = call - marketPrice;
            if (Math.abs(diff) < 1e-6)
                break;
            sigma -= diff / vega;
        }
        return sigma;
    }

    static double blackScholesCall(double S, double K, double T, double r, double sigma) {
        double d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
        double d2 = d1 - sigma * Math.sqrt(T);
        return S * normCdf(d1) - K * Math.exp(-r * T) * normCdf(d2);
    }

    static double normPdf(double x) {
        return 1.0 / Math.sqrt(2 * Math.PI) * Math.exp(-0.5 * x * x);
    }

    static double normCdf(double x) {
        return 0.5 * (1 + erf(x / Math.sqrt(2)));
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
