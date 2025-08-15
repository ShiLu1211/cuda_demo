package com.erayt.cuda.example;

import java.util.Random;

import com.erayt.cuda.GpuInterface;
import com.erayt.cuda.algo.BSAlgo;
import com.erayt.cuda.algo.RandAlgo;

public class RandMatrix {
    private static final Random RAND = new Random();

    public static void randMatrix() {
        System.out.println("rand matrix");

        int num_paths = 100_000;
        int num_steps = 252;
        int seed = 0;
        RandAlgo rand_matrix = RandAlgo.of(num_paths, num_steps, seed);

        double[] result = GpuInterface.runDouble(rand_matrix.getId(), rand_matrix.toArgs());

        double[][] matrix = new double[num_paths][num_steps];

        for (int i = 0; i < num_paths; i++) {
            System.arraycopy(result, i * num_steps, matrix[i], 0, num_steps);
        }

        // === CPU
        long start = System.currentTimeMillis();
        matrix = randMatrixJava(num_paths, num_steps, seed);
        long end = System.currentTimeMillis();
        System.out.println("CPU Time: " + (end - start) + " ms");
        System.out.println("Java Rand = " + matrix.length + " x " + matrix[0].length);

        System.out.println("====================");

        // === CPU rust
        start = System.currentTimeMillis();
        result = GpuInterface.runCpu(rand_matrix.getId(), rand_matrix.toArgs());

        matrix = new double[num_paths][num_steps];

        for (int i = 0; i < num_paths; i++) {
            System.arraycopy(result, i * num_steps, matrix[i], 0, num_steps);
        }
        end = System.currentTimeMillis();
        System.out.println("CPU Rust Time: " + (end - start) + " ms");
        System.out.println("GPU Rand = " + matrix.length + " x " + matrix[0].length);

        System.out.println("====================");

        // === GPU
        start = System.currentTimeMillis();
        result = GpuInterface.runDouble(rand_matrix.getId(), rand_matrix.toArgs());

        matrix = new double[num_paths][num_steps];

        for (int i = 0; i < num_paths; i++) {
            System.arraycopy(result, i * num_steps, matrix[i], 0, num_steps);
        }
        end = System.currentTimeMillis();
        System.out.println("GPU Time: " + (end - start) + " ms");
        System.out.println("GPU Rand = " + matrix.length + " x " + matrix[0].length);

        System.out.println("====================");

        double S0 = 100.0;
        double r = 0.05;
        double q = 0.0;
        double v = 0.2;
        double T = 1.0;

        start = System.currentTimeMillis();
        BSAlgo bs = BSAlgo.of(S0, r, q, v, T, matrix);
        result = GpuInterface.runDouble(bs.getId(), bs.toArgs());

        matrix = new double[num_paths][num_steps];

        for (int i = 0; i < num_paths; i++) {
            System.arraycopy(result, i * num_steps, matrix[i], 0, num_steps);
        }
        end = System.currentTimeMillis();
        System.out.println("GPU Time: " + (end - start) + " ms");
        System.out.println("GPU Simulate = " + matrix.length + " x " + matrix[0].length);

        // for (int i = 0; i < num_steps; i++) {
        // System.out.print(matrix[0][i] + " ");
        // }
        // System.out.println();

        System.out.println("====================");

        double[] args = bs.toArgs2();
        double[] newArgs = new double[args.length + 3];

        // 先拷贝原数据
        System.arraycopy(args, 0, newArgs, 0, args.length);

        // 追加三个值
        newArgs[args.length] = num_paths;
        newArgs[args.length + 1] = num_steps;
        newArgs[args.length + 2] = seed;

        start = System.currentTimeMillis();
        
        result = GpuInterface.runDouble(bs.getId2(), newArgs);

        matrix = new double[num_paths][num_steps];

        for (int i = 0; i < num_paths; i++) {
            System.arraycopy(result, i * num_steps, matrix[i], 0, num_steps);
        }
        end = System.currentTimeMillis();
        System.out.println("GPU Time: " + (end - start) + " ms");
        System.out.println("GPU Rand and Simulate = " + matrix.length + " x " + matrix[0].length);

        System.out.println("====================");

        start = System.currentTimeMillis();
        
        result = GpuInterface.runCpu(bs.getId2(), newArgs);

        matrix = new double[num_paths][num_steps];

        for (int i = 0; i < num_paths; i++) {
            System.arraycopy(result, i * num_steps, matrix[i], 0, num_steps);
        }
        end = System.currentTimeMillis();
        System.out.println("CPU Rust Time: " + (end - start) + " ms");
        System.out.println("CPU Rand and Simulate = " + matrix.length + " x " + matrix[0].length);
    }

    static double[][] randMatrixJava(int num_paths, int num_steps, int seed) {
        double[][] matrix = new double[num_steps][num_paths];

        for (int i = 0; i < num_steps; i++) {
            for (int j = 0; j < num_paths; j++) {
                matrix[i][j] = RAND.nextGaussian();
            }
        }
        return matrix;
    }
}
