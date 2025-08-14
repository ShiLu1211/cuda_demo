package com.erayt.cuda;

import com.erayt.cuda.example.*;

public class Main {
    static {
        NativeLoader.sharedInstance().load();
    }

    public static void main(String[] args) {
        // Monte.monto();

        // Var.var();

        // VolSurface.volSurface();

        RandMatrix.randMatrix();
    }

}
