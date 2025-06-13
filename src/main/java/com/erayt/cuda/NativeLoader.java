package com.erayt.cuda;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class NativeLoader {
    private static final Logger LOGGER = LoggerFactory.getLogger(NativeLoader.class);
    private static final String TMP_DIR = System.getProperty("java.io.tmpdir");
    private static final String OS_NAME = System.getProperty("os.name");
    private static final String EXT = (OS_NAME.toLowerCase().contains("win")) ? ".dll"
            : (OS_NAME.toLowerCase().contains("mac os") ? ".dylib" : ".so");

    private static final String SHORT_NAME = "cuda_demo";
    private static final String LIB_PREFIX = "lib" + SHORT_NAME;
    private static final String LIB_RESOURCE = LIB_PREFIX + EXT;
    private volatile boolean loaded = false;

    public static NativeLoader sharedInstance() {
        return NativeLoaderInner.Instance;
    }

    private static class NativeLoaderInner {
        final static NativeLoader Instance = new NativeLoader();
    }

    private NativeLoader() {
    }

    // 读环境变量
    public void loadShort() {
        if (loaded) {
            return;
        }
        synchronized (NativeLoader.class) {
            if (loaded) {
                return;
            }
            try {
                System.loadLibrary(SHORT_NAME);
                loaded = true;
            } catch (UnsatisfiedLinkError e) {
                if (LOGGER.isWarnEnabled()) {
                    LOGGER.warn(e.getMessage());
                }
            }
        }
    }

    // 先读环境变量，找不到再从jar里面加载
    public void load() {
        if (loaded) {
            return;
        }
        synchronized (NativeLoader.class) {
            if (loaded) {
                return;
            }
            try {
                System.loadLibrary(SHORT_NAME);
                loaded = true;
                return;
            } catch (UnsatisfiedLinkError error) {
                if (LOGGER.isWarnEnabled()) {
                    LOGGER.warn("jar同目录下没有, {}", LIB_RESOURCE);
                }
            }
            if (LOGGER.isInfoEnabled()) {
                LOGGER.info("加载jar里面的{}", LIB_RESOURCE);
            }
            try {
                String libPath = unleashJarLib();
                if (!libPath.isEmpty()) {
                    // print();("libPath: " + libPath);
                    System.load(libPath);
                    loaded = true;
                }
            } catch (Exception e) {
                if (LOGGER.isWarnEnabled()) {
                    LOGGER.warn("加载jar里面的{}失败", LIB_RESOURCE, e);
                }
            }
        }
    }

    private String unleashJarLib() {
        try (InputStream is = Thread.currentThread()
                .getContextClassLoader().getResourceAsStream(LIB_RESOURCE)) {
            if (is == null) {
                if (LOGGER.isWarnEnabled()) {
                    LOGGER.warn("{} is not found in classpath", LIB_RESOURCE);
                }
                return "";
            }
            Path tmpDir = Paths.get(TMP_DIR);
            Path tmpLib = Files.createTempFile(tmpDir, LIB_PREFIX, EXT);
            tmpLib.toFile().deleteOnExit();
            Files.copy(is, tmpLib, StandardCopyOption.REPLACE_EXISTING);
            return tmpLib.toAbsolutePath().toString();
        } catch (UnsatisfiedLinkError | IOException e) {
            if (LOGGER.isWarnEnabled()) {
                LOGGER.warn(e.getMessage(), e);
            }
        }
        return "";
    }
}
