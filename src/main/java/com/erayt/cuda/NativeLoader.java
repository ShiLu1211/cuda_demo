package com.erayt.cuda;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;

public final class NativeLoader {

    private static final Logger LOGGER = LoggerFactory.getLogger(NativeLoader.class);
    private static final String TMP_DIR = System.getProperty("java.io.tmpdir");
    private static final String OS_NAME = System.getProperty("os.name");
    private static final String EXT = (OS_NAME.toLowerCase().contains("win")) ? ".dll"
            : (OS_NAME.toLowerCase().contains("mac os") ? ".dylib" : ".so");

    private static final String SHORT_NAME = "cuda_demo";
    private static final String LIB_PREFIX = (OS_NAME.toLowerCase().contains("win")) ? SHORT_NAME : "lib" + SHORT_NAME;
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

    // 先读环境变量，再读jar同目录，找不到再从jar里面加载
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
                    LOGGER.warn("环境变量中没有, {}", LIB_RESOURCE);
                }
            }

            // === 尝试加载 JAR 所在目录下的动态库 ===
            try {
                String jarDir = getJarDirectory();
                Path localLib = Paths.get(jarDir, LIB_RESOURCE);
                if (Files.exists(localLib)) {
                    if (LOGGER.isInfoEnabled()) {
                        LOGGER.info("加载 JAR 同目录下的 {}", LIB_RESOURCE);
                    }
                    System.load(localLib.toAbsolutePath().toString());
                    loaded = true;
                    return;
                } else {
                    if (LOGGER.isInfoEnabled()) {
                        LOGGER.info("JAR 同目录下未找到 {}", LIB_RESOURCE);
                    }
                }
            } catch (Exception e) {
                if (LOGGER.isWarnEnabled()) {
                    LOGGER.warn("尝试加载 JAR 同目录下的 {} 失败", LIB_RESOURCE, e);
                }
            }

            // === 最后回退到 JAR 内部资源释放 ===
            if (LOGGER.isInfoEnabled()) {
                LOGGER.info("加载 JAR 包中的 {}", LIB_RESOURCE);
            }
            try {
                String libPath = unleashJarLib();
                if (!libPath.isEmpty()) {
                    System.load(libPath);
                    loaded = true;
                }
            } catch (Exception e) {
                if (LOGGER.isWarnEnabled()) {
                    LOGGER.warn("加载 JAR 包中的 {} 失败", LIB_RESOURCE, e);
                }
            }
        }
    }

    private String getJarDirectory() {
        try {
            Path jarPath = Paths.get(NativeLoader.class
                    .getProtectionDomain()
                    .getCodeSource()
                    .getLocation()
                    .toURI());

            Path parent = jarPath.getParent();
            if (parent != null) {
                return parent.toAbsolutePath().toString();
            } else {
                if (LOGGER.isWarnEnabled()) {
                    LOGGER.warn("JAR 文件没有上级目录: {}", jarPath);
                }
                return "";
            }
        } catch (Exception e) {
            if (LOGGER.isWarnEnabled()) {
                LOGGER.warn("获取 JAR 所在目录失败", e);
            }
            return "";
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
