# Java 调用 Rust + CUDA 的示例


``` bash
cargo build
cp target/debug/libcuda_demo.so src/main/resources

mvn clean package
java -jar target/cuda_demo-1.0.0-SNAPSHOT.jar
```

## 添加computex

方法直接添加到cuda/computex.cu中，jni中get_function("xxx")即可