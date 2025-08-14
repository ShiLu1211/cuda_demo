# Java 调用 Rust + CUDA 的示例


``` bash
cargo build -r
cp target/release/libcuda_demo.so src/main/resources

mvn package
java -jar target/cuda_demo-1.0.0-SNAPSHOT.jar
```

## 添加computex

cuda方法直接添加到cuda/中，java在com/erayt/cuda/algo/中实现接口GpuAlgorithm，定义参数格式

rust在src/cuda/wrappers.rs中实现cust调用，src/algo/mod.rs中添加dispatch调用