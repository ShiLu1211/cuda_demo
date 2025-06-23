# Java 调用 Rust + CUDA 的示例


``` bash
cargo build
cp target/debug/libcuda_demo.so src/main/resources

mvn package
java -jar target/cuda_demo-1.0.0-SNAPSHOT.jar
```

## 添加computex

cuda方法直接添加到cuda中，java在com/erayt/cuda/algo中实现接口GpuAlgorithm，定义参数格式

rust在src/cuda/wrappers.rs中实现jni函数，src/algo中添加dispatch调用