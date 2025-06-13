# Java 调用 Rust + CUDA 的蒙特卡洛期权定价示例


``` bash
cargo build
cp target/debug/libcuda_demo.so src/main/resources

mvn clean package
java -jar target/cuda_demo-1.0.0-SNAPSHOT.jar
```