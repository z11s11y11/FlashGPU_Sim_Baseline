## 运行这些测试文件的代码

# 可以直接使用该目录现成的 Makefile：

cd ~/FlashGPU_Sim_Baseline

make -C tests/src/microbench/memory \
    ARCH=sm_90a \
    PTX_PROFILE=compute_90a \
    all


# 编译完成后检查：
ls -lh tests/build/bin/microbench/memory/l2_latency_bench_cuda128_sm_90a

# 然后配置 H100 模拟环境：
source setup_environment

mkdir -p tests/run/l2_latency_h100

cp configs/SM90_H100/* tests/run/l2_latency_h100/

cp tests/build/bin/microbench/memory/l2_latency_bench_cuda128_sm_90a \
   tests/run/l2_latency_h100/

cd tests/run/l2_latency_h100

# 运行最小规模测试：
--warmup-steps=2048 可能太大了，要运行很久

./l2_latency_bench_cuda128_sm_90a \
    --case=l2-hit \
    --bytes=4096 \
    --steps=16 \
    --samples=8 \
    --warmup-steps=2 \
    --cache-op=cg >/home/zhangshuyi/FlashGPU_Sim_Baseline/tests/src/microbench/memory/reslut.log



## 在真机上测量

/home/zhangshuyi/FlashGPU_Sim_Baseline/tests/src/microbench/memory/l2_partition_latency_probe.cu

# 在真机的 Linux 终端执行：

```
cd /home/zhangshuyi/FlashGPU_Sim_Baseline

# 查看测试源码
sed -n '1,260p' tests/src/microbench/memory/l2_partition_latency_probe.cu

# 查看同目录的测试文件
ls tests/src/microbench/memory/

# 查看测试框架是否已经注册这个测试
grep -R -n "l2_partition_latency_probe" tests/
```

# 第二步：确认本机有可用的 NVIDIA GPU 和 CUDA 编译器

```
nvidia-smi
nvcc --version
```

# 第三步：编译这个测试程序

```
cd /home/zhangshuyi/FlashGPU_Sim_Baseline

mkdir -p tests/run/memory_probe

nvcc -O2 -std=c++17 \
  tests/src/microbench/memory/l2_partition_latency_probe.cu \
  -o tests/run/memory_probe/l2_partition_latency_probe
```

如果编译成功，执行文件就会生成在：tests/run/memory_probe/l2_partition_latency_probe

# 第四步：运行并保存测量结果

```
cd /home/zhangshuyi/FlashGPU_Sim_Baseline

./tests/run/memory_probe/l2_partition_latency_probe \
  --blocks=0 \
  --threads=32 \
  --repeats=256 \
  --warmup=64 \
  --bit-first=7 \
  --bit-last=27 \
  --dense-offsets=64 \
  --dense-stride=128 \
  --data-bytes=256M \
  --csv=tests/run/memory_probe/results.csv \
  2>&1 | tee tests/run/memory_probe/run.log
```