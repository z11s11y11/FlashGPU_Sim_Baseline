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