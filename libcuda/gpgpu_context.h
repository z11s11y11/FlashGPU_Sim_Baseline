#ifndef __gpgpu_context_h__
#define __gpgpu_context_h__
#include <atomic>
#include <string>
#include "../src/cuda-sim/cuda-sim.h"
#include "../src/cuda-sim/cuda_device_runtime.h"
#include "../src/cuda-sim/ptx-stats.h"
#include "../src/cuda-sim/ptx_loader.h"
#include "../src/cuda-sim/ptx_parser.h"
#include "../src/gpgpusim_entrypoint.h"
#include "cuda_api_object.h"

class gpgpu_context {
 public:
  gpgpu_context() {
    // ===== 第一组：UID 计数器（给各类对象发唯一身份证号，用于调试/追踪）=====
    // 全部从 0 或 1 起，每次 new 对应对象时取当前值再自增；
    // 注意混用（有的从 0 有的从 1）是历史遗留，语义上都是"下一个可用 id"
    g_global_allfiles_symbol_table = NULL;  // 全局符号表：所有已加载 PTX 文件的符号汇总，
                                            // 惰性创建（首次注册符号时才 new），
                                            // 供跨文件函数/变量解析
    sm_next_access_uid = 0;                 // memory access（访存操作）的 uid 计数器
    warp_inst_sm_next_uid = 0;              // warp 级指令（warp_inst_t）的 uid 计数器
    operand_info_sm_next_uid = 1;           // 操作数（operand_info）的 uid 计数器
    kernel_info_m_next_uid = 1;             // kernel（kernel_info_t）的 uid 计数器
    g_num_ptx_inst_uid = 0;                 // PTX 指令的 uid 计数器（指令编号）
    g_ptx_cta_info_uid = 1;                 // CTA 信息（ptx_cta_info）的 uid 计数器
    symbol_sm_next_uid = 1;                 // 符号（symbol）的 uid 计数器
    function_info_sm_next_uid = 1;          // 函数信息（function_info）的 uid 计数器

    // ===== 第二组：PTX 处理流程的功能开关（默认全关，由环境变量/选项打开）=====
    debug_tensorcore = 0;                        // Tensor Core 调试级别（0 = 不打印）
    ptx_register_allocator_enabled = false;      // PTX 级寄存器分配：对 PTX 虚拟寄存器
                                                 // 做二次分配（研究用，默认关）
    ptx_register_allocator_stats = false;        // 打印寄存器分配的统计信息
    ptx_reorder_enabled = false;                 // PTX 指令重排：加载时重排指令序列
                                                 // 以改善流水线（调度优化，默认关）
    ptx_reorder_sass_guided = false;             // 用编译器输出的 SASS（真实机器码顺序）
                                                 // 来指导 PTX 重排
    ptx_reorder_sass_extract_attempted = false;  // 是否已尝试过提取 SASS（防止重复尝试）
    ptx_reorder_sass_extract_ok = false;         // SASS 提取是否成功（失败则退回默认顺序）

    // ===== 第三组：子模块装配（把 this 传给每个子模块，形成"总线-分机"结构）=====
    api = new cuda_runtime_api(this);      // CUDA Runtime API 层：cudaLaunchKernel /
                                           // cudaMemcpy 等的模拟实现（应用代码直接打交道的层）
    ptxinfo = new ptxinfo_data(this);      // ptxas 元数据：解析真实编译器输出的
                                           // 寄存器数、shared memory 用量等信息
    ptx_parser = new ptx_recognizer(this); // PTX 语法解析器（flex/bison 生成的识别器），
                                           // cubin/PTX 加载时把文本转成指令流
    the_gpgpusim = new GPGPUsim_ctx(this); // ★ 入口层上下文：持有 g_the_gpu、g_the_gpu_config、
                                           //   g_stream_manager、模拟线程和信号量——
                                           //   之前看过的 GPGPUsim_ctx 类
    func_sim = new cuda_sim(this);         // 功能仿真器：逐条解释执行 PTX（纯功能、无时钟），
                                           //   kernel 首次运行的语义执行靠它
    device_runtime = new cuda_device_runtime(this);  // 设备端 runtime：CDP（动态并行）支持，
                                           //   处理 device 端 launch 等运行时服务
    stats = new ptx_stats(this);           // PTX 层统计：指令 mix、执行计数等
  }
  // global list
  symbol_table *g_global_allfiles_symbol_table;
  const char *g_filename;
  std::atomic<unsigned> sm_next_access_uid;
  std::atomic<unsigned> warp_inst_sm_next_uid;
  unsigned operand_info_sm_next_uid;  // uid for operand_info
  unsigned kernel_info_m_next_uid;    // uid for kernel_info_t
  unsigned g_num_ptx_inst_uid;        // uid for ptx inst inside ptx_instruction
  unsigned long long g_ptx_cta_info_uid;
  unsigned symbol_sm_next_uid;  // uid for symbol
  unsigned function_info_sm_next_uid;
  std::vector<ptx_instruction *>
      s_g_pc_to_insn;  // a direct mapping from PC to instruction
  bool debug_tensorcore;
  bool ptx_register_allocator_enabled;
  bool ptx_register_allocator_stats;
  bool ptx_reorder_enabled;
  bool ptx_reorder_sass_guided;
  bool ptx_reorder_sass_extract_attempted;
  bool ptx_reorder_sass_extract_ok;
  std::string ptx_reorder_sass_ptxline_file;
  std::string ptx_reorder_sass_ptxline_arch;
  std::string ptx_reorder_sass_ptxline_binary;

  // SST related
  bool requested_synchronize = false;

  // objects pointers for each file
  cuda_runtime_api *api;
  ptxinfo_data *ptxinfo;
  ptx_recognizer *ptx_parser;
  GPGPUsim_ctx *the_gpgpusim;
  cuda_sim *func_sim;
  cuda_device_runtime *device_runtime;
  ptx_stats *stats;
  // member function list
  void synchronize();
  bool synchronize_check();
  void exit_simulation();
  void print_simulation_time();
  int gpgpu_opencl_ptx_sim_main_perf(kernel_info_t *grid);
  void cuobjdumpParseBinary(unsigned int handle);
  class symbol_table *gpgpu_ptx_sim_load_ptx_from_string(const char *p,
                                                         unsigned source_num);
  class symbol_table *gpgpu_ptx_sim_load_ptx_from_filename(
      const char *filename);
  void gpgpu_ptx_info_load_from_filename(const char *filename,
                                         const char *arch_str);
  void gpgpu_ptxinfo_load_from_string(const char *p_for_info,
                                      unsigned source_num,
                                      unsigned sm_version = 20,
                                      int no_of_ptx = 0);
  void print_ptx_file(const char *p, unsigned source_num, const char *filename);
  class symbol_table *init_parser(const char *);
  class gpgpu_sim *gpgpu_ptx_sim_init_perf();
  void start_sim_thread(int api);
  struct _cuda_device_id *GPGPUSim_Init();
  void ptx_reg_options(option_parser_t opp);
  const ptx_instruction *pc_to_instruction(unsigned pc);
  const warp_inst_t *ptx_fetch_inst(address_type pc);
  unsigned translate_pc_to_ptxlineno(unsigned pc);
};
gpgpu_context *GPGPU_Context();

#endif /* __gpgpu_context_h__ */
