#pragma once

#ifdef USE_CUDA

#include <torchaudio/csrc/rnnt/workspace.h>
#include <torchaudio/csrc/rnnt/gpu/gpu_kernel_utils.cuh>
#include <torchaudio/csrc/rnnt/gpu/gpu_kernels.cuh>
#include <cmath>
#include <random>

#include <mutex>
//#define MY_VERBOSE

namespace torchaudio {
namespace rnnt {
namespace gpu {

static int incount = 0;

#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(
    cudaError_t code,
    const char* file,
    int line,
    bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(
        stderr,
        "\nGPUassert: %s %s %d\n",
        cudaGetErrorString(code),
        file,
        line);
    if (abort)
      exit(code);
  }
}

template <typename DTYPE, typename CAST_DTYPE>
status_t LogSumExp2D(
    cudaStream_t stream,
    int N,
    int D,
    const DTYPE* logits, // [N, D]
    CAST_DTYPE* outputs) {
  { // compute max among D.
    dim3 block_dims(N);
    dim3 thread_dims(REDUCE_THREADS);

    ReduceMax2D<REDUCE_THREADS, DTYPE, CAST_DTYPE>
        <<<block_dims, thread_dims, 0, stream>>>(
            /*dim=*/D,
            /*inputs=*/logits,
            /*outputs=*/outputs);

    // BUGBUG: These error codes are only accurate when launching with
    // blocking. Otherwise they usually reflect earlier errors.
    if (cudaGetLastError() != cudaSuccess) {
      return COMPUTE_DENOMINATOR_REDUCE_MAX_FAILED;
    }
  }

  { // compute log(sum(exp(d_i - max)))
    dim3 block_dims(N);
    dim3 thread_dims(REDUCE_THREADS);

    ReduceLogSumExpGivenMax2D<REDUCE_THREADS, DTYPE, CAST_DTYPE>
        <<<block_dims, thread_dims, 0, stream>>>(
            /*dim=*/D,
            /*inputs=*/logits,
            /*outputs=*/outputs);

    if (cudaGetLastError() != cudaSuccess) {
      return COMPUTE_DENOMINATOR_REDUCE_SUM_FAILED;
    }
  }

  return SUCCESS;
}

#define MAX_LOSS_MASK_NUM (256)
template <typename DTYPE, typename CAST_DTYPE>
status_t DrawLossRegMap_CPU(
    int B,
    int maxSrcLen,
    int maxTgtLen,
    const int* srcLengths,
    const int* tgtLengths,
    const DTYPE weight,
    const DTYPE ratio,
    const DTYPE lower,
    const DTYPE upper,
    CAST_DTYPE* outputs) {
  { // compute max among D.
    std::mutex my_mtx;
    std::lock_guard<std::mutex> lock(my_mtx); 
    const int& maxT = maxSrcLen;
    const int& maxU = maxTgtLen;
    Indexer3D indexer(maxT, maxU);

    std::random_device rd;
    std::mt19937 gen(rd());

    DTYPE* cpu_outputs = nullptr;
    //std::cout << "GARBAGE111123123122" << std::endl;
    cpu_outputs = new DTYPE[B * maxT * maxU];
#ifdef MY_VERBOSE
    std::cout << "cpu_outputs: " << cpu_outputs << std::endl;
    std::cout << "Batch Size: " << B << std::endl;
    std::cout << "maxT: " << maxT << std::endl;
    std::cout << "maxU: " << maxU << std::endl;
    int cpu_outputs_size = B * maxT * maxU;
    std::cout << "cpu_outputs Size: " << B * maxT * maxU << std::endl;
#endif
    if(cpu_outputs == nullptr){
      return FAILURE;
    }

    int* cpu_srcLengths = new int[B];
    int* cpu_tgtLengths = new int[B];
    cudaMemcpy(cpu_srcLengths, srcLengths, sizeof(int)*B, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_tgtLengths, tgtLengths, sizeof(int)*B, cudaMemcpyDeviceToHost);

    //std::cout << "GARBAGE1112" << std::endl;
    for(int b=0; b<B; b++){
      const int T = cpu_srcLengths[b];
      const int U = cpu_tgtLengths[b] + 1;
#ifdef MY_VERBOSE
      std::cout << "T: " << T << std::endl;
      std::cout << "U: " << U << std::endl;
#endif
      //make random range
      //for time
      int tm[MAX_LOSS_MASK_NUM][2];
      int tm_len = 0;
      int tm_count = ratio * T;
      if( tm_count == 0 ){ tm_count = 1; }
      int count = 0;
      int value = 0;
      for(int i=0; i<MAX_LOSS_MASK_NUM; i++){
        std::uniform_int_distribution<> dis_start(value, T);
        value = dis_start(gen);
        if( value >= T ){
          break;
        }
        std::uniform_int_distribution<> dis_len(1, tm_count);
        int width = dis_len(gen);
        if( tm_count < count + width ){
          width = tm_count - count;
        }
        if( T < value + width ){
          width = T - value;
        }
        
        tm[i][0] = value;
        tm[i][1] = width;

        count += width;
        value += width;
        tm_len++;

        if( tm_count <= count ){
          break;
        }
        if( tm_len >= MAX_LOSS_MASK_NUM ){
          break;
        }
      }

      //for label
      int lm[MAX_LOSS_MASK_NUM][2];
      int lm_len = 0;
      int lm_count = ratio * U;
      if( lm_count == 0){ lm_count = 1; }
      count = 0;
      value = 0;
      for(int i=0; i<MAX_LOSS_MASK_NUM; i++){
        std::uniform_int_distribution<> dis_start(value, U);
        value = dis_start(gen);
        if( value >= U ){
          break;
        }
        std::uniform_int_distribution<> dis_len(1, lm_count);
        int width = dis_len(gen);
        if( lm_count < count + width ){
          width = lm_count - count;
        }
        if( U < value + width ){
          width = U - value;
        }

        lm[i][0] = value;
        lm[i][1] = width;

        count += width;
        value += width;
        lm_len++;

        if( lm_count <= count ){
          break;
        }

        if( lm_len >= MAX_LOSS_MASK_NUM ){
          break;
        }
      }
#ifdef MY_VERBOSE
      std::cout << "b: " << b << std::endl;
      std::cout << "tm_len: " << tm_len << std::endl;
      for(int i=0; i<tm_len; i++){
        std::cout << "tm " << i << ": " << tm[i][0] << ", " << tm[i][1] << std::endl;
      }
      std::cout << "lm_len: " << lm_len << std::endl;
      for(int i=0; i<lm_len; i++){
        std::cout << "lm " << i << ": " << lm[i][0] << ", " << lm[i][1] << std::endl;
      }
#endif

      std::uniform_real_distribution<> dis_real(lower, upper);
      for(int t=0; t<T; t++){
        for(int u=0; u<U; u++){
          int idx = indexer(b, t, u);
#ifdef MY_VERBOSE
          if( cpu_outputs_size <= idx){
            std::cout << "idx: " << idx << std::endl;
          }
#endif
          cpu_outputs[idx] = (DTYPE)0;

          //for time
          for(int k=0; k<tm_len; k++){
            if(t>=tm[k][0] && t<(tm[k][0] + tm[k][1])){
              cpu_outputs[idx] = std::log( 1 + weight * (DTYPE)dis_real(gen));
            }
          }

          //for label
          for(int k=0; k<lm_len; k++){
            if(u>=lm[k][0] && u<(lm[k][0] + lm[k][1])){
              cpu_outputs[idx] = std::log( 1 + weight * (DTYPE)dis_real(gen));
            }
          }
        }
      }
    }
    
    cudaMemcpy(outputs, cpu_outputs, sizeof(DTYPE)*B*maxT*maxU, cudaMemcpyHostToDevice);
    if( cpu_outputs != nullptr ){ delete cpu_outputs; }
    if( cpu_srcLengths != nullptr ){ delete cpu_srcLengths; }
    if( cpu_tgtLengths != nullptr ){ delete cpu_tgtLengths; }
  }

  return SUCCESS;
}

// CPU version computMap

// Inputs:
//   workspace: workspace.
//   logits: pointer to (B, max_T, max_U, D) logits.
//   targets: pointer to (B, max_U - 1) targets in the batch.
//   srcLengths: pointer to (B, ) source lengths in the batch.
//   tgtLengths: pointer to (B, ) target lengths in the batch.
//
// Outputs:
//   costs: pointer to (B, ) costs in the batch.
//   gradients: pointer to (B, max_T, max_U, D) gradients in the batch.
template <typename DTYPE, typename CAST_DTYPE>
status_t Compute(
    const Workspace<CAST_DTYPE>& workspace,
    const DTYPE* logits,
    const int* targets,
    const int* srcLengths,
    const int* tgtLengths,
    DTYPE* costs,
    DTYPE* gradients = nullptr) {
  const Options& options = workspace.GetOptions();

  const cudaStream_t& stream = options.stream_;
  const int& B = options.batchSize_;
  const int& H = options.nHypos_;
  const int& max_T = options.maxSrcLen_;
  const int& max_U = options.maxTgtLen_;
  const int& D = options.numTargets_;
  const int& blank = options.blank_;
  const DTYPE clamp = options.clamp_;
  const bool fast_emit = options.fastEmit_;
  const DTYPE fast_emit_weight = options.fastEmitWeight_;
  const bool loss_regularization = options.lossRegularization_;
  const DTYPE loss_regularization_weight = options.lossRegularizationWeight_;
  const DTYPE loss_regularization_sigma = options.lossRegularizationSigma_;
  const bool loss_regularization_swing = options.lossRegularizationSwing_;

  { // compute denominators.
    status_t status = LogSumExp2D<DTYPE, CAST_DTYPE>(
        /*stream=*/stream,
        /*N=*/B * H * max_T * max_U,
        /*D=*/D,
        /*logits=*/logits,
        /*denominators=*/workspace.GetPointerToDenominators());

    if (status != SUCCESS) {
      return status;
    }
  }
  {
    //bool loss_regularizationt = true;
    if( loss_regularization ){
#ifdef MY_VERBOSE
      std::cout << "GARBAGE DrawLossRegMap_CPU starts" << std::endl;
#endif
      status_t status = DrawLossRegMap_CPU<DTYPE, CAST_DTYPE>(
                                          B,
                                          max_T,
                                          max_U,
                                          srcLengths,
                                          tgtLengths,
                                          /*weight*/options.lossRegularizationWeight_,
                                          /*ratio*/0.07,
                                          /*lower*/-1,
                                          /*upper*/0,
                                          workspace.GetPointerToLossRegularization());
#ifdef MY_VERBOSE
      std::cout << "GARBAGE DrawLossRegMap_CPU ends" << std::endl;
#endif
      if (status != SUCCESS){
        return status;
      }
    }
  }

  {
    //if (loss_regularization){
    if ( false ){
      int num_segments =
          (max_T + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
      dim3 block_dims(num_segments, max_U, B * H);
      dim3 thread_dims(MAX_THREADS_PER_BLOCK);

      DTYPE slope = max_T / max_U;
      DTYPE sigma = options.lossRegularizationSigma_;
      DTYPE weight = options.lossRegularizationWeight_;
      DTYPE denom = 1.0 / sqrt( 2 * M_PI * sigma * sigma );

      int *cpu_random_movings;
      int *random_movings;
      //std::cout << "GARBAGE" << std::endl;
      //std::cout << loss_regularization_swing << std::endl;
      if( loss_regularization_swing == true ){
        //std::cout << "GARBAGE TRUE" << std::endl;
        cpu_random_movings = new int[B];
        std::random_device rd;
        std::mt19937 gen(rd());

        std::uniform_int_distribution<> dis(-sigma, sigma);
        for(int i=0; i<B; ++i){
          cpu_random_movings[i] = dis(gen);
        }
        cudaMalloc(&random_movings, B * sizeof(int));
        cudaMemcpy(random_movings, cpu_random_movings, B * sizeof(int), cudaMemcpyHostToDevice);
      }
      else{
        //std::cout << "GARBAGE FALSE" << std::endl;
        cpu_random_movings = new int[B];
        for(int i=0; i<B; ++i){
          cpu_random_movings[i] = 0;
        }
        cudaMalloc(&random_movings, B * sizeof(int));
        cudaMemcpy(random_movings, cpu_random_movings, B * sizeof(int), cudaMemcpyHostToDevice);
      }

      //std::cout << "GARBAGE IS WORKING" << std::endl;
      //ComputeMap<DTYPE, CAST_DTYPE><<<block_dims, thread_dims, 0, stream>>>(
      ComputeGaussianMapU<DTYPE, CAST_DTYPE><<<block_dims, thread_dims, 0, stream>>>(
        max_T,
        max_U,
        srcLengths,
        tgtLengths,
        H, //slope, //sigma, denom,
        weight,
        sigma,
        workspace.GetPointerToLossRegularization(),
        random_movings
      );
    }

  }

  { // compute log probability pairs (blank and target).
    int num_segments =
        (max_T + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
    dim3 block_dims(num_segments, max_U, B * H);
    dim3 thread_dims(MAX_THREADS_PER_BLOCK);

    ComputeLogProbs<DTYPE, CAST_DTYPE><<<block_dims, thread_dims, 0, stream>>>(
        /*max_src_len=*/max_T,
        /*max_tgt_len=*/max_U,
        /*num_targets=*/D,
        /*blank=*/blank,
        /*logits=*/logits,
        /*targets=*/targets,
        /*srcLengths=*/srcLengths,
        /*tgtLengths=*/tgtLengths,
        /*denominators=*/workspace.GetPointerToDenominators(),
        /*log_probs=*/workspace.GetPointerToLogProbs(),
        H);

    if (cudaGetLastError() != cudaSuccess) {
      return FAILURE;
    }
  }

  { // compute alphas, betas and costs.
    // warp is usually a group of threads (32)
    int num_warps = (max_T + WARP_SIZE - 1) / WARP_SIZE;

    // each block is identified by 3 d tuple.
    // we are using num_warp * max_U * B * H blocks
    // where num_warp is division among Time axis
    dim3 block_dims(num_warps, max_U, B * H);

    // each thread is identified by a 2 d tuple
    // 2nd dim is 2. 1 for alpha, 1 for beta
    dim3 thread_dims(WARP_SIZE, 2);
#ifdef MY_VERBOSE
      std::cout << "GARBAGE ComputeAlpahsBetasCosts starts" << std::endl;
#endif
    ComputeAlphasBetasCosts<DTYPE, CAST_DTYPE>
        <<<block_dims, thread_dims, 0, stream>>>(
            /*max_src_len=*/max_T,
            /*max_tgt_len=*/max_U,
            /*num_targets=*/D,
            /*blank=*/blank,
            /*log_probs=*/workspace.GetPointerToLogProbs(),
            /*srcLengths=*/srcLengths,
            /*tgtLengths=*/tgtLengths,
            /*alpha_counters=*/workspace.GetPointerToAlphaCounters(),
            /*alphas=*/workspace.GetPointerToAlphas(),
            /*beta_counters=*/workspace.GetPointerToBetaCounters(),
            /*betas=*/workspace.GetPointerToBetas(),
            /*costs=*/costs,
            /*warp_size=*/WARP_SIZE,
            /*num_warps=*/num_warps,
            H,
            fast_emit,
            fast_emit_weight,
            loss_regularization,
            loss_regularization_weight,
            workspace.GetPointerToLossRegularization());
#ifdef MY_VERBOSE
      std::cout << "GARBAGE ComputeAlpahsBetasCosts ends" << std::endl;
#endif
    if (cudaGetLastError() != cudaSuccess) {
      return COMPUTE_ALPHAS_BETAS_COSTS_FAILED;
    }
  }

  if (gradients != nullptr) { // compute gradients.
    // don't set gradients to zero to here as gradients might reuse memory from
    // logits
//#define MY_DEBUG
#ifdef MY_DEBUG
      int mSize = options.BTU();
      DTYPE* garbage = new DTYPE[mSize];
      cudaMemcpy(garbage, workspace.GetPointerToLossRegularization(), mSize * sizeof(DTYPE), cudaMemcpyDeviceToHost);
      DTYPE* mAlphas = new DTYPE[mSize];
      int* tLen = new int[B];
      int* uLen = new int[B];
      float* mCosts = new float[B];
      cudaMemcpy(tLen, srcLengths, B * sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(uLen, tgtLengths, B * sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(mCosts, costs, B * sizeof(float), cudaMemcpyDeviceToHost);
      Indexer3D idxr(max_T, max_U);

      int bSize = options.BTU();
      DTYPE* betas = new DTYPE[bSize];
      cudaMemcpy(betas, workspace.GetPointerToBetas(), bSize * sizeof(DTYPE), cudaMemcpyDeviceToHost);
      cudaMemcpy(mAlphas, workspace.GetPointerToAlphas(), bSize * sizeof(DTYPE), cudaMemcpyDeviceToHost);

      for(int b=0; b<B; b++){
        int costIdx = b * max_T * max_U;
        std::cout << tLen[b] << ", " << uLen[b] << ", "
                  << garbage[idxr(b, 0, 0)] << ", "
                  << mCosts[b] << ", "
                  << betas[idxr(b, 0, 0)]
                  << " MY_DEBUG2" << std::endl;
      }
      int myt=0;
      int myu=0;
      std::cout << "loss regularization weight: " << loss_regularization_weight << std::endl;
      std::cout << "loss regularization sigma: " << loss_regularization_sigma << std::endl;
      float mydenom = 1.0 / sqrt( 2 * M_PI * pow(loss_regularization_sigma, 2 ));
      float myslope = max_T / max_U;
      float result = log(1 + loss_regularization_weight * exp(-(pow(myt-myslope * myu, 2))/(2*pow(loss_regularization_sigma, 2)))*mydenom);
      std::cout << "loss regularization map " <<myt << ", " << myu <<": " <<result <<std::endl;
      std::cout << options << std::endl;

      std::cout << "GARBAGE MAP" << std::endl;
      for(int b=0; b<1; b++){
        for(int u=0; u<uLen[b]; u++){
          for(int t=0; t<tLen[b]; t++){
            std::cout << garbage[idxr(b, t, u)] << ", ";
          }
          std::cout << std::endl;
        }
        std::cout << std::endl;
        std::cout << std::endl;
      }

      std::cout << "GARBAGE ALPHAS" << std::endl;
      for(int b=0; b<1; b++){
        for(int u=0; u<uLen[b]; u++){
          for(int t=0; t<tLen[b]; t++){
            std::cout << mAlphas[idxr(b, t, u)] << ", ";
          }
          std::cout << std::endl;
        }
        std::cout << std::endl;
        std::cout << std::endl;
      }

      std::cout << "GARBAGE BETAS" << std::endl;
      for(int b=0; b<1; b++){
        for(int u=0; u<uLen[b]; u++){
          for(int t=0; t<tLen[b]; t++){
            std::cout << betas[idxr(b, t, u)] << ", ";
          }
          std::cout << std::endl;
        }
        std::cout << std::endl;
        std::cout << std::endl;
      }

#endif
    int num_blocks =
        (max_T + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
    dim3 block_dims(num_blocks, max_U, B * H);
    dim3 thread_dims(MAX_THREADS_PER_BLOCK);
#ifdef MY_VERBOSE
      std::cout << "GARBAGE ComputeGradients starts" << std::endl;
#endif
    ComputeGradients<DTYPE, CAST_DTYPE><<<block_dims, thread_dims, 0, stream>>>(
        /*max_src_len=*/max_T,
        /*max_tgt_len=*/max_U,
        /*num_targets=*/D,
        /*blank=*/blank,
        /*clamp=*/clamp,
        /*logits=*/logits,
        /*targets=*/targets,
        /*srcLengths=*/srcLengths,
        /*tgtLengths=*/tgtLengths,
        /*denominators=*/workspace.GetPointerToDenominators(),
        /*alphas=*/workspace.GetPointerToAlphas(),
        /*betas=*/workspace.GetPointerToBetas(),
        /*gradients=*/gradients,
        H,
        fast_emit,
        fast_emit_weight,
        false,//loss_regularization,
        loss_regularization_weight,
        workspace.GetPointerToLossRegularization());
#ifdef MY_VERBOSE
      std::cout << "GARBAGE ComputeGradients ends" << std::endl;
#endif
    if (cudaGetLastError() != cudaSuccess) {
      return COMPUTE_GRADIENTS_FAILED;
    }
  }
/*
  {
    if (loss_regularization){
      //delete [] cpu_random_movings;
      //cudaFree(random_movings);
    }
  }
*/
#ifdef MY_DEBUG
    if(incount == 2){
      exit(3);
    }
    else{
      std::cout << std::endl;
      incount++;
    }
#endif

  return SUCCESS;
}

template <typename DTYPE, typename CAST_DTYPE>
status_t ComputeAlphas(
    const Workspace<CAST_DTYPE>& workspace,
    const DTYPE* logits,
    const int* targets,
    const int* srcLengths,
    const int* tgtLengths,
    DTYPE* alphas) {
  const Options& options = workspace.GetOptions();

  const cudaStream_t& stream = options.stream_;
  const int& B = options.batchSize_;
  const int& H = options.nHypos_;
  const int& max_T = options.maxSrcLen_;
  const int& max_U = options.maxTgtLen_;
  const int& D = options.numTargets_;
  const int& blank = options.blank_;

  { // compute denominators.
    status_t status = LogSumExp2D<DTYPE, CAST_DTYPE>(
        /*stream=*/stream,
        /*N=*/B * H * max_T * max_U,
        /*D=*/D,
        /*logits=*/logits,
        /*denominators=*/workspace.GetPointerToDenominators());

    if (status != SUCCESS) {
      return status;
    }
  }

  { // compute log probability pairs (blank and target).
    int num_segments =
        (max_T + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
    dim3 block_dims(num_segments, max_U, B * H);
    dim3 thread_dims(MAX_THREADS_PER_BLOCK);

    ComputeLogProbs<DTYPE, CAST_DTYPE><<<block_dims, thread_dims, 0, stream>>>(
        /*max_src_len=*/max_T,
        /*max_tgt_len=*/max_U,
        /*num_targets=*/D,
        /*blank=*/blank,
        /*logits=*/logits,
        /*targets=*/targets,
        /*srcLengths=*/srcLengths,
        /*tgtLengths=*/tgtLengths,
        /*denominators=*/workspace.GetPointerToDenominators(),
        /*log_probs=*/workspace.GetPointerToLogProbs(),
        H);

    if (cudaGetLastError() != cudaSuccess) {
      return COMPUTE_LOG_PROBS_FAILED;
    }
  }
  { // compute alphas
    // warp is usually a group of threads (32)
    int num_warps = (max_T + WARP_SIZE - 1) / WARP_SIZE;

    // each block is identified by 3 d tuple.
    // we are using num_warp * max_U * B blocks
    // where num_warp is division among Time axis
    dim3 block_dims(num_warps, max_U, B * H);

    // each thread is identified by a 2 d tuple
    // 2nd dim is 1 for alpha only
    dim3 thread_dims(WARP_SIZE, 1);

    ComputeAlphasWrapper<DTYPE, CAST_DTYPE>
        <<<block_dims, thread_dims, 0, stream>>>(
            /*max_src_len=*/max_T,
            /*max_tgt_len=*/max_U,
            /*num_targets=*/D,
            /*blank=*/blank,
            /*log_probs=*/workspace.GetPointerToLogProbs(),
            /*srcLengths=*/srcLengths,
            /*tgtLengths=*/tgtLengths,
            /*alpha_counters=*/workspace.GetPointerToAlphaCounters(),
            /*alphas=*/(volatile DTYPE*)alphas,
            H);

    if (cudaGetLastError() != cudaSuccess) {
      return COMPUTE_ALPHAS_BETAS_COSTS_FAILED;
    }
  }

  return SUCCESS;
}

template <typename DTYPE, typename CAST_DTYPE>
status_t ComputeBetas(
    const Workspace<CAST_DTYPE>& workspace,
    const DTYPE* logits,
    const int* targets,
    const int* srcLengths,
    const int* tgtLengths,
    DTYPE* costs,
    DTYPE* betas) {
  const Options& options = workspace.GetOptions();

  const cudaStream_t& stream = options.stream_;
  const int& B = options.batchSize_;
  const int& H = options.nHypos_;
  const int& max_T = options.maxSrcLen_;
  const int& max_U = options.maxTgtLen_;
  const int& D = options.numTargets_;
  const int& blank = options.blank_;

  { // compute denominators.
    status_t status = LogSumExp2D<DTYPE, CAST_DTYPE>(
        /*stream=*/stream,
        /*N=*/B * H * max_T * max_U,
        /*D=*/D,
        /*logits=*/logits,
        /*denominators=*/workspace.GetPointerToDenominators());

    if (status != SUCCESS) {
      return status;
    }
  }

  { // compute log probability pairs (blank and target).
    int num_segments =
        (max_T + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
    dim3 block_dims(num_segments, max_U, B * H);
    dim3 thread_dims(MAX_THREADS_PER_BLOCK);

    ComputeLogProbs<DTYPE, CAST_DTYPE
    ><<<block_dims, thread_dims, 0, stream>>>(
        /*max_src_len=*/max_T,
        /*max_tgt_len=*/max_U,
        /*num_targets=*/D,
        /*blank=*/blank,
        /*logits=*/logits,
        /*targets=*/targets,
        /*srcLengths=*/srcLengths,
        /*tgtLengths=*/tgtLengths,
        /*denominators=*/workspace.GetPointerToDenominators(),
        /*log_probs=*/workspace.GetPointerToLogProbs(),
        H);

    if (cudaGetLastError() != cudaSuccess) {
      return COMPUTE_LOG_PROBS_FAILED;
    }
  }
  { // compute betas
    // warp is usually a group of threads (32)
    int num_warps = (max_T + WARP_SIZE - 1) / WARP_SIZE;

    // each block is identified by 3 d tuple.
    // we are using num_warp * max_U * B blocks
    // where num_warp is division among Time axis
    dim3 block_dims(num_warps, max_U, B * H);

    // each thread is identified by a 2 d tuple
    // 2nd dim is 1 for betas only
    dim3 thread_dims(WARP_SIZE, 1);

    ComputeBetasWrapper<DTYPE, CAST_DTYPE>
        <<<block_dims, thread_dims, 0, stream>>>(
            /*max_src_len=*/max_T,
            /*max_tgt_len=*/max_U,
            /*num_targets=*/D,
            /*blank=*/blank,
            /*log_probs=*/workspace.GetPointerToLogProbs(),
            /*srcLengths=*/srcLengths,
            /*tgtLengths=*/tgtLengths,
            /*alpha_counters=*/workspace.GetPointerToBetaCounters(),
            /*alphas=*/(volatile DTYPE*)betas,
            costs,
            H);

    if (cudaGetLastError() != cudaSuccess) {
      return COMPUTE_ALPHAS_BETAS_COSTS_FAILED;
    }
  }

  return SUCCESS;
}

} // namespace gpu
} // namespace rnnt
} // namespace torchaudio

#endif // USE_CUDA
