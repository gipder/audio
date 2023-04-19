#pragma once

#ifdef USE_CUDA

#include <torchaudio/csrc/rnnt/workspace.h>
#include <torchaudio/csrc/rnnt/gpu/gpu_kernel_utils.cuh>
#include <torchaudio/csrc/rnnt/gpu/gpu_kernels.cuh>
#include <cmath>

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

template <typename DTYPE, typename CAST_DTYPE>
status_t ComputeMap(
    const Workspace<CAST_DTYPE>& workspace,
    const int* srcLengths,
    const int* tgtLengths,
    CAST_DTYPE* outputs) {
  { 
    const Options& options = workspace.GetOptions();

    const cudaStream_t& stream = options.stream_;
    const int& B = options.batchSize_;
    const int& H = options.nHypos_;
    const int& max_T = options.maxSrcLen_;
    const int& max_U = options.maxTgtLen_;
    const int& D = options.numTargets_;
    const int& blank = options.blank_;
   
    int num_segments =
          (max_T + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
    dim3 block_dims(num_segments, max_U, B * H);
    dim3 thread_dims(MAX_THREADS_PER_BLOCK);

    DTYPE slope = max_T / max_U;
    DTYPE sigma = options.lossRegularizationSigma_;
    DTYPE denom = 1.0 / sqrt( 2 * M_PI * sigma * sigma );    
    ComputeGaussianMap<DTYPE, CAST_DTYPE><<<block_dims, thread_dims, 0, stream>>>(
            /*maxSrcLen*/max_T,
            /*maxTgtLen*/max_U,
            /*srcLengths*/srcLengths,
            /*tgtLengths*/tgtLengths,
            /*nHypothesis*/H,
            /*slope*/slope,
            /*sigma*/sigma,
            /*denom*/denom,
            /*outputs*/outputs);

    // BUGBUG: These error codes are only accurate when launching with
    // blocking. Otherwise they usually reflect earlier errors.
    if (cudaGetLastError() != cudaSuccess) {
      return COMPUTE_DENOMINATOR_REDUCE_MAX_FAILED;
    }
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
    if (loss_regularization){
      int num_segments =
          (max_T + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
      dim3 block_dims(num_segments, max_U, B * H);
      dim3 thread_dims(MAX_THREADS_PER_BLOCK);

      DTYPE slope = max_T / max_U;
      DTYPE sigma = options.lossRegularizationSigma_;
      DTYPE denom = 1.0 / sqrt( 2 * M_PI * sigma * sigma );    
      //std::cout << "GARBAGE IS WORKING" << std::endl;
      //ComputeMap<DTYPE, CAST_DTYPE><<<block_dims, thread_dims, 0, stream>>>(
      ComputeGaussianMap<DTYPE, CAST_DTYPE><<<block_dims, thread_dims, 0, stream>>>(
        max_T,
        max_U,
        srcLengths,
        tgtLengths,
        H,
        slope,
        sigma,
        denom,
        workspace.GetPointerToLossRegularization()
      );
//#define MY_DEBUG
#ifdef MY_DEBUG      
      int mSize = options.BTU();
      DTYPE* garbage = new DTYPE[mSize];      
      cudaMemcpy(garbage, workspace.GetPointerToLossRegularization(), mSize * sizeof(DTYPE), cudaMemcpyDeviceToHost);
      int* tLen = new int[B];
      int* uLen = new int[B];
      cudaMemcpy(tLen, srcLengths, B * sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(uLen, tgtLengths, B * sizeof(int), cudaMemcpyDeviceToHost);
      Indexer3D idxr(max_T, max_U);
      
      for(int b=0; b<B; b++){
        std::cout << tLen[b] << ", " << uLen[b] << std::endl;  
      }
      
      for(int b=0; b<B; b++){
        std::cout << garbage[idxr(b, 0, 0)] << std::endl;        
      }
#endif      
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
    if (cudaGetLastError() != cudaSuccess) {
      return COMPUTE_ALPHAS_BETAS_COSTS_FAILED;
    }
  }

  if (gradients != nullptr) { // compute gradients.
    // don't set gradients to zero to here as gradients might reuse memory from
    // logits
#define MY_DEBUG2
#ifdef MY_DEBUG2
      int mSize = options.BTU();
      DTYPE* garbage = new DTYPE[mSize];
      cudaMemcpy(garbage, workspace.GetPointerToLossRegularization(), mSize * sizeof(DTYPE), cudaMemcpyDeviceToHost);
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
      
      for(int b=0; b<B; b++){
        int costIdx = b * max_T * max_U;
        std::cout << tLen[b] << ", " << uLen[b] << ", "         
                  << garbage[idxr(b, 0, 0)] << ", " 
                  << mCosts[b] << ", " 
                  << betas[idxr(b, 0, 0)]
                  << " MY_DEBUG2" << std::endl;
      }
      /*
            for(int b=0; b<B; b++){
        for(int u=0; u<uLen[b]; u++){
          for(int t=0; t<tLen[b]; t++){
            std::cout << garbage[idxr(b, t, u)] << ", ";
          }
          std::cout << std::endl;
        }
        std::cout << std::endl;
        std::cout << std::endl;
      }      
      */
     
#endif      
    int num_blocks =
        (max_T + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
    dim3 block_dims(num_blocks, max_U, B * H);
    dim3 thread_dims(MAX_THREADS_PER_BLOCK);

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
        loss_regularization,
        loss_regularization_weight,
        workspace.GetPointerToLossRegularization());    
//#define MY_DEBUG3
#ifdef MY_DEBUG3
      int mSize = options.BTU();
      DTYPE* garbage = new DTYPE[mSize];
      cudaMemcpy(garbage, workspace.GetPointerToLossRegularization(), mSize * sizeof(DTYPE), cudaMemcpyDeviceToHost);
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
      
      for(int b=0; b<B; b++){
        int costIdx = b * max_T * max_U;
        std::cout << tLen[b] << ", " << uLen[b] << ", "         
                  << garbage[idxr(b, 0, 0)] << ", " 
                  << mCosts[b] << ", " 
                  << betas[idxr(b, 0, 0)]
                  << " MY_DEBUG3" << std::endl;
      }     
#endif
    if(incount == 2){
      exit(3);
    }
    else{
      std::cout << std::endl;
      incount++;
    }
    
    if (cudaGetLastError() != cudaSuccess) {
      return COMPUTE_GRADIENTS_FAILED;
    }
  }

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
