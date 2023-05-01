#pragma once

#ifdef USE_CUDA

#include <cassert>

#include <torchaudio/csrc/rnnt/gpu/kernel_utils.h>
#include <torchaudio/csrc/rnnt/gpu/kernels.h>
#include <torchaudio/csrc/rnnt/gpu/math.cuh>

namespace torchaudio {
namespace rnnt {

template <typename DTYPE, typename CAST_DTYPE>
__global__ void ComputeGaussianMap(
    int maxSrcLen,
    int maxTgtLen,
    const int* srcLengths,
    const int* tgtLengths,
    int H, //DTYPE slope, DTYPE sigma, DTYPE denom, // 1/sqrt( 2*pi*sigma^2 )
    DTYPE weight,
    CAST_DTYPE* outputs) {
  const int& maxT = maxSrcLen;
  const int& maxU = maxTgtLen;
  
  const int bTgt = blockIdx.z; // 0 <= b < B
  const int bSrc = bTgt / H;
  const int T = srcLengths[bSrc];
  const int U = tgtLengths[bTgt] + 1;
  DTYPE slope = (DTYPE)T / (DTYPE)U;
  DTYPE sigma = (DTYPE)U;
  DTYPE denom = 1.0 / std::sqrt(2 * M_PI * sigma * sigma);

  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  const int u = blockIdx.y;

  if (t >= T || u >= U) { // out of boundary.
    return;
  }

  Indexer3D indexer(maxT, maxU);
  
  int idx = indexer(bTgt, t, u);
  
  //for blank
  //TODO: check if it is thread safe
  outputs[idx] = std::log(1 + weight * std::exp( -(t - slope * u) * (t - slope * u) / (2 * sigma * sigma)) * denom);
  if( outputs[idx] == -INFINITY ){
      outputs[idx] = 0; //std::log(1)
  }
}

template <typename DTYPE, typename CAST_DTYPE>
__global__ void ComputeLogProbs(
    int maxSrcLen,
    int maxTgtLen,
    int numTargets,
    int blank,
    const DTYPE* logits,
    const int* targets,
    const int* srcLengths,
    const int* tgtLengths,
    const CAST_DTYPE* denominators,
    CAST_DTYPE* logProbs,
    int H = 1) {
  const int& maxT = maxSrcLen;
  const int& maxU = maxTgtLen;
  const int& D = numTargets;

  const int bTgt = blockIdx.z; // 0 <= b < B
  const int bSrc = bTgt / H;
  const int T = srcLengths[bSrc];
  const int U = tgtLengths[bTgt] + 1;

  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  const int u = blockIdx.y;

  if (t >= T || u >= U) { // out of boundary.
    return;
  }

  Indexer3D indexer(maxT, maxU);

  int idx = indexer(bTgt, t, u);

  // skip: log_prob(b, t, u).skip() = logits(b, t, u, blank) - denom(b, t, u).
  logProbs[(idx << 1) + LOG_PROBS_SKIP_IDX] =
      CAST_DTYPE(logits[idx * D + blank]) - denominators[idx];

  if (u < U - 1) {
    // emit: log_prob(b, t, u).emit() = logits(b, t, u, tgt[u]) - denom(b, t,
    // u).
    int target = targets[Indexer2D(maxU - 1)(bTgt, u)];
    logProbs[(idx << 1) + LOG_PROBS_EMIT_IDX] =
        CAST_DTYPE(logits[idx * D + target]) - denominators[idx];
  }
}

template <typename DTYPE, typename CAST_DTYPE>
__device__ void ComputeAlphas(
    int maxSrcLen,
    int maxTgtLen,
    int numTargets,
    int blank,
    const CAST_DTYPE* logProbs,
    const int* srcLengths,
    const int* tgtLengths,
    int* alpha_counters,
    volatile CAST_DTYPE* alphas,
    int H = 1) {
  const int& maxT = maxSrcLen;
  const int& maxU = maxTgtLen;

  const int bTgt = blockIdx.z; // 0 <= b < B
  const int bSrc = bTgt / H;
  const int T = srcLengths[bSrc];
  const int U = tgtLengths[bTgt] + 1;

  const int t = blockIdx.x * blockDim.x + threadIdx.x + 1;
  const int u = blockIdx.y + 1;

  if (t >= T || u >= U) { // out of boundary.
    return;
  }

  int* counter = alpha_counters + Indexer2D(maxU)(bTgt, blockIdx.y);

  Indexer3D idxr(maxT, maxU);

  if (t == 1 && u == 1) {
    alphas[idxr(bTgt, 0, 0)] = 0;
  }

  if (blockIdx.x > 0) { // wait for previous warp (in t-axis) is ready.
    while (atomicAdd(counter, 0) < blockIdx.x) {
    }
  }

  if (blockIdx.y > 0) { // wait for previous warp (in u-axis) is ready.
    while (atomicAdd(counter - 1, 0) <= blockIdx.x) {
    }
  }

  if (t == 1 && u < U) {
    // alpha(0, u) = alpha(0, u - 1) + logProbs(0, u - 1).emit().
    alphas[idxr(bTgt, 0, u)] = alphas[idxr(bTgt, 0, u - 1)] +
        logProbs[(idxr(bTgt, 0, u - 1) << 1) + LOG_PROBS_EMIT_IDX];
  }

  if (blockIdx.y == 0 && t < T) {
    CAST_DTYPE skip_prob =
        logProbs[(idxr(bTgt, t - 1, 0) << 1) + LOG_PROBS_SKIP_IDX];
    CAST_DTYPE val;

#pragma unroll
    for (int i = 1; i < warpSize; i <<= 1) {
      val = __shfl_up_sync(0xffffffff, skip_prob, i);
      if (i <= threadIdx.x) {
        skip_prob = skip_prob + val;
      }
    }

    val = alphas[idxr(bTgt, blockIdx.x * blockDim.x, 0)];
    alphas[idxr(bTgt, t, 0)] = skip_prob + val;
  }
  
  if (t < T && u < U) {
    CAST_DTYPE skip_prob =
        logProbs[(idxr(bTgt, t - 1, u) << 1) + LOG_PROBS_SKIP_IDX];
    CAST_DTYPE emit_prob =
        logProbs[(idxr(bTgt, t, u - 1) << 1) + LOG_PROBS_EMIT_IDX];

    CAST_DTYPE skip =
        alphas[idxr(bTgt, blockIdx.x * blockDim.x, u)] + skip_prob;
    CAST_DTYPE emit = alphas[idxr(bTgt, t, u - 1)] + emit_prob;

    CAST_DTYPE val = math::lse(skip, emit);
    CAST_DTYPE out = val;

    for (int i = 1; i < warpSize; ++i) {
      val = __shfl_up_sync(0xffffffff, val, 1);
      if (i == threadIdx.x) {
        val = math::lse(val + skip_prob, emit);
        out = val;
      }
    }

    alphas[idxr(bTgt, t, u)] = out;
  }

  if (threadIdx.x == 0) {
    __threadfence();
    atomicAdd(counter, 1);
  }
}

template <typename DTYPE, typename CAST_DTYPE>
__device__ void ComputeBetasCosts(
    int maxSrcLen,
    int maxTgtLen,
    int numTargets,
    int blank,
    const CAST_DTYPE* logProbs,
    const int* srcLengths,
    const int* tgtLengths,
    int* betaCounters,
    volatile CAST_DTYPE* betas,
    DTYPE* costs,
    int H = 1,
    const bool fastEmit = false,
    const DTYPE fastEmitWeight = 0.0,
    const bool lossRegularization = false,
    const DTYPE lossRegWeight = 0.0,
    volatile CAST_DTYPE* lossRegMap = nullptr) {
  const int& maxT = maxSrcLen;
  const int& maxU = maxTgtLen;

  const int bTgt = blockIdx.z; // 0 <= b < B
  const int bSrc = bTgt / H;
  const int T = srcLengths[bSrc];
  const int U = tgtLengths[bTgt] + 1;

  const int t = T - 2 - blockIdx.x * blockDim.x - threadIdx.x;
  const int u = U - 2 - blockIdx.y;

  if (t < 0 || u < 0) { // out of boundary.
    return;
  }

  int* counter = betaCounters + Indexer2D(maxU)(bTgt, blockIdx.y);

  Indexer3D idxr(maxT, maxU);

  if (t == T - 2 && u == U - 2) {
    CAST_DTYPE regMap;
    if( lossRegularization ){
       regMap = lossRegMap[idxr(bTgt, t, u)];
    }
    else{
      regMap = 0;
    }
    betas[idxr(bTgt, T - 1, U - 1)] =
        logProbs[(idxr(bTgt, T - 1, U - 1) << 1) + LOG_PROBS_SKIP_IDX] + regMap;
    
  }
  
  if (blockIdx.x > 0) { // wait for previous warp (in t-axis) is ready.
    while (atomicAdd(counter, 0) < blockIdx.x) {
    }
  }

  if (blockIdx.y > 0) { // wait for previous warp (in u-axis) is ready.
    while (atomicAdd(counter - 1, 0) <= blockIdx.x) {
    }
  }

  if (t == T - 2 && u >= 0) {
    CAST_DTYPE regMap;
    if( lossRegularization ){
       regMap = lossRegMap[idxr(bTgt, t, u)];
    }
    else{
      regMap = 0;
    }
    betas[idxr(bTgt, T - 1, u)] = betas[idxr(bTgt, T - 1, u + 1)] +
        logProbs[(idxr(bTgt, T - 1, u) << 1) + LOG_PROBS_EMIT_IDX] + regMap;
  }

  if (blockIdx.y == 0 && t >= 0) {
    CAST_DTYPE skip_prob =
        logProbs[(idxr(bTgt, t, U - 1) << 1) + LOG_PROBS_SKIP_IDX];
    CAST_DTYPE val;

#pragma unroll
    for (int i = 1; i < warpSize; i <<= 1) {
      val = __shfl_up_sync(0xffffffff, skip_prob, i);
      if (i <= threadIdx.x) {
        skip_prob = skip_prob + val;
      }
    }

    CAST_DTYPE regMap;
    if( lossRegularization ){
       regMap = lossRegMap[idxr(bTgt, t, u)];
    }
    else{
      regMap = 0;
    }
    betas[idxr(bTgt, t, U - 1)] =
        betas[idxr(bTgt, T - 1 - blockIdx.x * blockDim.x, U - 1)] + skip_prob + regMap;
  }

  if (t >= 0 && u >= 0) {
    CAST_DTYPE skip_prob =
        logProbs[(idxr(bTgt, t, u) << 1) + LOG_PROBS_SKIP_IDX];
    CAST_DTYPE emit_prob =
        logProbs[(idxr(bTgt, t, u) << 1) + LOG_PROBS_EMIT_IDX];

    CAST_DTYPE skip = betas[idxr(bTgt, t + threadIdx.x + 1, u)] + skip_prob;
    CAST_DTYPE emit = betas[idxr(bTgt, t, u + 1)] + emit_prob;

    CAST_DTYPE val = math::lse(skip, emit);
    CAST_DTYPE out = val;

    for (int i = 1; i < warpSize; ++i) {
      val = __shfl_up_sync(0xffffffff, val, 1);
      if (i == threadIdx.x) {
        val = math::lse(val + skip_prob, emit);
        out = val;
      }
    }
    
    CAST_DTYPE regMap;
    if( lossRegularization ){
       regMap = lossRegMap[idxr(bTgt, t, u)];
    }
    else{
      regMap = 0;
    }

    betas[idxr(bTgt, t, u)] = out + regMap;

    if (t == 0 && u == 0) { // use -beta(0, 0) as cost.
      costs[bTgt] = DTYPE(-(out + regMap));
    }
  }

  if (threadIdx.x == 0) {
    __threadfence();
    atomicAdd(counter, 1);
  }
}

template <typename DTYPE, typename CAST_DTYPE>
__global__ void ComputeAlphasBetasCosts(
    int maxSrcLen,
    int maxTgtLen,
    int numTargets,
    int blank,
    const CAST_DTYPE* logProbs,
    const int* srcLengths,
    const int* tgtLengths,
    int* alpha_counters,
    volatile CAST_DTYPE* alphas,
    int* betaCounters,
    volatile CAST_DTYPE* betas,
    DTYPE* costs,
    int warpSize = 0,
    int numWarps = 0,
    int H = 1,
    const bool fastEmit = false,
    const DTYPE fastEmitWeight = 0.0,
    const bool lossRegularization = false,
    const DTYPE lossRegWeight = 0.0,
    CAST_DTYPE* lossRegMap = nullptr) {
  assert(threadIdx.y == 0 || threadIdx.y == 1);

  if (threadIdx.y == 0) {
    ComputeAlphas<DTYPE, CAST_DTYPE>(
        /*maxSrcLen=*/maxSrcLen,
        /*maxTgtLen=*/maxTgtLen,
        /*numTargets=*/numTargets,
        /*blank=*/blank,
        /*logProbs=*/logProbs,
        /*srcLengths=*/srcLengths,
        /*tgtLengths=*/tgtLengths,
        /*alpha_counters=*/alpha_counters,
        /*alphas=*/alphas,
        H);
  } else { // threadIdx.y == 1
    ComputeBetasCosts<DTYPE, CAST_DTYPE>(
        /*maxSrcLen=*/maxSrcLen,
        /*maxTgtLen=*/maxTgtLen,
        /*numTargets=*/numTargets,
        /*blank=*/blank,
        /*logProbs=*/logProbs,
        /*srcLengths=*/srcLengths,
        /*tgtLengths=*/tgtLengths,
        /*betaCounters=*/betaCounters,
        /*beta=*/betas,
        /*costs=*/costs,
        H,
        fastEmit,
        fastEmitWeight,
        lossRegularization,
        lossRegWeight,
        lossRegMap);
  }
}

template <typename DTYPE, typename CAST_DTYPE>
__global__ void ComputeGradients(
    int maxSrcLen,
    int maxTgtLen,
    int numTargets,
    int blank,
    CAST_DTYPE clamp,
    const DTYPE* logits,
    const int* targets,
    const int* srcLengths,
    const int* tgtLengths,
    const CAST_DTYPE* denominators,
    const CAST_DTYPE* alphas,
    const CAST_DTYPE* betas,
    DTYPE* gradients,
    int H = 1,
    const bool fastEmit = false,
    const DTYPE fastEmitWeight = 0.0,
    const bool lossRegularization = false, 
    const DTYPE lossRegWeight = 0.0,
    CAST_DTYPE* lossRegMap = nullptr) {
  const int bTgt = blockIdx.z; // 0 <= b < B
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  const int u = blockIdx.y;

  ComputeGradientsElement(
      bTgt,
      t,
      u,
      maxSrcLen,
      maxTgtLen,
      numTargets,
      blank,
      clamp,
      logits,   
      targets,
      srcLengths,
      tgtLengths,
      denominators,
      alphas,
      betas,
      gradients,
      H,
      fastEmit,
      fastEmitWeight,
      lossRegularization,
      lossRegWeight,
      lossRegMap);
}

// This is a __global__ wrapper around ComputeAlphas
// device kernel to enable unit testing
template <typename DTYPE, typename CAST_DTYPE>
__global__ void ComputeAlphasWrapper(
    int maxSrcLen,
    int maxTgtLen,
    int numTargets,
    int blank,
    const CAST_DTYPE* logProbs,
    const int* srcLengths,
    const int* tgtLengths,
    int* alpha_counters,
    volatile CAST_DTYPE* alphas,
    int H = 1) {
  ComputeAlphas<DTYPE, CAST_DTYPE>(
      maxSrcLen,
      maxTgtLen,
      numTargets,
      blank,
      logProbs,
      srcLengths,
      tgtLengths,
      alpha_counters,
      alphas,
      H);
}

// This is a __global__ wrapper around ComputeBetas
// device kernel to enable unit testing
template <typename DTYPE, typename CAST_DTYPE>
__global__ void ComputeBetasWrapper(
    int maxSrcLen,
    int maxTgtLen,
    int numTargets,
    int blank,
    const CAST_DTYPE* logProbs,
    const int* srcLengths,
    const int* tgtLengths,
    int* betaCounters,
    volatile CAST_DTYPE* betas,
    DTYPE* costs,
    int H = 1,
    const bool fastEmit = false,
    const DTYPE fastEmitWeight = 0.0,
    const bool lossRegularization = false, 
    const DTYPE lossRegWeight = 0.0,
    CAST_DTYPE* lossRegMap = nullptr) {
  ComputeBetasCosts<DTYPE, CAST_DTYPE>(
      maxSrcLen,
      maxTgtLen,
      numTargets,
      blank,
      logProbs,
      srcLengths,
      tgtLengths,
      betaCounters,
      betas,
      costs,
      H,
      fastEmit,
      fastEmitWeight,
      lossRegularization,
      lossRegWeight,
      lossRegMap);
}

// #undef LOG_PROBS_SKIP_IDX
// #undef LOG_PROBS_EMIT_IDX

} // namespace rnnt
} // namespace torchaudio

#endif // USE_CUDA
