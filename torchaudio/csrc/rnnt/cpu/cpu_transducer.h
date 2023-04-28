#pragma once
//#define GARBAGE_DEBUG
#include <torchaudio/csrc/rnnt/cpu/cpu_kernels.h>
#include <torchaudio/csrc/rnnt/workspace.h>

namespace torchaudio {
namespace rnnt {
namespace cpu {

static int incount = 0;
// Inputs:
//   workspace: workspace.
//   logits: pointer to (B, maxT, maxU, D) logits.
//   targets: pointer to (B, maxU - 1) targets in the batch.
//   srcLengths: pointer to (B, ) source lengths in the batch.
//   tgtLengths: pointer to (B, ) target lengths in the batch.
//
// Outputs:
//   costs: pointer to (B, ) costs in the batch.
//   gradients: pointer to (B, maxT, maxU, D) gradients in the batch.
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

  CHECK_EQ(options.device_, CPU);

  const int& B = options.batchSize_;
  const int& maxT = options.maxSrcLen_;
  const int& maxU = options.maxTgtLen_;
  const int& D = options.numTargets_;
  const int& blank = options.blank_;
  const DTYPE clamp = options.clamp_;
  const bool fast_emit = options.fastEmit_;
  const DTYPE fast_emit_weight = options.fastEmitWeight_;
  const bool loss_regularization = options.lossRegularization_;
  const DTYPE loss_regularization_weight = options.lossRegularizationWeight_;
  const DTYPE loss_regularization_sigma = options.lossRegularizationSigma_;

  { // compute denominators.
    LogSumExp2D<DTYPE, CAST_DTYPE>(
        /*N=*/B * maxT * maxU,
        /*D=*/D,
        /*logits=*/logits,
        /*denominators=*/workspace.GetPointerToDenominators());
  }

  {
    if (loss_regularization){
      DTYPE slope = maxT / maxU;
      DTYPE sigma = options.lossRegularizationSigma_;
      DTYPE weight = options.lossRegularizationWeight_;
      DTYPE denom = 1.0 / sqrt( 2 * M_PI * sigma * sigma );
      
      ComputeGaussianMap<DTYPE, CAST_DTYPE>(
        maxT,
        maxU,
        D,
        srcLengths,
        tgtLengths,
        B,
        slope,
        sigma,
        weight,
        denom,
        workspace.GetPointerToLossRegularization()
      );
    }
  }

  { // compute log prob pairs.
    ComputeLogProbs<DTYPE, CAST_DTYPE>(
        /*options=*/options,
        /*logits=*/logits,
        /*targets=*/targets,
        /*srcLengths=*/srcLengths,
        /*tgtLengths=*/tgtLengths,
        /*denominators=*/workspace.GetPointerToDenominators(),
        /*log_probs=*/workspace.GetPointerToLogProbs());
  }

  { // compute alphas and betas.
    ComputeAlphasBetas<DTYPE, CAST_DTYPE>(
        /*options=*/options,
        /*log_probs=*/workspace.GetPointerToLogProbs(),
        /*srcLengths=*/srcLengths,
        /*tgtLengths=*/tgtLengths,
        /*alphas=*/workspace.GetPointerToAlphas(),
        /*betas=*/workspace.GetPointerToBetas(),
        /*costs=*/costs,
        fast_emit,
        fast_emit_weight,
        loss_regularization,
        loss_regularization_weight,
        workspace.GetPointerToLossRegularization());
  }
#ifdef GARBAGE_DEBUG
  {
    std::vector<TensorView<CAST_DTYPE>> gg;
    std::vector<TensorView<CAST_DTYPE>> ggA;
    std::vector<TensorView<CAST_DTYPE>> ggB;
    float* mCosts = new float[B];
    memcpy(mCosts, costs, B*sizeof(float));
    for(int b=0; b<B; b++){
      int costIdx = b * maxT * maxU;
      gg.push_back(TensorView<CAST_DTYPE>({maxT, maxU}, workspace.GetPointerToLossRegularization() + costIdx));
      ggA.push_back(TensorView<CAST_DTYPE>({maxT, maxU}, workspace.GetPointerToAlphas() + costIdx));
      ggB.push_back(TensorView<CAST_DTYPE>({maxT, maxU}, workspace.GetPointerToBetas() + costIdx));
      std::cout << srcLengths[b] << ", " << tgtLengths[b] << ", "
                << gg[b]({0,0}) << ", "
                << mCosts[b] << ", "
                << ggB[b]({0,0})
                << " MY_DEBUG2" << std::endl;
    }
    int myt=0;
    int myu=0;
    std::cout << "loss regularization weight: " << loss_regularization_weight << std::endl;
    std::cout << "loss regularization sigma: " << loss_regularization_sigma << std::endl;
    float mydenom = 1.0 / sqrt( 2 * M_PI * pow(loss_regularization_sigma, 2 ));
    float myslope = maxT / maxU;
    float result = log(1 + loss_regularization_weight * exp(-(pow(myt-myslope * myu, 2))/(2*pow(loss_regularization_sigma, 2)))*mydenom);
    std::cout << "loss regularization map " <<myt << ", " << myu <<": " <<result <<std::endl;
    std::cout << options << std::endl;
    std::cout << std::endl; 

    std::cout << "GARBAGE MAP" << std::endl;
    std::vector<TensorView<CAST_DTYPE>> garbage;
    for(int b=0; b<1; b++){
      garbage.push_back(
          TensorView<CAST_DTYPE>({maxT, maxU}, workspace.GetPointerToLossRegularization() + b * maxT * maxU));
      for(int u=0; u<tgtLengths[b]; u++){
        for(int t=0; t<srcLengths[b]; t++){
          std::cout << garbage[b]({t, u}) << ", ";
        }
        std::cout << std::endl;
      }
      std::cout << std::endl;
      std::cout << std::endl;
    }
    
    std::cout << "GARBAGE ALPHAS" << std::endl;
    std::vector<TensorView<CAST_DTYPE>> garbage_alpha;
    for(int b=0; b<1; b++){
      garbage_alpha.push_back(
          TensorView<CAST_DTYPE>({maxT, maxU}, workspace.GetPointerToAlphas() + b * maxT * maxU));
      for(int u=0; u<tgtLengths[b]; u++){
        for(int t=0; t<srcLengths[b]; t++){
          std::cout << garbage_alpha[b]({t, u}) << ", ";
        }
        std::cout << std::endl;
      }
      std::cout << std::endl;
      std::cout << std::endl;
    }

    std::cout << "GARBAGE BETAS" << std::endl;
    std::vector<TensorView<CAST_DTYPE>> garbage_beta;
    for(int b=0; b<1; b++){
      garbage_beta.push_back(
          TensorView<CAST_DTYPE>({maxT, maxU}, workspace.GetPointerToBetas() + b * maxT * maxU));
      for(int u=0; u<tgtLengths[b]; u++){
        for(int t=0; t<srcLengths[b]; t++){
          std::cout << garbage_beta[b]({t, u}) << ", ";
        }
        std::cout << std::endl;
      }
      std::cout << std::endl;
      std::cout << std::endl;
    }
    
  }
#endif
  if (gradients != nullptr) {
    ComputeGradients<DTYPE, CAST_DTYPE>(
        /*options=*/options,
        /*logits=*/logits,
        /*targets=*/targets,
        /*srcLengths=*/srcLengths,
        /*tgtLengths=*/tgtLengths,
        /*denominators=*/workspace.GetPointerToDenominators(),
        /*alphas=*/workspace.GetPointerToAlphas(),
        /*betas=*/workspace.GetPointerToBetas(),
        /*gradients=*/gradients,
        fast_emit,
        fast_emit_weight,
        loss_regularization,
        loss_regularization_weight,
        workspace.GetPointerToLossRegularization());
  }
#ifdef GARBAGE_DEBUG
  if( incount == 2 ){
    std::cout << "GARBAGE IS OVER" << std::endl;
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

  CHECK_EQ(options.device_, CPU);

  const int& B = options.batchSize_;
  const int& maxT = options.maxSrcLen_;
  const int& maxU = options.maxTgtLen_;
  const int& D = options.numTargets_;

  { // compute denominators.
    LogSumExp2D<DTYPE, CAST_DTYPE>(
        /*N=*/B * maxT * maxU,
        /*D=*/D,
        /*logits=*/logits,
        /*denominators=*/workspace.GetPointerToDenominators());
  }

  { // compute log prob pairs.
    ComputeLogProbs<DTYPE, CAST_DTYPE>(
        /*options=*/options,
        /*logits=*/logits,
        /*targets=*/targets,
        /*srcLengths=*/srcLengths,
        /*tgtLengths=*/tgtLengths,
        /*denominators=*/workspace.GetPointerToDenominators(),
        /*log_probs=*/workspace.GetPointerToLogProbs());
  }

  { // compute alphas.
    ComputeAlphas<DTYPE, CAST_DTYPE>(
        /*options=*/options,
        /*log_probs=*/workspace.GetPointerToLogProbs(),
        /*srcLengths=*/srcLengths,
        /*tgtLengths=*/tgtLengths,
        /*alphas=*/alphas);
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

  CHECK_EQ(options.device_, CPU);

  const int& B = options.batchSize_;
  const int& maxT = options.maxSrcLen_;
  const int& maxU = options.maxTgtLen_;
  const int& D = options.numTargets_;

  { // compute denominators.
    LogSumExp2D<DTYPE, CAST_DTYPE>(
        /*N=*/B * maxT * maxU,
        /*D=*/D,
        /*logits=*/logits,
        /*denominators=*/workspace.GetPointerToDenominators());
  }

  { // compute log prob pairs.
    ComputeLogProbs<DTYPE, CAST_DTYPE>(
        /*options=*/options,
        /*logits=*/logits,
        /*targets=*/targets,
        /*srcLengths=*/srcLengths,
        /*tgtLengths=*/tgtLengths,
        /*denominators=*/workspace.GetPointerToDenominators(),
        /*log_probs=*/workspace.GetPointerToLogProbs());
  }

  { // compute betas.
    ComputeBetas<DTYPE, CAST_DTYPE>(
        /*options=*/options,
        /*log_probs=*/workspace.GetPointerToLogProbs(),
        /*srcLengths=*/srcLengths,
        /*tgtLengths=*/tgtLengths,
        /*costs=*/costs,
        /*betas=*/betas);
  }

  return SUCCESS;
}

} // namespace cpu
} // namespace rnnt
} // namespace torchaudio
