#pragma once

#include <torch/script.h>

std::tuple<torch::Tensor, c10::optional<torch::Tensor>> rnnt_loss(
    torch::Tensor& logits,
    const torch::Tensor& targets,
    const torch::Tensor& logit_lengths,
    const torch::Tensor& target_lengths,
    int64_t blank,
    double clamp,
    const bool fast_emit,
    const double fast_emit_weight,
    const bool loss_regularization,
    const double loss_regularization_weight,
    const double loss_regularization_sigma);