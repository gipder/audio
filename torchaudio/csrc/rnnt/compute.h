#pragma once

#include <torch/script.h>

std::tuple<torch::Tensor, c10::optional<torch::Tensor>> rnnt_loss(
    torch::Tensor& logits,
    const torch::Tensor& targets,
    const torch::Tensor& logit_lengths,
    const torch::Tensor& target_lengths,
    int64_t blank,
    double clamp,
    bool fast_emit,
    double fast_emit_weight,
    bool loss_regularization,
    double loss_regularization_weight,
    double loss_regularization_sigma);