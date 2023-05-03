#include <torch/script.h>
#include <torchaudio/csrc/rnnt/compute.h>

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
    double loss_regularization_sigma,
    bool loss_regularization_swing) {
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("torchaudio::rnnt_loss", "")
                       .typed<decltype(rnnt_loss)>();

  return op.call(logits, targets, logit_lengths, target_lengths, blank, clamp, 
                 fast_emit, fast_emit_weight, 
                 loss_regularization, loss_regularization_weight, loss_regularization_sigma,
                 loss_regularization_swing);
}

TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
  m.def(
      "rnnt_loss(Tensor logits,"
      "Tensor targets,"
      "Tensor logit_lengths,"
      "Tensor target_lengths,"
      "int blank,"
      "float clamp,"
      "bool fast_emit,"
      "float fast_emit_weight,"
      "bool loss_regularization,"
      "float loss_regularization_weight,"
      "float loss_regularization_sigma,"
      "bool loss_regularization_swing) -> (Tensor, Tensor?)");            
}
