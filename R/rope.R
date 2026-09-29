#' Rotary Position Embeddings
#'
#' Default RoPE over the full head dimension. Positions restart at 0 for
#' every chunk.
#'
#' @param seq_len Number of positions.
#' @param head_dim Attention head dimension.
#' @param theta RoPE base.
#' @param device Torch device.
#' @param dtype Output dtype.
#' @return A list of \code{cos} and \code{sin} tensors, each
#'   \code{(1, 1, seq_len, head_dim)}.
#' @noRd
rope_cos_sin <- function(seq_len, head_dim, theta = 10000, device = "cpu",
                         dtype = torch::torch_float32()) {
  exponents <- torch::torch_arange(0, head_dim - 1L, 2,
                                   dtype = torch::torch_float32(),
                                   device = device) / head_dim
  base <- torch::torch_scalar_tensor(theta, dtype = torch::torch_float32(),
                                     device = device)
  inv_freq <- 1 / torch::torch_pow(base, exponents)
  positions <- torch::torch_arange(0, seq_len - 1L,
                                   dtype = torch::torch_float32(),
                                   device = device)
  freqs_LK <- torch::torch_outer(positions, inv_freq)
  emb_LD <- torch::torch_cat(list(freqs_LK, freqs_LK), dim = -1L)
  list(cos = emb_LD$cos()$to(dtype = dtype)$view(c(1L, 1L, seq_len, -1L)),
       sin = emb_LD$sin()$to(dtype = dtype)$view(c(1L, 1L, seq_len, -1L)))
}

rotate_half <- function(x) {
  half <- x$shape[length(x$shape)] %/% 2L
  x1 <- x$narrow(-1L, 1L, half)
  x2 <- x$narrow(-1L, half + 1L, half)
  torch::torch_cat(list(-x2, x1), dim = -1L)
}

apply_rope <- function(x, rope) {
  x * rope$cos + rotate_half(x) * rope$sin
}
