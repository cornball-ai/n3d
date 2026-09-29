# Encoder modules. Dimension key: B batch, L encoder frames, T mel frames,
# M mel bins, D hidden size, H heads, K head dim.

#' Feature Stacking
#'
#' Concatenates each group of \code{subsampling_factor} mel frames and
#' projects it to the hidden size. The last incomplete group is zero-padded.
#' @noRd
feature_stacking <- torch::nn_module(
                                     "FeatureStacking",
                                     initialize = function(n_mels, hidden_size, factor) {
    self$factor <- factor
    self$projection <- torch::nn_linear(factor * n_mels, hidden_size,
                                        bias = FALSE)
},
                                     forward = function(features_BTM) {
    num_frames <- features_BTM$shape[2]
    pad <- (-num_frames) %% self$factor
    if (pad > 0L) {
        features_BTM <- torch::nnf_pad(features_BTM, c(0L, 0L, 0L, pad))
    }
    b <- features_BTM$shape[1]
    stacked <- features_BTM$reshape(c(b, (num_frames + pad) %/% self$factor,
                                      -1L))
    self$projection(stacked)
}
)

encoder_attention <- torch::nn_module(
                                      "EncoderAttention",
                                      initialize = function(hidden_size, num_heads) {
    self$num_heads <- num_heads
    self$head_dim <- hidden_size %/% num_heads
    self$q_proj <- torch::nn_linear(hidden_size, hidden_size, bias = FALSE)
    self$k_proj <- torch::nn_linear(hidden_size, hidden_size, bias = FALSE)
    self$v_proj <- torch::nn_linear(hidden_size, hidden_size, bias = FALSE)
    self$o_proj <- torch::nn_linear(hidden_size, hidden_size, bias = TRUE)
},
                                      forward = function(x_BLD, rope, mask = NULL) {
    b <- x_BLD$shape[1]
    l <- x_BLD$shape[2]
    split <- function(t) {
        t$view(c(b, l, self$num_heads, self$head_dim))$transpose(2L, 3L)
    }
    q_BHLK <- apply_rope(split(self$q_proj(x_BLD)), rope)
    k_BHLK <- apply_rope(split(self$k_proj(x_BLD)), rope)
    v_BHLK <- split(self$v_proj(x_BLD))
    out_BHLK <- torch::torch_scaled_dot_product_attention(q_BHLK, k_BHLK,
        v_BHLK, attn_mask = mask)
    out_BLD <- out_BHLK$transpose(2L, 3L)$reshape(c(b, l, -1L))
    self$o_proj(out_BLD)
}
)

encoder_mlp <- torch::nn_module(
                                "EncoderMLP",
                                initialize = function(hidden_size, intermediate_size) {
    self$fc1 <- torch::nn_linear(hidden_size, intermediate_size)
    self$fc2 <- torch::nn_linear(intermediate_size, hidden_size)
},
                                forward = function(x) {
    self$fc2(torch::nnf_gelu(self$fc1(x)))
}
)

encoder_layer <- torch::nn_module(
                                  "EncoderLayer",
                                  initialize = function(hidden_size, intermediate_size, num_heads) {
    self$self_attn <- encoder_attention(hidden_size, num_heads)
    self$layer_norm1 <- torch::nn_layer_norm(hidden_size)
    self$mlp <- encoder_mlp(hidden_size, intermediate_size)
    self$layer_norm2 <- torch::nn_layer_norm(hidden_size)
},
                                  forward = function(x, rope, mask = NULL) {
    x <- x + self$self_attn(self$layer_norm1(x), rope, mask)
    x + self$mlp(self$layer_norm2(x))
}
)

#' Audio Tower
#'
#' Pre-norm transformer encoder over stacked mel frames with RoPE.
#' @noRd
audio_tower <- torch::nn_module(
                                "AudioTower",
                                initialize = function(cfg) {
    self$num_heads <- cfg$num_heads
    self$head_dim <- cfg$hidden_size %/% cfg$num_heads
    self$rope_theta <- cfg$rope_theta
    self$embedder <- feature_stacking(cfg$n_mels, cfg$hidden_size,
                                      cfg$subsampling_factor)
    self$input_layer_norm <- torch::nn_layer_norm(cfg$hidden_size)
    self$layers <- torch::nn_module_list()
    for (i in seq_len(cfg$num_layers)) {
        self$layers$append(encoder_layer(cfg$hidden_size,
                cfg$intermediate_size,
                cfg$num_heads))
    }
    self$layer_norm <- torch::nn_layer_norm(cfg$hidden_size)
},
                                # embeds_BLD: stacked-frame embeddings; valid_BL: optional logical mask of
                                # valid key frames.
                                forward = function(embeds_BLD, valid_BL = NULL) {
    rope <- rope_cos_sin(embeds_BLD$shape[2], self$head_dim, self$rope_theta,
                         device = embeds_BLD$device, dtype = embeds_BLD$dtype)
    mask <- NULL
    if (!is.null(valid_BL)) {
        # additive key mask, (B, 1, 1, L)
        mask <- torch::torch_zeros_like(valid_BL, dtype = embeds_BLD$dtype)
        mask <- mask$masked_fill(valid_BL$logical_not(), -Inf)
        mask <- mask$view(c(valid_BL$shape[1], 1L, 1L, -1L))
    }
    h <- self$input_layer_norm(embeds_BLD)
    for (i in seq_along(self$layers)) {
        h <- self$layers[[i]](h, rope, mask)
    }
    self$layer_norm(h)
}
)
