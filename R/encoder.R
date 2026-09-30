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

# The layer's parameters live in these modules so their names match the
# checkpoint (layers.0.self_attn.q_proj.weight, ...); the computation is
# layer_forward(), a function of tensors only, so one traced copy of it can
# serve every layer.
encoder_attention <- torch::nn_module(
                                      "EncoderAttention",
                                      initialize = function(hidden_size, num_heads) {
    self$q_proj <- torch::nn_linear(hidden_size, hidden_size, bias = FALSE)
    self$k_proj <- torch::nn_linear(hidden_size, hidden_size, bias = FALSE)
    self$v_proj <- torch::nn_linear(hidden_size, hidden_size, bias = FALSE)
    self$o_proj <- torch::nn_linear(hidden_size, hidden_size, bias = TRUE)
}
)

encoder_mlp <- torch::nn_module(
                                "EncoderMLP",
                                initialize = function(hidden_size, intermediate_size) {
    self$fc1 <- torch::nn_linear(hidden_size, intermediate_size)
    self$fc2 <- torch::nn_linear(intermediate_size, hidden_size)
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
                                  # The layer's tensors in layer_forward()'s argument order.
                                  weights = function() {
    list(self$layer_norm1$weight, self$layer_norm1$bias,
         self$self_attn$q_proj$weight, self$self_attn$k_proj$weight,
         self$self_attn$v_proj$weight, self$self_attn$o_proj$weight,
         self$self_attn$o_proj$bias, self$layer_norm2$weight,
         self$layer_norm2$bias, self$mlp$fc1$weight, self$mlp$fc1$bias,
         self$mlp$fc2$weight, self$mlp$fc2$bias)
}
)

#' One Encoder Layer as a Function of Tensors
#'
#' Pre-norm self-attention with RoPE, then a GELU MLP, each with a residual.
#' Every argument is a tensor so the function can be traced once and reused
#' for all layers. The sequence length is never read, so a trace accepts any
#' length; batch size is 1.
#'
#' @param num_heads,head_dim Attention geometry, fixed by the closure.
#' @return A function of \code{(h, cos, sin, mask, <13 layer weights>)}.
#' @noRd
make_layer_forward <- function(num_heads, head_dim) {
    hidden <- num_heads * head_dim
    force(hidden)
    function(h, cos, sin, mask, ln1_w, ln1_b, q_w, k_w, v_w, o_w, o_b,
             ln2_w, ln2_b, fc1_w, fc1_b, fc2_w, fc2_b) {
        split <- function(t) {
            t$view(c(1L, -1L, num_heads, head_dim))$transpose(2L, 3L)
        }
        rope <- list(cos = cos, sin = sin)
        x <- torch::nnf_layer_norm(h, hidden, ln1_w, ln1_b)
        q_BHLK <- apply_rope(split(torch::nnf_linear(x, q_w)), rope)
        k_BHLK <- apply_rope(split(torch::nnf_linear(x, k_w)), rope)
        v_BHLK <- split(torch::nnf_linear(x, v_w))
        a_BHLK <- torch::torch_scaled_dot_product_attention(q_BHLK, k_BHLK,
            v_BHLK, attn_mask = mask)
        a_BLD <- a_BHLK$transpose(2L, 3L)$reshape(c(1L, -1L, hidden))
        h <- h + torch::nnf_linear(a_BLD, o_w, o_b)
        x <- torch::nnf_layer_norm(h, hidden, ln2_w, ln2_b)
        x <- torch::nnf_gelu(torch::nnf_linear(x, fc1_w, fc1_b))
        h + torch::nnf_linear(x, fc2_w, fc2_b)
    }
}

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
    self$layer_forward <- make_layer_forward(self$num_heads, self$head_dim)
    # traced layer_forward(), one per CUDA device and dtype; see layer_fn()
    self$traces <- new.env(parent = emptyenv())
},
                                # embeds_BLD: stacked-frame embeddings, batch size 1; rope: from
                                # rope_cos_sin(); mask: optional additive key mask, (1, 1, 1, L).
                                forward = function(embeds_BLD, rope, mask = NULL) {
    h <- self$input_layer_norm(embeds_BLD)
    fn <- self$layer_fn(h)
    if (!identical(fn, self$layer_forward) && is.null(mask)) {
        mask <- torch::torch_zeros(1L, 1L, 1L, h$shape[2], dtype = h$dtype,
                                   device = h$device)
    }
    for (i in seq_along(self$layers)) {
        h <- do.call(fn, c(list(h, rope$cos, rope$sin, mask),
                           self$layers[[i]]$weights()))
    }
    self$layer_norm(h)
},
                                # On CUDA, layer_forward() traced to TorchScript. Inside a traced graph
                                # intermediates are freed as soon as they are consumed; run op by op from
                                # R, each stays allocated until R's garbage collector runs, which roughly
                                # quadruples peak device memory. Tracing one layer (weights are inputs)
                                # rather than all 31 keeps the executor's compile and warm-up under a
                                # second. The CPU gains nothing from it and runs the plain function.
                                layer_fn = function(x) {
    if (x$device$type != "cuda" || !isTRUE(getOption("n3d.jit", TRUE))) {
        return(self$layer_forward)
    }
    key <- paste(x$device$index, x$dtype$.type())
    if (is.null(self$traces[[key]])) {
        self$traces[[key]] <- trace_layer_forward(self, x$device, x$dtype)
    }
    self$traces[[key]]
}
)

# Traces layer_forward() with the first layer's weights and runs it at
# three lengths. TorchScript's profiling executor specializes on static
# shapes twice, then compiles a dynamic-shape graph; after these runs any
# length is served by that graph, so no real chunk pays for compilation.
trace_layer_forward <- function(tower, device, dtype) {
    weights <- tower$layers[[1]]$weights()
    hidden <- tower$num_heads * tower$head_dim
    example <- function(l) {
        rope <- rope_cos_sin(l, tower$head_dim, tower$rope_theta,
                             device = device, dtype = dtype)
        h <- torch::torch_zeros(1L, l, hidden, dtype = dtype, device = device)
        mask <- torch::torch_zeros(1L, 1L, 1L, l, dtype = dtype,
                                   device = device)
        c(list(h, rope$cos, rope$sin, mask), weights)
    }
    torch::with_no_grad({
        args <- c(list(tower$layer_forward), example(16L))
        traced <- do.call(torch::jit_trace, args)
        for (l in c(16L, 16L, 17L, 17L, 18L, 18L, 18L)) {
            do.call(traced, example(l))
        }
    })
    traced
}
