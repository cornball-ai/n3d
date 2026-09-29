# Model modules. Parameter names mirror the checkpoint's keys, e.g.
# model.audio_tower.layers.0.self_attn.q_proj.weight.

#' Sub-pixel Upsampler
#'
#' A kernel-3 convolution to \code{factor} times the channels, reshaped back
#' to one frame per mel frame.
#' @noRd
subpixel_upsampler <- torch::nn_module(
                                       "SubpixelUpsampler",
                                       initialize = function(hidden_size, factor) {
    self$factor <- factor
    self$conv <- torch::nn_conv1d(hidden_size, hidden_size * factor,
                                  kernel_size = 3L, padding = 1L)
},
                                       forward = function(x_BLD) {
    b <- x_BLD$shape[1]
    l <- x_BLD$shape[2]
    d <- x_BLD$shape[3]
    y <- self$conv(x_BLD$transpose(2L, 3L))$transpose(2L, 3L)
    y$reshape(c(b, l * self$factor, d))
}
)

classification_head <- torch::nn_module(
                                        "ClassificationHead",
                                        initialize = function(hidden_size, num_speakers) {
    self$dense <- torch::nn_linear(hidden_size, hidden_size)
    self$out_proj <- torch::nn_linear(hidden_size, num_speakers)
},
                                        forward = function(x) {
    self$out_proj(torch::nnf_relu(self$dense(torch::nnf_relu(x))))
}
)

encoder_model <- torch::nn_module(
                                  "EncoderModel",
                                  initialize = function(cfg) {
    self$audio_tower <- audio_tower(cfg)
    self$proj <- torch::nn_linear(cfg$hidden_size, cfg$head_hidden_size)
    self$upsampler <- subpixel_upsampler(cfg$head_hidden_size,
        cfg$subsampling_factor)
},
                                  forward = function(embeds_BLD, valid_BL = NULL) {
    self$upsampler(self$proj(self$audio_tower(embeds_BLD, valid_BL)))
}
)

#' Nemotron 3 Diarization Network
#' @noRd
n3d_net <- torch::nn_module(
                            "Nemotron3Diarization",
                            # cfg has no default: nn_module evaluates defaults outside the namespace
                            initialize = function(cfg) {
    self$cfg <- cfg
    self$model <- encoder_model(cfg)
    self$classifier <- classification_head(cfg$head_hidden_size,
        cfg$num_speakers)
    self$silence_embeds <- torch::nn_parameter(
        torch::torch_zeros(cfg$hidden_size)
    )
},
                            # One encoder step: logits at the mel frame rate for the given embeddings.
                            step = function(embeds_BLD, valid_BL = NULL) {
    self$classifier(self$model(embeds_BLD, valid_BL))
}
)

#' Load the Diarization Model
#'
#' Builds the network and loads the Nemotron 3 Diarization weights from the
#' Hugging Face cache.
#'
#' @param device Torch device, \code{"cuda"} when available, otherwise
#'   \code{"cpu"}.
#' @param download Whether to download the weights when they are not cached.
#'   Defaults to \code{FALSE}; see \code{\link{download_n3d}}.
#' @param revision Hugging Face commit to load. Defaults to the commit this
#'   package was validated against.
#' @return An \code{n3d_model} object: a list holding the network and its
#'   configuration.
#' @examples
#' \donttest{
#' if (n3d_exists()) {
#'   model <- load_n3d(device = "cpu")
#' }
#' }
#' @export
load_n3d <- function(device = default_device(), download = FALSE,
                     revision = n3d_revision()) {
    if (!n3d_exists(revision)) {
        if (!download) {
            stop("Nemotron 3 Diarization weights are not in the Hugging Face ",
                 "cache. Run download_n3d() first, or pass download = TRUE.",
                 call. = FALSE)
        }
        download_n3d(revision = revision)
    }
    path <- hfhub::hub_download(.n3d_repo, "model.safetensors",
                                revision = revision, local_files_only = TRUE)

    torch::with_no_grad({
        net <- n3d_net(n3d_config())
        weights <- safetensors::safe_load_file(path, framework = "torch")
        load_weights(net, weights)
        rm(weights)
        net$eval()
        net$to(device = device)
    })
    structure(list(net = net, cfg = net$cfg, device = device),
              class = "n3d_model")
}

# Copies every checkpoint tensor into the matching parameter; any key left
# over on either side is an error.
load_weights <- function(net, weights) {
    params <- net$state_dict()
    missing <- setdiff(names(params), names(weights))
    unused <- setdiff(names(weights), names(params))
    if (length(missing) || length(unused)) {
        stop("checkpoint does not match the network. Missing: ",
             paste(utils::head(missing, 5L), collapse = ", "), "; unused: ",
             paste(utils::head(unused, 5L), collapse = ", "), call. = FALSE)
    }
    torch::with_no_grad({
        for (name in names(params)) {
            params[[name]]$copy_(weights[[name]])
        }
    })
    invisible(net)
}

default_device <- function() {
    if (torch::cuda_is_available()) "cuda" else "cpu"
}

#' @exportS3Method print n3d_model
print.n3d_model <- function(x, ...) {
    cat("Nemotron 3 Diarization (", x$cfg$num_speakers, " speakers, ",
        x$cfg$num_layers, " layers) on ", x$device, "\n", sep = "")
    invisible(x)
}
