#' Speaker Diarization
#'
#' Finds who spoke when in a recording. Speakers are numbered from 1 in
#' order of first arrival; up to eight are tracked.
#'
#' The recording is processed offline in 27.2 s chunks with 3.2 s of
#' look-ahead, carrying the arrival-order speaker cache between chunks, so
#' its length is not limited.
#'
#' @param audio Path to an audio or video file, or a numeric vector of 16 kHz
#'   mono samples.
#' @param model A model from \code{\link{load_n3d}}. Loaded on first use
#'   when \code{NULL}.
#' @param threshold Speaker probability above which a frame counts as
#'   speech.
#' @param probs Whether to also return the per-frame probabilities.
#' @param sample_rate Sampling rate of numeric \code{audio}; must be 16000.
#' @return A data frame of segments with columns \code{start}, \code{end}
#'   (seconds) and \code{speaker}. With \code{probs = TRUE}, a list of
#'   \code{segments} and \code{probs}, a frames x 8 matrix with one frame per
#'   10 ms.
#' @examples
#' \donttest{
#' if (n3d_exists()) {
#'   model <- load_n3d()
#'   mp3 <- system.file("samples", "Synapsis-Wonderland.mp3", package = "av")
#'   diarize(mp3, model)
#' }
#' }
#' @export
diarize <- function(audio, model = NULL, threshold = 0.5, probs = FALSE,
                    sample_rate = 16000L) {
    samples <- as_samples(audio, sample_rate)
    model <- model %||% cached_model()
    p <- speaker_probs(model, samples)
    segments <- probs_to_segments(p, threshold)
    if (probs) list(segments = segments, probs = p) else segments
}

# Offline per-frame probabilities for one recording, frames x speakers.
speaker_probs <- function(model, samples) {
    torch::with_no_grad({
        feats <- log_mel(samples, center = TRUE, device = model$device)
        num_frames <- feats$features$shape[2]
        valid <- (torch::torch_arange(1, num_frames, device = model$device) <=
                        feats$num_valid)$unsqueeze(1L)
        out <- chunked_forward(model$net, feats$features, valid)
        p <- out$logits$sigmoid()
        p <- p * valid$unsqueeze(-1L)$to(dtype = p$dtype)
        as.matrix(p[1,,]$cpu())
    })
}

.n3d_env <- new.env(parent = emptyenv())

# Loads the default model once per session.
cached_model <- function() {
    if (is.null(.n3d_env$model)) {
        .n3d_env$model <- load_n3d()
    }
    .n3d_env$model
}
