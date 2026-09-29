#' Log-Mel Features
#'
#' Pre-emphasis, STFT with a symmetric 400-sample Hann window zero-padded to
#' 512, power spectrum, Slaney mel projection and natural log with a
#' \code{2^-24} guard. The features are not normalized. Frames past the
#' valid length are zeroed.
#'
#' With \code{center = TRUE} the signal is zero-padded by \code{n_fft / 2} on
#' both sides (offline use and the first streaming chunk). Later streaming
#' chunks use \code{center = FALSE} and start \code{n_fft / 2} samples before
#' their first frame, which reproduces the frames of a single centered pass.
#'
#' @param audio Numeric vector of 16 kHz mono samples.
#' @param center Whether to center the analysis windows.
#' @param device Torch device for the computation.
#' @return A list with \code{features}, a float tensor of shape
#'   \code{(1, frames, 128)}, and \code{num_valid}, the number of valid
#'   frames.
#' @noRd
log_mel <- function(audio, center = TRUE, device = "cpu") {
  cfg <- n3d_config()
  n <- length(audio)
  num_valid <- if (center) {
    n %/% cfg$hop_length
  } else {
    (n - cfg$n_fft) %/% cfg$hop_length + 1L
  }
  if (num_valid < 1L) {
    stop("audio is too short for one feature frame (", n, " samples)",
         call. = FALSE)
  }

  x <- torch::torch_tensor(audio, dtype = torch::torch_float32(),
                           device = device)
  if (n > 1L) {
    x <- torch::torch_cat(list(x[1:1],
                               x[2:n] - cfg$preemphasis * x[1:(n - 1L)]))
  }

  window <- torch::torch_hann_window(cfg$win_length, periodic = FALSE,
                                     device = device)
  spec <- torch::torch_stft(x$unsqueeze(1L), n_fft = cfg$n_fft,
                            hop_length = cfg$hop_length,
                            win_length = cfg$win_length, window = window,
                            center = center, pad_mode = "constant",
                            return_complex = TRUE)
  # magnitude then square, in the reference's order of rounding
  power <- torch::torch_view_as_real(spec)$pow(2)$sum(-1L)$sqrt()$pow(2)

  filters <- mel_filter_tensor(device)
  mel_BMT <- torch::torch_matmul(filters, power)
  mel_BTM <- torch::torch_log(mel_BMT + 2^-24)$permute(c(1L, 3L, 2L))

  num_frames <- mel_BTM$shape[2]
  if (num_valid < num_frames) {
    mask <- torch::torch_arange(1, num_frames, device = device) <= num_valid
    mel_BTM <- mel_BTM * mask$unsqueeze(-1L)$to(dtype = mel_BTM$dtype)
  }
  list(features = mel_BTM, num_valid = min(num_valid, num_frames))
}

# The filterbank as a float32 tensor, rounded through float32 once like
# librosa's default dtype.
mel_filter_tensor <- function(device = "cpu") {
  cfg <- n3d_config()
  fb <- mel_filterbank(cfg$sample_rate, cfg$n_fft, cfg$n_mels)
  torch::torch_tensor(fb, dtype = torch::torch_float32(), device = device)
}
