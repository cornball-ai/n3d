#' Slaney Mel Filterbank
#'
#' Matches \code{librosa.filters.mel(sr, n_fft, n_mels, fmin = 0,
#' fmax = sr / 2, norm = "slaney")}: Slaney mel scale, triangular filters,
#' area normalization. Computed in double precision like librosa.
#'
#' @param sample_rate Sampling rate in Hz.
#' @param n_fft FFT size.
#' @param n_mels Number of mel bands.
#' @return Numeric matrix, \code{n_mels} x \code{n_fft / 2 + 1}.
#' @noRd
mel_filterbank <- function(sample_rate = 16000L, n_fft = 512L, n_mels = 128L) {
  fft_freqs <- seq(0, sample_rate / 2, length.out = n_fft %/% 2L + 1L)
  mel_max <- hz_to_mel(sample_rate / 2)
  mel_f <- mel_to_hz(seq(0, mel_max, length.out = n_mels + 2L))

  fdiff <- diff(mel_f)
  ramps <- outer(mel_f, fft_freqs, "-")
  weights <- matrix(0, n_mels, length(fft_freqs))
  for (i in seq_len(n_mels)) {
    lower <- -ramps[i, ] / fdiff[i]
    upper <- ramps[i + 2L, ] / fdiff[i + 1L]
    weights[i, ] <- pmax(0, pmin(lower, upper))
  }
  enorm <- 2 / (mel_f[3:(n_mels + 2L)] - mel_f[seq_len(n_mels)])
  weights * enorm
}

# Slaney mel scale: linear below 1 kHz, logarithmic above.
hz_to_mel <- function(hz) {
  f_sp <- 200 / 3
  min_log_hz <- 1000
  min_log_mel <- min_log_hz / f_sp
  logstep <- log(6.4) / 27
  ifelse(hz >= min_log_hz, min_log_mel + log(hz / min_log_hz) / logstep,
         hz / f_sp)
}

mel_to_hz <- function(mel) {
  f_sp <- 200 / 3
  min_log_hz <- 1000
  min_log_mel <- min_log_hz / f_sp
  logstep <- log(6.4) / 27
  ifelse(mel >= min_log_mel, min_log_hz * exp(logstep * (mel - min_log_mel)),
         mel * f_sp)
}
