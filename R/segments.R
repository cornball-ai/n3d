#' Speaker Probabilities to Segments
#'
#' Thresholds per-frame speaker probabilities and returns one row per
#' contiguous run of activity. Overlapping speech gives overlapping segments.
#'
#' @param probs Numeric matrix, frames x speakers, of speaker activity
#'   probabilities (one frame per 10 ms), as returned by
#'   \code{\link{diarize}} with \code{probs = TRUE} or by
#'   \code{\link{n3d_stream_push}}.
#' @param threshold Probability above which a frame counts as speech.
#' @param frame_duration Seconds per frame.
#' @return A data frame with columns \code{start} and \code{end} (seconds)
#'   and \code{speaker} (integer, numbered from 1 in order of first
#'   arrival), sorted by start time then speaker.
#' @examples
#' p <- matrix(0, 300, 8)
#' p[1:120, 1] <- 0.9
#' p[100:300, 2] <- 0.8
#' probs_to_segments(p)
#' @export
probs_to_segments <- function(probs, threshold = 0.5, frame_duration = 0.01) {
  if (!is.matrix(probs)) {
    stop("probs must be a frames x speakers matrix", call. = FALSE)
  }
  rows <- list()
  for (spk in seq_len(ncol(probs))) {
    active <- as.integer(probs[, spk] > threshold)
    changes <- diff(c(0L, active, 0L))
    starts <- which(changes == 1L) - 1L
    ends <- which(changes == -1L) - 1L
    if (length(starts)) {
      rows[[length(rows) + 1L]] <- data.frame(
        start = round(starts * frame_duration, 2),
        end = round(ends * frame_duration, 2),
        speaker = spk
      )
    }
  }
  if (!length(rows)) {
    return(data.frame(start = numeric(0), end = numeric(0),
                      speaker = integer(0)))
  }
  out <- do.call(rbind, rows)
  out <- out[order(out$start, out$speaker), , drop = FALSE]
  rownames(out) <- NULL
  out
}
