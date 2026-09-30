#' Streaming Diarization
#'
#' Opens a streaming session: push audio as it arrives and get speaker
#' probabilities for each completed chunk. The speaker cache keeps speaker
#' numbering consistent across the whole stream.
#'
#' Latency is the audio a chunk waits for, its frames plus look-ahead:
#' 1.04 s for \code{"low_latency"}, 0.64 s for \code{"very_low_latency"} and
#' 0.32 s for \code{"ultra_low_latency"}, excluding compute time.
#'
#' Every step re-encodes the speaker cache and FIFO queue (about 540
#' encoder frames) to score one chunk, so keeping up with live audio needs
#' a GPU; on a CPU a step takes seconds.
#'
#' @param model A model from \code{\link{load_n3d}}.
#' @param mode Streaming mode: \code{"low_latency"},
#'   \code{"very_low_latency"} or \code{"ultra_low_latency"}.
#' @return An \code{n3d_stream} session.
#' @seealso \code{\link{n3d_stream_push}}, \code{\link{n3d_stream_finish}}
#' @examples
#' \donttest{
#' if (n3d_exists()) {
#'   model <- load_n3d()
#'   mp3 <- system.file("samples", "Synapsis-Wonderland.mp3", package = "av")
#'   samples <- load_audio(mp3)
#'   s <- n3d_stream(model)
#'   probs <- list()
#'   for (start in seq(1, length(samples), by = 8000)) {
#'     piece <- samples[start:min(start + 7999, length(samples))]
#'     probs[[length(probs) + 1]] <- n3d_stream_push(s, piece)
#'   }
#'   probs[[length(probs) + 1]] <- n3d_stream_finish(s)
#'   probs_to_segments(do.call(rbind, probs))
#' }
#' }
#' @export
n3d_stream <- function(model, mode = "low_latency") {
    sizes <- streaming_mode_sizes(mode)
    cfg <- model$cfg
    s <- new.env(parent = emptyenv())
    s$model <- model
    s$mode <- mode
    s$chunk_length <- sizes[1]
    s$right_context <- sizes[2]
    s$frames_per_chunk <- (sizes[1] + sizes[2]) * cfg$subsampling_factor
    s$frames_per_step <- sizes[1] * cfg$subsampling_factor
    s$samples_first <- (s$frames_per_chunk - 1L) * cfg$hop_length +
    cfg$win_length %/% 2L
    s$samples_later <- s$frames_per_chunk * cfg$hop_length + cfg$win_length
    s$latency_ms <- round(sum(sizes) * cfg$subsampling_factor *
                          cfg$hop_length / cfg$sample_rate * 1000)
    s$buffer <- numeric(0)
    s$buffer_start <- 0 # absolute index (0-based) of buffer[1]
    s$received <- 0
    s$start_frame <- 0L
    s$first <- TRUE
    s$cache <- NULL
    s$finished <- FALSE
    class(s) <- "n3d_stream"
    s
}

#' Push Audio to a Stream
#'
#' @param stream A session from \code{\link{n3d_stream}}.
#' @param samples Numeric vector of 16 kHz mono samples.
#' @return A matrix of speaker probabilities, frames x 8, for the frames
#'   completed by this push (possibly zero rows), one frame per 10 ms.
#' @seealso \code{\link{n3d_stream}} for a complete example.
#' @examples
#' \donttest{
#' if (n3d_exists()) {
#'   s <- n3d_stream(load_n3d())
#'   p <- n3d_stream_push(s, numeric(32000)) # 2 s of silence
#'   dim(p)
#' }
#' }
#' @export
n3d_stream_push <- function(stream, samples) {
    check_stream(stream)
    stream$buffer <- c(stream$buffer, as.numeric(samples))
    stream$received <- stream$received + length(samples)
    out <- list(empty_probs(stream))
    repeat {
        span <- next_chunk_span(stream)
        # a chunk reaching the end of the received audio could be the last one;
        # wait for more audio or n3d_stream_finish() to decide
        if (span[2] >= stream$received) break
        out[[length(out) + 1L]] <- stream_step(stream, span, is_last = FALSE)
    }
    do.call(rbind, out)
}

#' Finish a Stream
#'
#' Scores the remaining buffered audio, including the look-ahead frames of
#' the previous chunk, and closes the session.
#'
#' @param stream A session from \code{\link{n3d_stream}}.
#' @return A matrix of speaker probabilities for the remaining frames.
#' @seealso \code{\link{n3d_stream}} for a complete example.
#' @examples
#' \donttest{
#' if (n3d_exists()) {
#'   s <- n3d_stream(load_n3d())
#'   n3d_stream_push(s, numeric(32000))
#'   dim(n3d_stream_finish(s))
#' }
#' }
#' @export
n3d_stream_finish <- function(stream) {
    check_stream(stream)
    if (stream$received == 0) {
        stream$finished <- TRUE
        return(empty_probs(stream))
    }
    span <- next_chunk_span(stream)
    out <- stream_step(stream, c(span[1], stream$received), is_last = TRUE)
    stream$finished <- TRUE
    out
}

#' @exportS3Method print n3d_stream
print.n3d_stream <- function(x, ...) {
    cat("n3d stream (", x$mode, ", ", x$latency_ms, " ms input latency): ",
        round(x$received / 16000, 2), " s received, ",
        round(x$start_frame / 100, 2), " s scored",
        if (x$finished) ", finished", "\n", sep = "")
    invisible(x)
}

# Absolute (0-based, end-exclusive) sample span of the next chunk.
next_chunk_span <- function(stream) {
    cfg <- stream$model$cfg
    if (stream$first) {
        return(c(0, stream$samples_first))
    }
    start <- stream$start_frame * cfg$hop_length - cfg$n_fft %/% 2L
    c(start, start + stream$samples_later)
}

stream_step <- function(stream, span, is_last) {
    model <- stream$model
    from <- span[1] - stream$buffer_start
    to <- min(span[2], stream$received) - stream$buffer_start
    piece <- stream$buffer[(from + 1):to]

    probs <- torch::with_no_grad({
        feats <- log_mel(piece, center = stream$first, device = "cpu")
        num_frames <- feats$num_valid
        features <- feats$features$narrow(2L, 1L, num_frames)$to(
            device = model$device)
        if (!is_last && num_frames != stream$frames_per_chunk) {
            stop("internal error: chunk holds ", num_frames, " mel frames, ",
                 "expected ", stream$frames_per_chunk, call. = FALSE)
        }
        lookahead <- if (is_last) NULL else stream$right_context
        out <- chunked_forward(model$net, features, cache = stream$cache,
                               num_lookahead = lookahead)
        stream$cache <- out$cache
        as.matrix(out$logits$sigmoid()[1,,, drop = FALSE]$squeeze(1L)$cpu())
    })

    stream$start_frame <- stream$start_frame + stream$frames_per_step
    stream$first <- FALSE
    # drop samples no later chunk needs
    keep_from <- next_chunk_span(stream)[1]
    drop <- max(0, keep_from - stream$buffer_start)
    if (drop > 0) {
        stream$buffer <- stream$buffer[-seq_len(min(drop,
                    length(stream$buffer)))]
        stream$buffer_start <- stream$buffer_start + drop
    }
    probs
}

empty_probs <- function(stream) {
    matrix(numeric(0), 0L, stream$model$cfg$num_speakers)
}

check_stream <- function(stream) {
    if (!inherits(stream, "n3d_stream")) {
        stop("stream must come from n3d_stream()", call. = FALSE)
    }
    if (stream$finished) {
        stop("stream is finished; open a new one with n3d_stream()",
             call. = FALSE)
    }
}
