#' Model Configuration
#'
#' The released checkpoint's architecture and speaker-cache settings, taken
#' from its config.json and processor_config.json.
#'
#' @return A named list.
#' @noRd
n3d_config <- function() {
    list(
         sample_rate = 16000L,
         n_fft = 512L,
         win_length = 400L,
         hop_length = 160L,
         n_mels = 128L,
         preemphasis = 0.97,
         hidden_size = 512L,
         intermediate_size = 2048L,
         num_layers = 31L,
         num_heads = 8L,
         rope_theta = 10000,
         subsampling_factor = 8L,
         head_hidden_size = 192L,
         num_speakers = 8L,
         # offline chunking, in encoder frames (80 ms each)
         chunk_length = 340L,
         chunk_right_context = 40L,
         fifo_length = 40L,
         speaker_cache_update_period = 300L,
         # speaker cache policy; FIFO sizes here are the streaming ones
         streaming = list(speaker_cache_length = 264L, fifo_length = 264L,
                          speaker_cache_update_period = 222L,
                          silence_frames_per_speaker = 1L,
                          prediction_score_threshold = 0.25,
                          latest_frames_score_boost = 0.05,
                          min_positive_scores_rate = 0.5,
                          strong_boost_rate = 0.75, weak_boost_rate = 1.5)
    )
}

# Streaming modes: (chunk_length, chunk_right_context) in encoder frames.
.streaming_modes <- list(low_latency = c(9L, 4L),
                         very_low_latency = c(6L, 2L),
                         ultra_low_latency = c(3L, 1L))

streaming_mode_sizes <- function(mode) {
    if (!is.character(mode) || length(mode) != 1L ||
        !mode %in% names(.streaming_modes)) {
        stop("mode must be one of: ",
             paste(sprintf("\"%s\"", names(.streaming_modes)), collapse = ", "),
             call. = FALSE)
    }
    .streaming_modes[[mode]]
}
