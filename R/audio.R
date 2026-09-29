#' Load Audio
#'
#' Decodes an audio or video file with \pkg{av}, downmixed to mono and
#' resampled to 16 kHz.
#'
#' @param file Path to an audio or video file in any format FFmpeg reads.
#' @return Numeric vector of samples in \code{[-1, 1]}.
#' @examples
#' mp3 <- system.file("samples", "Synapsis-Wonderland.mp3", package = "av")
#' samples <- load_audio(mp3)
#' length(samples) / 16000 # seconds
#' @export
load_audio <- function(file) {
    if (!is.character(file) || length(file) != 1L || !file.exists(file)) {
        stop("audio file not found: ", file, call. = FALSE)
    }
    # FFmpeg's resampler prints "Insufficient memory to recode all samples"
    # on benign input; keep it off the console
    pcm <- NULL
    utils::capture.output(
                          pcm <- av::read_audio_bin(file, channels = 1L, sample_rate = 16000L),
                          type = "message"
    )
    as.numeric(pcm) / 2147483648
}

# Accepts a file path or a numeric sample vector and returns samples.
as_samples <- function(audio, sample_rate) {
    if (is.character(audio)) {
        return(load_audio(audio))
    }
    if (!is.numeric(audio) || length(audio) == 0L) {
        stop("audio must be a file path or a non-empty numeric vector",
             call. = FALSE)
    }
    if (sample_rate != 16000L) {
        stop("numeric audio must be sampled at 16000 Hz; got ", sample_rate,
             ". Resample it first, or pass a file path.", call. = FALSE)
    }
    as.numeric(audio)
}
