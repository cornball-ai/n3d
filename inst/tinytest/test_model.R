# Needs the ~400 MB weights in the Hugging Face cache; never downloads.
if (!at_home()) exit_file("model tests run at home only")
if (!torch::torch_is_installed()) exit_file("torch backend not installed")
if (!n3d::n3d_exists()) exit_file("weights not cached; run download_n3d()")

model <- n3d::load_n3d(device = "cpu")
mp3 <- system.file("samples", "Synapsis-Wonderland.mp3", package = "av")
audio <- n3d::load_audio(mp3)[seq_len(16000 * 12)]

res <- n3d::diarize(audio, model, probs = TRUE)
expect_equal(dim(res$probs), c(1201L, 8L))
expect_true(all(res$probs >= 0 & res$probs <= 1))
expect_equal(names(res$segments), c("start", "end", "speaker"))
expect_true(all(res$segments$end > res$segments$start))

# a stream's output does not depend on how the audio is split into pushes
run_stream <- function(piece_size) {
  s <- n3d::n3d_stream(model, "very_low_latency")
  out <- list()
  for (start in seq(1L, length(audio), by = piece_size)) {
    end <- min(start + piece_size - 1L, length(audio))
    out[[length(out) + 1L]] <- n3d::n3d_stream_push(s, audio[start:end])
  }
  out[[length(out) + 1L]] <- n3d::n3d_stream_finish(s)
  do.call(rbind, out)
}
whole <- run_stream(length(audio))
pieces <- run_stream(1234L)
# the last chunk's windows are uncentered, so frames whose window runs past
# the end of the audio are not produced
expect_equal(dim(whole), c((length(audio) - 256L) %/% 160L + 1L, 8L))
expect_equal(pieces, whole, tolerance = 1e-5)

s <- n3d::n3d_stream(model)
expect_equal(s$latency_ms, 1040)
n3d::n3d_stream_finish(s)
expect_error(n3d::n3d_stream_push(s, numeric(10)), "finished")
expect_error(n3d::n3d_stream(model, "fast"), "mode must be one of")
