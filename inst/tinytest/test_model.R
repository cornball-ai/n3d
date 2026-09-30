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

# On CUDA the encoder layers run as one traced graph; it must match the
# plain R functions at lengths other than the ones it was traced and warmed
# up at, with and without a key mask.
if (torch::cuda_is_available()) {
    gpu <- n3d::load_n3d(device = "cuda")
    tower <- gpu$net$model$audio_tower
    expect_false(identical(tower$layer_fn(torch::torch_zeros(1, device = "cuda")),
                           tower$layer_forward))
    torch::with_no_grad({
        x <- torch::torch_randn(1, 203, 512, device = "cuda")
        valid <- torch::torch_arange(1, 203, device = "cuda")$unsqueeze(1) <= 190
        for (v in list(NULL, valid)) {
            traced <- gpu$net$step(x, v)
            local({
                old <- options(n3d.jit = FALSE)
                on.exit(options(old), add = TRUE)
                plain <- gpu$net$step(x, v)
                # relative: fused kernels sum in a different order, and
                # random inputs give logits in the tens
                rel <- (traced - plain)$abs()$max()$item() /
                    plain$abs()$max()$item()
                expect_true(rel < 1e-4, info = paste("mask:", !is.null(v)))
            })
        }
    })
}
