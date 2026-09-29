if (!torch::torch_is_installed()) exit_file("torch backend not installed")

ns <- asNamespace("n3d")
cfg <- ns$n3d_config()

cache <- ns$new_speaker_cache(cfg)
expect_equal(cache$fifo_length, 264L)
expect_equal(cache$min_positive, 16)
expect_equal(cache$num_strong, 24L)
expect_equal(cache$num_weak, 48L)

# nothing moves until the FIFO overflows, then at least the update period
expect_equal(ns$num_popped_frames(cache, 264L), 0L)
expect_equal(ns$num_popped_frames(cache, 265L), 222L)
expect_equal(ns$num_popped_frames(cache, 600L), 336L)

# compression keeps speaker_cache_length slots, grouped by speaker in
# arrival order, with one silence slot per speaker; silent speakers score
# -Inf everywhere, so the active speakers fill the rest
torch::torch_manual_seed(1)
n <- 400L
d <- 4L
embeds <- torch::torch_arange(1, n)$view(c(1L, n, 1L))$expand(c(1L, n, d))
probs <- torch::torch_zeros(1L, n, 8L)
probs[1, 1:200, 1] <- 0.9
probs[1, 201:400, 2] <- 0.9
silence <- torch::torch_full(d, -1)
out <- ns$compress_cache(cache, embeds, probs, silence)
expect_equal(out$embeds$shape, c(1L, 264L, d))
expect_equal(out$probs$shape, c(1L, 264L, 8L))
frame_ids <- as.numeric(out$embeds[1, , 1])
expect_equal(sum(frame_ids == -1), 8)
kept <- frame_ids[frame_ids > 0]
expect_equal(length(kept), 256L)
# speaker 1's frames first, then speaker 2's, each in arrival order
first_spk2 <- which(kept > 200)[1]
expect_true(all(kept[seq_len(first_spk2 - 1L)] <= 200))
expect_false(is.unsorted(kept[seq_len(first_spk2 - 1L)]))
expect_false(is.unsorted(kept[first_spk2:length(kept)]))
# the latest-frame boost keeps every frame popped past the capacity
expect_true(all(265:400 %in% kept))
