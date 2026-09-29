if (!torch::torch_is_installed()) exit_file("torch backend not installed")

log_mel <- n3d:::log_mel
set.seed(1)
x <- stats::rnorm(16000) * 0.1

f <- log_mel(x, center = TRUE)
expect_equal(f$features$shape, c(1L, 101L, 128L))
expect_equal(f$num_valid, 100L)
# the frame past the valid length is zeroed
expect_equal(f$features[1, 101, ]$abs()$sum()$item(), 0)
expect_true(f$features[1, 100, ]$abs()$sum()$item() > 0)

g <- log_mel(x, center = FALSE)
expect_equal(g$num_valid, (16000L - 512L) %/% 160L + 1L)
expect_equal(g$features$shape[2], g$num_valid)

expect_error(log_mel(numeric(100), center = FALSE), "too short")
