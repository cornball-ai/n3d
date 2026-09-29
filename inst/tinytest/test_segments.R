p <- matrix(0, 300, 8)
p[1:120, 1] <- 0.9
p[100:300, 2] <- 0.8
p[200:210, 1] <- 0.7
seg <- n3d::probs_to_segments(p)

expect_equal(nrow(seg), 3L)
expect_equal(seg$start, c(0, 0.99, 1.99))
expect_equal(seg$end, c(1.2, 3, 2.1))
expect_equal(seg$speaker, c(1L, 2L, 1L))

# threshold is strict
expect_equal(nrow(n3d::probs_to_segments(p, threshold = 0.9)), 0L)

empty <- n3d::probs_to_segments(matrix(0, 10, 8))
expect_equal(names(empty), c("start", "end", "speaker"))
expect_equal(nrow(empty), 0L)

expect_error(n3d::probs_to_segments(1:10), "matrix")
