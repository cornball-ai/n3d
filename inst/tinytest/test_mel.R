mel_filterbank <- n3d:::mel_filterbank
hz_to_mel <- n3d:::hz_to_mel
mel_to_hz <- n3d:::mel_to_hz

hz <- c(0, 100, 999, 1000, 1001, 4000, 8000)
expect_equal(mel_to_hz(hz_to_mel(hz)), hz)
expect_equal(hz_to_mel(1000), 15)

fb <- mel_filterbank(16000L, 512L, 128L)
expect_equal(dim(fb), c(128L, 257L))
expect_true(all(fb >= 0))
expect_true(all(rowSums(fb) > 0))
