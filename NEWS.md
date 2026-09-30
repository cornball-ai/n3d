# n3d 0.0.1.1

- On CUDA, encoder layers run as one traced TorchScript function shared by
  all 31 layers, and each chunk ends with a minor garbage collection. Peak
  device memory drops from 1.7-2.1 GiB to 0.47-0.55 GiB (weights are
  0.37 GiB), and a streaming step from 55 ms to about 18 ms on an RTX 5060
  Ti. `load_n3d()` traces and warms up the layer once (about 1 s).
  `options(n3d.jit = FALSE)` runs the plain R functions; the CPU always
  does.
- Log-mel features are computed on the CPU and moved to the device, which
  keeps a whole-file STFT off the GPU.

# n3d 0.0.1

- Initial version: native R 'torch' port of NVIDIA Nemotron 3 Diarization
  with offline diarization (`diarize()`), streaming sessions
  (`n3d_stream()`, `n3d_stream_push()`, `n3d_stream_finish()`), log-mel
  features and the arrival-order speaker cache implemented in R, and
  pinned-revision weight download through 'hfhub'.
