# n3d 0.0.1

- Initial version: native R 'torch' port of NVIDIA Nemotron 3 Diarization
  with offline diarization (`diarize()`), streaming sessions
  (`n3d_stream()`, `n3d_stream_push()`, `n3d_stream_finish()`), log-mel
  features and the arrival-order speaker cache implemented in R, and
  pinned-revision weight download through 'hfhub'.
