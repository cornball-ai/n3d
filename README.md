# n3d

Speaker diarization ("who spoke when") in R, with a native R 'torch' port of
NVIDIA's [Nemotron 3 Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization).
The model is a Streaming Sortformer that tracks up to 8 speakers, numbered in
order of first arrival, and runs offline or on streaming audio with 0.32 s to
1.04 s of input latency. No Python is needed.

## Install

```r
remotes::install_github("cornball-ai/n3d")
```

n3d needs a working 'torch' backend (`torch::install_torch()`).

## Use

```r
library(n3d)
download_n3d()          # ~400 MB into the Hugging Face cache, once
model <- load_n3d()     # "cuda" when available, otherwise "cpu"

diarize("meeting.wav", model)
#>   start   end speaker
#> 1  0.35  9.31       1
#> 2  9.15 13.81       2
#> ...
```

`diarize(..., probs = TRUE)` also returns the per-frame speaker
probabilities (one row per 10 ms). Any format FFmpeg reads works, through the
'av' package; numeric vectors of 16 kHz mono samples work too.

### Streaming

```r
s <- n3d_stream(model, mode = "low_latency")   # 1.04 s input latency
p <- n3d_stream_push(s, samples)               # probabilities for completed chunks
p <- rbind(p, n3d_stream_finish(s))
probs_to_segments(p)
```

Modes are `"low_latency"` (1.04 s), `"very_low_latency"` (0.64 s) and
`"ultra_low_latency"` (0.32 s). Push any number of samples at a time; the
output does not depend on how the audio is split.

Every streaming step re-encodes the speaker cache and FIFO (about 540
encoder frames) to score one chunk, so real-time streaming needs a GPU. On
an RTX 5060 Ti a `"low_latency"` step takes about 55 ms per 0.72 s chunk;
on a 20-thread CPU it takes about 5.5 s. Offline diarization is cheaper:
30 s of audio takes 0.3 s on that GPU and 11 s on that CPU.

## Accuracy

The port was validated against the Hugging Face transformers implementation
on a 97.7 s, 6-speaker recording: offline logits agree to within 2e-5
(float32), `"low_latency"` streaming probabilities to within 2e-6, and the
resulting segments are identical in both modes.

## License

Apache License (>= 2). The architecture code is ported from the
transformers implementation (Apache-2.0); see `inst/COPYRIGHTS`. The model
weights are published by NVIDIA under the OpenMDW License 1.1 and are
downloaded separately.
