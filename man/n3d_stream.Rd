% tinyrox says don't edit this manually, but it can't stop you!
\name{n3d_stream}
\alias{n3d_stream}
\title{Streaming Diarization}
\usage{
n3d_stream(model, mode = "low_latency")
}
\arguments{
\item{model}{A model from \code{\link{load_n3d}}.}

\item{mode}{Streaming mode: \code{"low_latency"},
\code{"very_low_latency"} or \code{"ultra_low_latency"}.}
}
\value{
An \code{n3d_stream} session.
}
\description{
Opens a streaming session: push audio as it arrives and get speaker
probabilities for each completed chunk. The speaker cache keeps speaker
numbering consistent across the whole stream.
}
\details{
Latency is the audio a chunk waits for, its frames plus look-ahead:
1.04 s for \code{"low_latency"}, 0.64 s for \code{"very_low_latency"} and
0.32 s for \code{"ultra_low_latency"}, excluding compute time.

Every step re-encodes the speaker cache and FIFO queue (about 540
encoder frames) to score one chunk, so keeping up with live audio needs
a GPU; on a CPU a step takes seconds.

}
\examples{
\donttest{
if (n3d_exists()) {
  model <- load_n3d()
  mp3 <- system.file("samples", "Synapsis-Wonderland.mp3", package = "av")
  samples <- load_audio(mp3)
  s <- n3d_stream(model)
  probs <- list()
  for (start in seq(1, length(samples), by = 8000)) {
    piece <- samples[start:min(start + 7999, length(samples))]
    probs[[length(probs) + 1]] <- n3d_stream_push(s, piece)
  }
  probs[[length(probs) + 1]] <- n3d_stream_finish(s)
  probs_to_segments(do.call(rbind, probs))
}
}
}
\seealso{
\code{\link{n3d_stream_push}}, \code{\link{n3d_stream_finish}}
}
