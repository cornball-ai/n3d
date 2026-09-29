% tinyrox says don't edit this manually, but it can't stop you!
\name{probs_to_segments}
\alias{probs_to_segments}
\title{Speaker Probabilities to Segments}
\usage{
probs_to_segments(probs, threshold = 0.5, frame_duration = 0.01)
}
\arguments{
\item{probs}{Numeric matrix, frames x speakers, of speaker activity
probabilities (one frame per 10 ms), as returned by
\code{\link{diarize}} with \code{probs = TRUE} or by
\code{\link{n3d_stream_push}}.}

\item{threshold}{Probability above which a frame counts as speech.}

\item{frame_duration}{Seconds per frame.}
}
\value{
A data frame with columns \code{start} and \code{end} (seconds)
  and \code{speaker} (integer, numbered from 1 in order of first
  arrival), sorted by start time then speaker.
}
\description{
Thresholds per-frame speaker probabilities and returns one row per
contiguous run of activity. Overlapping speech gives overlapping segments.
}
\examples{
p <- matrix(0, 300, 8)
p[1:120, 1] <- 0.9
p[100:300, 2] <- 0.8
probs_to_segments(p)
}
