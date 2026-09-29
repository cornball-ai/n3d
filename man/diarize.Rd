% tinyrox says don't edit this manually, but it can't stop you!
\name{diarize}
\alias{diarize}
\title{Speaker Diarization}
\usage{
diarize(
  audio,
  model = NULL,
  threshold = 0.5,
  probs = FALSE,
  sample_rate = 16000L
)
}
\arguments{
\item{audio}{Path to an audio or video file, or a numeric vector of 16 kHz
mono samples.}

\item{model}{A model from \code{\link{load_n3d}}. Loaded on first use
when \code{NULL}.}

\item{threshold}{Speaker probability above which a frame counts as
speech.}

\item{probs}{Whether to also return the per-frame probabilities.}

\item{sample_rate}{Sampling rate of numeric \code{audio}; must be 16000.}
}
\value{
A data frame of segments with columns \code{start}, \code{end}
  (seconds) and \code{speaker}. With \code{probs = TRUE}, a list of
  \code{segments} and \code{probs}, a frames x 8 matrix with one frame per
  10 ms.
}
\description{
Finds who spoke when in a recording. Speakers are numbered from 1 in
order of first arrival; up to eight are tracked.
}
\details{
The recording is processed offline in 27.2 s chunks with 3.2 s of
look-ahead, carrying the arrival-order speaker cache between chunks, so
its length is not limited.

}
\examples{
\donttest{
if (n3d_exists()) {
  model <- load_n3d()
  mp3 <- system.file("samples", "Synapsis-Wonderland.mp3", package = "av")
  diarize(mp3, model)
}
}
}
