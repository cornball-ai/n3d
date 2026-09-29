% tinyrox says don't edit this manually, but it can't stop you!
\name{load_audio}
\alias{load_audio}
\title{Load Audio}
\usage{
load_audio(file)
}
\arguments{
\item{file}{Path to an audio or video file in any format FFmpeg reads.}
}
\value{
Numeric vector of samples in \code{[-1, 1]}.
}
\description{
Decodes an audio or video file with \pkg{av}, downmixed to mono and
resampled to 16 kHz.
}
\examples{
mp3 <- system.file("samples", "Synapsis-Wonderland.mp3", package = "av")
samples <- load_audio(mp3)
length(samples) / 16000 # seconds
}
