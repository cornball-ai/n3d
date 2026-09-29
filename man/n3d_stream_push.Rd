% tinyrox says don't edit this manually, but it can't stop you!
\name{n3d_stream_push}
\alias{n3d_stream_push}
\title{Push Audio to a Stream}
\usage{
n3d_stream_push(stream, samples)
}
\arguments{
\item{stream}{A session from \code{\link{n3d_stream}}.}

\item{samples}{Numeric vector of 16 kHz mono samples.}
}
\value{
A matrix of speaker probabilities, frames x 8, for the frames
  completed by this push (possibly zero rows), one frame per 10 ms.
}
\description{
Push Audio to a Stream
}
\examples{
\donttest{
if (n3d_exists()) {
  s <- n3d_stream(load_n3d())
  p <- n3d_stream_push(s, numeric(32000)) # 2 s of silence
  dim(p)
}
}
}
\seealso{
\code{\link{n3d_stream}} for a complete example.
}
