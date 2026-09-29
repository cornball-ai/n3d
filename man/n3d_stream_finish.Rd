% tinyrox says don't edit this manually, but it can't stop you!
\name{n3d_stream_finish}
\alias{n3d_stream_finish}
\title{Finish a Stream}
\usage{
n3d_stream_finish(stream)
}
\arguments{
\item{stream}{A session from \code{\link{n3d_stream}}.}
}
\value{
A matrix of speaker probabilities for the remaining frames.
}
\description{
Scores the remaining buffered audio, including the look-ahead frames of
the previous chunk, and closes the session.
}
\examples{
\donttest{
if (n3d_exists()) {
  s <- n3d_stream(load_n3d())
  n3d_stream_push(s, numeric(32000))
  dim(n3d_stream_finish(s))
}
}
}
\seealso{
\code{\link{n3d_stream}} for a complete example.
}
