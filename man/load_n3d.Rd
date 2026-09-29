% tinyrox says don't edit this manually, but it can't stop you!
\name{load_n3d}
\alias{load_n3d}
\title{Load the Diarization Model}
\usage{
load_n3d(
  device = default_device(),
  download = FALSE,
  revision = n3d_revision()
)
}
\arguments{
\item{device}{Torch device, \code{"cuda"} when available, otherwise
\code{"cpu"}.}

\item{download}{Whether to download the weights when they are not cached.
Defaults to \code{FALSE}; see \code{\link{download_n3d}}.}

\item{revision}{Hugging Face commit to load. Defaults to the commit this
package was validated against.}
}
\value{
An \code{n3d_model} object: a list holding the network and its
  configuration.
}
\description{
Builds the network and loads the Nemotron 3 Diarization weights from the
Hugging Face cache.
}
\examples{
\donttest{
if (n3d_exists()) {
  model <- load_n3d(device = "cpu")
}
}
}
