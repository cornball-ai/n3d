% tinyrox says don't edit this manually, but it can't stop you!
\name{download_n3d}
\alias{download_n3d}
\title{Download the Model Weights}
\usage{
download_n3d(revision = n3d_revision(), force = FALSE)
}
\arguments{
\item{revision}{Hugging Face commit to download.}

\item{force}{Download again even when the weights are cached.}
}
\value{
The path of the cached \code{model.safetensors}, invisibly.
}
\description{
Downloads the Nemotron 3 Diarization weights (about 400 MB) into the
Hugging Face cache with \pkg{hfhub}. In interactive sessions it asks for
consent first. The weights are published by NVIDIA under the OpenMDW
License 1.1.
}
\examples{
\donttest{
if (interactive()) {
  download_n3d()
}
}
}
