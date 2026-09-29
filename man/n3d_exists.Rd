% tinyrox says don't edit this manually, but it can't stop you!
\name{n3d_exists}
\alias{n3d_exists}
\title{Check Whether the Weights Are Cached}
\usage{
n3d_exists(revision = n3d_revision())
}
\arguments{
\item{revision}{Hugging Face commit to look for.}
}
\value{
\code{TRUE} if the model weights are in the Hugging Face cache.
}
\description{
Check Whether the Weights Are Cached
}
\examples{
n3d_exists()
}
