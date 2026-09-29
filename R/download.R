.n3d_repo <- "nvidia/Nemotron-3-Diarization"

# The Hugging Face commit this package was validated against. An exact
# commit makes hfhub read snapshots/<commit>/ directly, without refs/ or the
# network, and keeps a moving branch from changing the weights underneath.
.n3d_validated_revision <- "f667ed73aee57d40cc39428eb768b4fd87a0a29e"

#' Model Revision
#'
#' The Hugging Face commit of \code{nvidia/Nemotron-3-Diarization} that
#' \code{\link{load_n3d}} and \code{\link{download_n3d}} use by default.
#'
#' @return A 40-character commit hash.
#' @examples
#' n3d_revision()
#' @export
n3d_revision <- function() {
  .n3d_validated_revision
}

check_revision <- function(revision) {
  if (!is.character(revision) || length(revision) != 1L ||
      is.na(revision) || !grepl("^[0-9a-f]{40}$", revision)) {
    stop("revision must be a single 40-character hex commit, not a branch ",
         "name", call. = FALSE)
  }
  revision
}

#' Check Whether the Weights Are Cached
#'
#' @param revision Hugging Face commit to look for.
#' @return \code{TRUE} if the model weights are in the Hugging Face cache.
#' @examples
#' n3d_exists()
#' @export
n3d_exists <- function(revision = n3d_revision()) {
  # validated outside tryCatch so a bad revision is an error, not FALSE
  check_revision(revision)
  tryCatch({
    path <- hfhub::hub_download(.n3d_repo, "model.safetensors",
                                revision = revision,
                                local_files_only = TRUE)
    file.exists(path)
  }, error = function(e) FALSE)
}

#' Download the Model Weights
#'
#' Downloads the Nemotron 3 Diarization weights (about 400 MB) into the
#' Hugging Face cache with \pkg{hfhub}. In interactive sessions it asks for
#' consent first. The weights are published by NVIDIA under the OpenMDW
#' License 1.1.
#'
#' @param revision Hugging Face commit to download.
#' @param force Download again even when the weights are cached.
#' @return The path of the cached \code{model.safetensors}, invisibly.
#' @examples
#' \donttest{
#' if (interactive()) {
#'   download_n3d()
#' }
#' }
#' @export
download_n3d <- function(revision = n3d_revision(), force = FALSE) {
  check_revision(revision)
  if (!force && n3d_exists(revision)) {
    return(invisible(hfhub::hub_download(.n3d_repo, "model.safetensors",
                                         revision = revision,
                                         local_files_only = TRUE)))
  }
  if (interactive()) {
    ok <- utils::askYesNo(paste0(
      "Download Nemotron 3 Diarization weights (~400 MB, OpenMDW-1.1) ",
      "from huggingface.co/", .n3d_repo, "?"
    ))
    if (!isTRUE(ok)) {
      stop("download cancelled", call. = FALSE)
    }
  }
  path <- hfhub::hub_download(.n3d_repo, "model.safetensors",
                              revision = revision,
                              force_download = force)
  invisible(path)
}
