#' Chunked Forward
#'
#' Runs the network chunk by chunk, each chunk attending to the speaker
#' cache, the FIFO queue and its look-ahead frames.
#'
#' Offline mode (\code{cache = NULL}, \code{num_lookahead = NULL}): the
#' features are a whole recording, split into chunks of
#' \code{cfg$chunk_length} encoder frames with up to
#' \code{cfg$chunk_right_context} look-ahead frames, using the offline FIFO
#' sizes.
#'
#' Streaming mode (a \code{cache}, or \code{num_lookahead} given): the
#' features are one chunk plus \code{num_lookahead} trailing look-ahead
#' encoder frames, pushed through the given cache.
#'
#' @param net The \code{n3d_net} module.
#' @param features_BTM Log-mel features, \code{(B, T, 128)}.
#' @param valid_BT Optional logical mask of valid mel frames.
#' @param cache Speaker cache from a previous streaming step, or \code{NULL}.
#' @param num_lookahead Trailing look-ahead encoder frames, or \code{NULL}.
#' @return A list with \code{logits} \code{(B, T, S)}, on the device of
#'   \code{features_BTM}, and the \code{cache} (\code{NULL} in offline mode).
#' @noRd
chunked_forward <- function(net, features_BTM, valid_BT = NULL, cache = NULL,
                            num_lookahead = NULL) {
    cfg <- net$cfg
    is_streaming <- !is.null(cache) || !is.null(num_lookahead)
    if (is.null(cache)) {
        cache <- if (is_streaming) {
            new_speaker_cache(cfg)
        } else {
            new_speaker_cache(cfg, cfg$fifo_length,
                              cfg$speaker_cache_update_period)
        }
    }
    num_lookahead <- num_lookahead %||% 0L
    factor <- cfg$subsampling_factor

    # Features and logits for the whole input stay where the caller has them
    # (normally the CPU); only one chunk at a time goes to the model's
    # device, so device memory does not grow with the length of the audio.
    device <- net$silence_embeds$device
    b <- features_BTM$shape[1]
    num_frames <- features_BTM$shape[2]
    num_embeds <- (num_frames + factor - 1L) %/% factor
    num_chunk_embeds <- num_embeds - num_lookahead
    if (num_lookahead < 0L || num_chunk_embeds < 1L) {
        stop("num_lookahead (", num_lookahead, ") must be between 0 and one ",
             "less than the number of encoder frames (", num_embeds, ")",
             call. = FALSE)
    }

    valid_BN <- NULL
    if (!is.null(valid_BT)) {
        keep <- seq(1L, num_frames, by = factor)
        valid_BN <- valid_BT[, keep, drop = FALSE]
    }

    if (is_streaming) {
        chunk_length <- num_chunk_embeds
        right_context <- num_lookahead
    } else {
        chunk_length <- cfg$chunk_length
        right_context <- cfg$chunk_right_context
    }

    logits <- list()
    for (start0 in seq(0L, num_chunk_embeds - 1L, by = chunk_length)) {
        end0 <- min(start0 + chunk_length, num_chunk_embeds)
        num_chunk <- end0 - start0
        span <- min(end0 + right_context, num_embeds) - start0
        # stacking groups whole blocks of `factor` mel frames, so embedding a
        # block-aligned slice equals slicing the embedded whole
        first_frame <- start0 * factor
        mel_BTM <- features_BTM$narrow(2L, first_frame + 1L,
                                       min(span * factor,
                                           num_frames - first_frame))
        chunk_BND <- net$model$audio_tower$embedder(mel_BTM$to(device = device))

        cached_BND <- cache_embeds(cache, chunk_BND)
        num_cached <- cached_BND$shape[2]
        input_BND <- torch::torch_cat(list(cached_BND, chunk_BND), dim = 2L)

        step_valid <- NULL
        if (!is.null(valid_BN)) {
            chunk_valid <- valid_BN$narrow(2L, start0 + 1L, span)$to(
                device = device)
            ones <- torch::torch_ones(b, num_cached, dtype = torch::torch_bool(),
                                      device = device)
            step_valid <- torch::torch_cat(list(ones, chunk_valid), dim = 2L)
        }

        step_logits <- net$step(input_BND, step_valid)
        cache_update(cache, input_BND, step_logits, net$silence_embeds,
                     num_chunk, step_valid)

        logits[[length(logits) + 1L]] <- step_logits$narrow(
            2L, num_cached * factor + 1L, num_chunk * factor
        )$to(device = features_BTM$device)
        # A step leaves ~50 MB of dead tensors (layer outputs, head
        # temporaries) that R frees only when it collects, and R's heap
        # barely grows, so on a GPU they pile up over a stream. A minor
        # collection (~2 ms) releases them while they are still young.
        if (device$type == "cuda") {
            invisible(gc(full = FALSE))
        }
    }

    # with no look-ahead, the last encoder frame may be feature-stacking padding
    logits <- torch::torch_cat(logits, dim = 2L)
    logits <- logits$narrow(2L, 1L, min(num_frames, logits$shape[2]))
    list(logits = logits, cache = if (is_streaming) cache else NULL)
}
