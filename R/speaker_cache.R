# Arrival-Order Speaker Cache and FIFO queue of Streaming Sortformer.
# Dimension key: B batch, C cache frames, F fifo frames, N input frames,
# D hidden size, S speakers.

#' New Speaker Cache
#'
#' @param cfg Model configuration from \code{n3d_config()}.
#' @param fifo_length,update_period FIFO capacity and number of frames moved
#'   to the cache when it overflows; default to the streaming sizes.
#' @return An environment holding the streaming state.
#' @noRd
new_speaker_cache <- function(cfg, fifo_length = NULL, update_period = NULL) {
    sc <- cfg$streaming
    cache <- new.env(parent = emptyenv())
    cache$fifo_length <- fifo_length %||% sc$fifo_length
    cache$update_period <- update_period %||% sc$speaker_cache_update_period
    cache$length <- sc$speaker_cache_length
    cache$num_silence <- sc$silence_frames_per_speaker
    cache$threshold <- sc$prediction_score_threshold
    cache$latest_boost <- sc$latest_frames_score_boost
    cache$num_speakers <- cfg$num_speakers
    cache$factor <- cfg$subsampling_factor

    budget <- cache$length %/% cache$num_speakers - cache$num_silence
    cache$min_positive <- floor(budget * sc$min_positive_scores_rate)
    cache$num_strong <- as.integer(floor(budget * sc$strong_boost_rate))
    cache$num_weak <- as.integer(floor(budget * sc$weak_boost_rate))

    cache$embeds <- NULL # (B, C, D)
    cache$probs <- NULL # (B, C, S)
    cache$fifo <- NULL # (B, F, D)
    cache$is_compressed <- FALSE
    class(cache) <- "n3d_speaker_cache"
    cache
}

# Cached frames to prepend to a chunk: the speaker cache, then the FIFO.
cache_embeds <- function(cache, chunk_BND) {
    if (is.null(cache$embeds)) {
        b <- chunk_BND$shape[1]
        d <- chunk_BND$shape[3]
        opts <- list(dtype = chunk_BND$dtype, device = chunk_BND$device)
        cache$embeds <- do.call(torch::torch_zeros, c(list(b, 0L, d), opts))
        cache$probs <- do.call(torch::torch_zeros,
                               c(list(b, 0L, cache$num_speakers), opts))
        cache$fifo <- do.call(torch::torch_zeros, c(list(b, 0L, d), opts))
    }
    torch::torch_cat(list(cache$embeds, cache$fifo), dim = 2L)
}

# Speaker probabilities at the encoder frame rate, zeroed on padding frames.
pool_probs <- function(cache, logits_BTS, valid_BN = NULL) {
    probs <- torch::nnf_avg_pool1d(logits_BTS$sigmoid()$transpose(2L, 3L),
                                   cache$factor, cache$factor)$transpose(2L, 3L)
    if (!is.null(valid_BN)) {
        probs <- probs * valid_BN$to(dtype = probs$dtype)$unsqueeze(-1L)
    }
    probs
}

num_popped_frames <- function(cache, num_fifo) {
    if (num_fifo <= cache$fifo_length) {
        return(0L)
    }
    min(max(cache$update_period, num_fifo - cache$fifo_length), num_fifo)
}

#' Push a Processed Chunk
#'
#' Appends the chunk's frames to the FIFO queue and, when it overflows, moves
#' its oldest frames to the speaker cache, compressing the cache when it
#' exceeds its capacity.
#'
#' @param input_BND Encoder input of the step: cached frames, chunk and
#'   look-ahead.
#' @param logits_BTS Speaker logits of the step at the mel frame rate.
#' @param silence_D Learned silence embedding.
#' @param num_chunk_frames Chunk frames (excluding look-ahead) to enqueue.
#' @param valid_BN Optional logical mask of valid input frames.
#' @noRd
cache_update <- function(cache, input_BND, logits_BTS, silence_D,
                         num_chunk_frames, valid_BN = NULL) {
    num_cache <- cache$embeds$shape[2]
    num_fifo <- cache$fifo$shape[2]
    probs_BNS <- pool_probs(cache, logits_BTS, valid_BN)

    chunk_BND <- input_BND$narrow(2L, num_cache + num_fifo + 1L,
                                  num_chunk_frames)
    fifo_BFD <- torch::torch_cat(list(cache$fifo, chunk_BND), dim = 2L)
    total_fifo <- fifo_BFD$shape[2]

    popped <- num_popped_frames(cache, total_fifo)
    if (popped > 0L) {
        fifo_probs <- probs_BNS$narrow(2L, num_cache + 1L, total_fifo)
        # an uncompressed cache holds plain chunk frames whose probabilities this
        # step re-estimates; a compressed one keeps the probabilities it stored
        stored <- if (cache$is_compressed) {
            cache$probs
        } else {
            probs_BNS$narrow(2L, 1L, num_cache)
        }
        new_embeds <- torch::torch_cat(list(cache$embeds,
                fifo_BFD$narrow(2L, 1L, popped)),
                                       dim = 2L)
        new_probs <- torch::torch_cat(list(stored,
                fifo_probs$narrow(2L, 1L, popped)),
                                      dim = 2L)
        fifo_BFD <- fifo_BFD$narrow(2L, popped + 1L, total_fifo - popped)

        if (new_embeds$shape[2] > cache$length) {
            compressed <- compress_cache(cache, new_embeds, new_probs, silence_D)
            new_embeds <- compressed$embeds
            new_probs <- compressed$probs
            cache$is_compressed <- TRUE
        }
        cache$embeds <- new_embeds
        cache$probs <- new_probs
    }
    cache$fifo <- fifo_BFD
    invisible(cache)
}

# Per-frame, per-speaker importance scores; -Inf marks frames to drop.
frame_scores <- function(cache, probs_BCS) {
    thr <- cache$threshold
    log_p <- torch::torch_log(probs_BCS$clamp(min = thr))
    log_q <- torch::torch_log((1 - probs_BCS)$clamp(min = thr))
    scores <- log_p - log_q + log_q$sum(dim = -1L, keepdim = TRUE) - log(0.5)

    is_speech <- probs_BCS > 0.5
    scores <- scores$masked_fill(is_speech$logical_not(), -Inf)
    is_positive <- scores > 0
    enough <- is_positive$sum(dim = 2L, keepdim = TRUE) >= cache$min_positive
    drop <- is_positive$logical_not()$logical_and(is_speech)$logical_and(enough)
    scores$masked_fill(drop, -Inf)
}

boost_scores <- function(scores_BCS, k, boost) {
    idx <- torch::torch_topk(scores_BCS, k, dim = 2L, sorted = FALSE)[[2]]
    scores_BCS$scatter_add(2L, idx, torch::torch_full_like(idx, boost,
            dtype = scores_BCS$dtype))
}

#' Compress the Speaker Cache
#'
#' Keeps the \code{speaker_cache_length} highest-scoring (frame, speaker)
#' slots, grouped by speaker and in arrival order within a speaker, with
#' \code{silence_frames_per_speaker} slots per speaker holding the silence
#' embedding.
#' @noRd
compress_cache <- function(cache, embeds_BCD, probs_BCS, silence_D) {
    b <- probs_BCS$shape[1]
    num_frames <- probs_BCS$shape[2]
    s <- probs_BCS$shape[3]

    scores <- frame_scores(cache, probs_BCS)
    # frames beyond the cache capacity are the ones just popped from the FIFO
    latest <- torch::torch_zeros_like(scores)
    if (num_frames > cache$length) {
        latest[, (cache$length + 1L):num_frames,] <- cache$latest_boost
    }
    scores <- scores + latest
    scores <- boost_scores(scores, cache$num_strong, -2 * log(0.5))
    scores <- boost_scores(scores, cache$num_weak, -log(0.5))
    scores <- torch::nnf_pad(scores, c(0L, 0L, 0L, cache$num_silence),
                             value = Inf)
    silence <- silence_D$to(dtype = embeds_BCD$dtype,
                            device = embeds_BCD$device)$view(c(1L, 1L, -1L))
    embeds_BCD <- torch::torch_cat(list(embeds_BCD,
                                        silence$expand(c(b, 1L, -1L))),
                                   dim = 2L)
    probs_BCS <- torch::nnf_pad(probs_BCS, c(0L, 0L, 0L, 1L))

    num_scored <- num_frames + cache$num_silence
    sentinel <- num_scored * s
    flat <- scores$transpose(2L, 3L)$reshape(c(b, -1L))
    top <- torch::torch_topk(flat, cache$length, dim = 2L, sorted = FALSE)
    # 0-based flat indices (speaker-major), as in the reference
    idx0 <- top[[2]] - 1L
    idx0 <- idx0$masked_fill(top[[1]] == -Inf, sentinel)
    idx0 <- torch::torch_sort(idx0, dim = 2L)[[1]]
    frame0 <- torch::torch_where(idx0 == sentinel,
                                 torch::torch_full_like(idx0, num_frames),
                                 (idx0 %% num_scored)$clamp(max = num_frames))
    frame1 <- frame0 + 1L

    kept_embeds <- vector("list", b)
    kept_probs <- vector("list", b)
    for (i in seq_len(b)) {
        rows <- frame1[i,]
        kept_embeds[[i]] <- embeds_BCD[i,,]$index_select(1L, rows)
        kept_probs[[i]] <- probs_BCS[i,,]$index_select(1L, rows)
    }
    list(embeds = torch::torch_stack(kept_embeds),
         probs = torch::torch_stack(kept_probs))
}
