import sentencepiece as spm
import torch
import torch.nn.functional as F

import allennlp
import allennlp.nn
import allennlp.nn.beam_search


def ids_to_strs(Y, sp):
    if len(Y.shape) == 1:
        ids = []
        eos_id = sp.PieceToId("</s>")
        pad_id = sp.PieceToId("<pad>")
        for idx in Y:
            ids.append(int(idx))
            if int(idx) == eos_id or int(idx) == pad_id:
                break
        return sp.DecodeIds(ids)
    return [ids_to_strs(y, sp) for y in Y]


def greedy_decode(model, X, sp: spm.SentencePieceProcessor, max_decode_len=20, sample=True):
    start_token = sp.PieceToId("<s>")
    pad_token = sp.PieceToId("<pad>")
    B = X.size(0)
    model.eval()

    with torch.no_grad():
        decoded_batch = torch.zeros((B, 1), device=X.device).long()
        decoded_batch[:, 0] = start_token
        for t in range(max_decode_len):
            logits = model(X, decoded_batch)
            _, topi = logits[:, -1, :].topk(1)
            decoded_batch = torch.cat((decoded_batch, topi.view(-1, 1)), -1)
    Y_hat = decoded_batch.cpu().numpy()
    Y_hat_str = ids_to_strs(Y_hat, sp)
    model.train()
    return Y_hat_str


def beam_search_decode_eos(model, X, X_lengths, sp: spm.SentencePieceProcessor, eos_id, max_decode_len=20, k=3):
    # TODO: Implement constrained decoding (e.g. only alphanumeric)
    B = X.size(0)
    bos_id = sp.PieceToId("<s>")
    V = sp.GetPieceSize()  # Size of vocab
    model.eval()

    with torch.no_grad():
        # initial Y_hat and batchwise score tensors
        sequences = [
            (
                torch.zeros(B, max_decode_len, dtype=torch.long, device=X.device).fill_(bos_id),  # Y_hat
                torch.ones(B, dtype=torch.long),  # Y_hat_lengths
                torch.zeros(B, device=X.device),  # scores
                torch.zeros(B, dtype=torch.long, device=X.device),  # ended
            )
        ]
        # walk over each item in output sequence
        for t in range(max_decode_len - 1):
            all_candidates = []
            # expand each current candidate
            for Y_hat, Y_hat_lengths, scores, ended in sequences:
                Y_hat = Y_hat.to(X.device)
                scores = scores.to(X.device)
                logits = model(X, Y_hat[:, :-1].to(X.device), X_lengths, Y_hat_lengths)
                logits_t = logits[:, t, :]
                logprobs_t = F.log_softmax(logits_t, dim=-1).to(scores.device)  # [B, V] tensor
                for j in range(V):
                    # TODO: Only add probability if the sequence has not ended (generated </s>)
                    log_p_j = logprobs_t[:, j]  # log p(Y_t=j | Y_{<t-1}, X)
                    candidate_Y_hat = Y_hat.clone()
                    candidate_Y_hat[:, t + 1] = j
                    candidate_Y_hat_lengths = Y_hat_lengths.clone()
                    candidate_Y_hat_lengths = j
                    # candidate_ended = ended or j == eos_id
                    if j == eos_id:
                        candidate_ended = torch.ones_like(ended)
                    else:
                        candidate_ended = ended.clone()
                    candidate = (candidate_Y_hat, candidate_Y_hat_lengths, scores + log_p_j, candidate_ended)
                    all_candidates.append(candidate)
            # stack candidates
            beam_Y, beam_Y_lengths, beam_scores = zip(*all_candidates)
            beam_Y = torch.stack(beam_Y, dim=1)  # [B, V, T]
            beam_Y_lengths = (torch.stack(beam_Y_lengths, dim=1),)  # [B, V]
            beam_scores = torch.stack(beam_scores, dim=1)  # [B, V]
            # seleck k best per batch item
            topk_scores, topk_idx = torch.topk(beam_scores, k, dim=1, sorted=True)
            topk_Y = torch.gather(beam_Y, 1, topk_idx.unsqueeze(-1).expand(B, k, max_decode_len))
            topk_Y_lengths = torch.gather(beam_Y_lengths, 1, topk_idx.unsqueeze(-1).expand(B, k))
            # set beam
            sequences = [(topk_Y[:, j, :], topk_Y_lengths[:, j, :], topk_scores[:, j]) for j in range(k)]
            # TODO: exit early if all sentences in all beam sequences contain </s>

    # stack sequences
    beam_Y, beam_scores = zip(*sequences)
    beam_Y = torch.stack(beam_Y, dim=1)  # [B, k, T]
    beam_scores = torch.stack(beam_scores, dim=1)  # [B, k]
    model.train()
    return ids_to_strs(beam_Y, sp), beam_scores


@torch.no_grad()
def beam_search_decode(model, X, X_lengths, sp: spm.SentencePieceProcessor, max_decode_len, k):
    # TODO: Implement constrained decoding (e.g. only alphanumeric)
    B = X.size(0)
    pad_id = sp.PieceToId("[PAD]")
    bos_id = sp.PieceToId("<s>")
    eos_id = sp.PieceToId("</s>")
    V = sp.GetPieceSize()  # Size of vocab
    model.eval()

    # Encode X
    # if X.size(1) < 1024:
    #     assert X.ndim == 2
    #     X_padded = torch.zeros(X.size(0), 1024, dtype=X.dtype, device=X.device)
    #     X_padded.fill_(pad_id)
    #     X_padded[:, :X.size(1)] = X
    #     X = X_padded
    memory = model.encode(X).transpose(0, 1)  # [B, T, d_model]

    # gumbel_sampler = allennlp.nn.beam_search.GumbelSampler()
    allen_bs = allennlp.nn.beam_search.BeamSearch(
        end_index=eos_id,
        max_steps=max_decode_len,
        beam_size=k,
        # sampler=gumbel_sampler,
    )

    start_predictions = torch.tensor([bos_id] * B, dtype=torch.long, device=X.device)
    start_state = {
        "prev_tokens": torch.zeros(B, 0, dtype=torch.long, device=X.device),
        "memory": memory
    }
    def step(last_tokens, current_state, t):
        """
        Args:
            last_tokens: (group_size,)
            current_state: {}
            t: int
        """
        group_size = last_tokens.size(0)
        prev_tokens = torch.cat([
            current_state["prev_tokens"],
            last_tokens.unsqueeze(1)
        ], dim=-1)  # [B*k, t+1]

        all_log_probs = model.decode(current_state["memory"].transpose(0, 1), prev_tokens)
        next_log_probs = all_log_probs[:, -1, :]
        next_log_probs = torch.nn.functional.log_softmax(next_log_probs, dim=-1)
        assert next_log_probs.shape == (group_size, V)
        return (next_log_probs, {
            "prev_tokens": prev_tokens,
            "memory": current_state["memory"]
        })

    predictions, log_probs = allen_bs.search(
        start_predictions=start_predictions,
        start_state=start_state,
        step=step)

    model.train()
    prediction = ids_to_strs(predictions, sp)
    return prediction, log_probs


def beam_search_decode_old(model, X, X_lengths, sp: spm.SentencePieceProcessor, max_decode_len=20, k=3):
    # TODO: Implement constrained decoding (e.g. only alphanumeric)
    B = X.size(0)
    bos_id = sp.PieceToId("<s>")
    V = sp.GetPieceSize()  # Size of vocab
    model.eval()

    with torch.no_grad():
        Y_hat_lengths = torch.ones(B, dtype=torch.long).to(X.device)  # Y_hat_lengths
        # initial Y_hat and batchwise score tensors
        sequences = [
            (
                torch.zeros(B, max_decode_len).long().to(X.device) + bos_id,
                torch.zeros(B).to(X.device),
            )
        ]
        # walk over each item in output sequence
        for t in range(max_decode_len - 1):
            all_candidates = []
            # expand each current candidate
            for Y_hat, scores in sequences:
                Y_hat = Y_hat.to(X.device)
                scores = scores.to(X.device)
                logits = model(X, Y_hat[:, : t + 1].to(X.device), src_lengths=X_lengths, tgt_lengths=Y_hat_lengths + 1)
                logits_t = logits[:, t, :]
                logprobs_t = F.log_softmax(logits_t, dim=-1).to(scores.device)  # [B, V] tensor
                for j in range(V):
                    log_p_j = logprobs_t[:, j]  # log p(Y_t=j | Y_{<t-1}, X)
                    candidate_Y_hat = Y_hat.clone()
                    candidate_Y_hat[:, t + 1] = j
                    candidate = (candidate_Y_hat, scores + log_p_j)
                    all_candidates.append(candidate)
            # stack candidates
            beam_Y, beam_scores = zip(*all_candidates)
            beam_Y = torch.stack(beam_Y, dim=1)  # [B, V, T]
            beam_scores = torch.stack(beam_scores, dim=1)  # [B, V]
            # seleck k best per batch item
            topk_scores, topk_idx = torch.topk(beam_scores, k, dim=1, sorted=True)
            topk_Y = torch.gather(beam_Y, 1, topk_idx.unsqueeze(-1).expand(B, k, max_decode_len))
            # set beam
            sequences = [(topk_Y[:, j, :], topk_scores[:, j]) for j in range(k)]
            # TODO: exit early if all sentences in all beam sequences contain </s>
            Y_hat_lengths = Y_hat_lengths + 1

    # stack sequences
    beam_Y, beam_scores = zip(*sequences)
    beam_Y = torch.stack(beam_Y, dim=1)  # [B, k, T]
    beam_scores = torch.stack(beam_scores, dim=1)  # [B, k]
    model.train()
    return ids_to_strs(beam_Y, sp), beam_scores
