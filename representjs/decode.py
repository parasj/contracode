import sentencepiece as spm
import torch
import torch.nn.functional as F


def ids_to_strs(Y, sp):
    if len(Y.shape) == 1:
        ids = []
        eos_id = sp.PieceToId("</s>")
        for idx in Y:
            ids.append(int(idx))
            if int(idx) == eos_id:
                break
        return sp.DecodeIds(ids)
    return [ids_to_strs(y, sp) for y in Y]


def greedy_decode(model, X, sp: spm.SentencePieceProcessor, max_decode_len=20, sample=True):
    # TODO: Implement constrained decoding (e.g. only alphanumeric)
    B = X.size(0)
    model.eval()
    with torch.no_grad():
        Y_hat = torch.zeros(B, max_decode_len, device=X.device).long()
        Y_hat.fill_(sp.PieceToId("<s>"))
        for t in range(max_decode_len - 1):
            logits = model(X, Y_hat)
            if sample:
                idx_t = torch.distributions.categorical.Categorical(logits=logits[:, t, :]).sample()
            else:
                idx_t = logits[:, t, :].argmax(dim=-1)
            Y_hat[:, t + 1] = idx_t
    Y_hat = Y_hat.cpu().numpy()
    Y_hat_str = ids_to_strs(Y_hat, sp)
    model.train()
    return Y_hat_str


def beam_search_decode(model, X, sp: spm.SentencePieceProcessor, max_decode_len=20, k=3):
    # TODO: Implement constrained decoding (e.g. only alphanumeric)
    B = X.size(0)
    bos_id = sp.PieceToId("<s>")
    V = sp.GetPieceSize()  # Size of vocab
    model.eval()

    with torch.no_grad():
        # initial Y_hat and batchwise score tensors
        sequences = [(torch.zeros(B, max_decode_len).long().to(X.device) + bos_id, torch.zeros(B).to(X.device))]
        # walk over each item in output sequence
        for t in range(max_decode_len - 1):
            all_candidates = []
            # expand each current candidate
            for Y_hat, scores in sequences:
                Y_hat = Y_hat.to(X.device)
                scores = scores.to(X.device)
                logits = model(X, Y_hat[:, :-1].to(X.device))
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

    # stack sequences
    beam_Y, beam_scores = zip(*sequences)
    beam_Y = torch.stack(beam_Y, dim=1)  # [B, k, T]
    beam_scores = torch.stack(beam_scores, dim=1)  # [B, k]
    model.train()
    return ids_to_strs(beam_Y, sp), beam_scores
