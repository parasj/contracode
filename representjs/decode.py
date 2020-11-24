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


@torch.no_grad()
def beam_search_decode(
    model,
    X,
    sp: spm.SentencePieceProcessor,
    max_decode_len,
    k,
    per_node_k=None,
    constrain_decoding=False,
    sampler="deterministic",
    top_p_threshold=0.9,
    top_p_temperature=1.0,
):
    if sampler == "top_p":
        sampler = allennlp.nn.beam_search.TopPSampler(p=top_p_threshold, temperature=top_p_temperature)
    elif sampler == "deterministic":
        sampler = None
    else:
        raise ValueError("Unsupported sampler")

    # TODO: Implement constrained decoding (e.g. only alphanumeric)
    B = X.size(0)
    pad_id = sp.PieceToId("[PAD]")
    bos_id = sp.PieceToId("<s>")
    eos_id = sp.PieceToId("</s>")
    V_full = sp.GetPieceSize()  # Size of vocab
    invalid_vocab_mask = torch.zeros(V_full, dtype=torch.bool, device=X.device)
    if constrain_decoding:
        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890_ "
        for id in range(V_full):
            piece = sp.IdToPiece(id)
            if not (id in [pad_id, bos_id, eos_id] or all(c in alphabet for c in piece)):
                invalid_vocab_mask[id] = True
    V = V_full
    model.eval()

    # Encode X
    allen_bs = allennlp.nn.beam_search.BeamSearch(
        end_index=eos_id, max_steps=max_decode_len, beam_size=k, per_node_beam_size=per_node_k, sampler=sampler,
    )

    start_predictions = torch.tensor([bos_id] * B, dtype=torch.long, device=X.device)
    start_state = {
        "prev_tokens": torch.zeros(B, 0, dtype=torch.long, device=X.device),
        "memory": model.encode(X).transpose(0, 1),  # [B, T, d_model]
    }

    def step(last_tokens, current_state, t):
        """
        Args:
            last_tokens: (group_size,)
            current_state: {}
            t: int
        """
        group_size = last_tokens.size(0)
        prev_tokens = torch.cat([current_state["prev_tokens"], last_tokens.unsqueeze(1)], dim=-1)  # [B*k, t+1]

        all_log_probs = model.decode(current_state["memory"].transpose(0, 1), prev_tokens)
        next_log_probs = all_log_probs[:, -1, :]
        if constrain_decoding:
            next_log_probs = next_log_probs.masked_fill(invalid_vocab_mask, float("-inf"))
        next_log_probs = torch.nn.functional.log_softmax(next_log_probs, dim=-1)
        assert next_log_probs.shape == (group_size, V)
        return (next_log_probs, {"prev_tokens": prev_tokens, "memory": current_state["memory"]})

    predictions, log_probs = allen_bs.search(start_predictions=start_predictions, start_state=start_state, step=step)

    model.train()
    prediction = ids_to_strs(predictions, sp)
    return prediction, log_probs
