import jsbeautifier
from loguru import logger
import sentencepiece as spm
import torch
from torch.nn.utils.rnn import pad_sequence

from representjs.data.util import normalize_program


TYPED_MARKER_START = "__LS__"
TYPED_MARKER_END = "__LE__"


def _tokenize(deeptyper_line, sp, target_to_id, max_length, split_source_targets_by_tab=False):
    """Given a line from the .txt data files in DeepTyper, format and 
    tokenize the code into subwords while retaining a mapping between
    type labels and the subwords.
    
    Returns:
        Beautified program
        List of subword IDs
        List of (label_id, label_start, label_end) tuples where label_start/end specify a range of subword IDs"""
    assert TYPED_MARKER_START not in deeptyper_line
    assert TYPED_MARKER_END not in deeptyper_line
    cap_length = max_length != -1

    if split_source_targets_by_tab:
        # Code tokens and type labels are delimeted by tab, as in .json files
        js_tokens, labels = deeptyper_line.split("\t")
    else:
        # Code tokens and type labels are delimited by space after </s>, as in .txt files
        js_end = deeptyper_line.index("</s>") + len("</s>")
        js_tokens = deeptyper_line[:js_end]
        labels = deeptyper_line[js_end + 1 :]

    # Split code by spaces to get DeepTyper tokens, excluding <s>, </s>
    js_tokens = js_tokens.split(" ")[1:-1]
    labels = labels.split(" ")[1:-1]
    assert len(js_tokens) == len(labels)

    # Add markers to each labeled token: tokens without no-type label
    js_tokens_with_markers = []
    for tok, label in zip(js_tokens, labels):
        if label != "O":
            tok = f"{TYPED_MARKER_START}{tok}{TYPED_MARKER_END}"
        js_tokens_with_markers.append(tok)

    # Normalize program by beautifying
    js_beautified = jsbeautifier.beautify(" ".join(js_tokens_with_markers))
    js_beautified_norm = normalize_program(js_beautified)
    js_beautified_norm = js_beautified_norm

    # Subword tokenize, separately tokenizing each marked identifier
    js_buffer = js_beautified_norm
    subword_ids = [sp.PieceToId("<s>")]
    label_segments = []

    last_identifier_end = 0
    start = js_buffer.find(TYPED_MARKER_START)
    labels = list(filter(lambda l: l != "O", labels))
    label_i = 0
    if start < 0:
        # No labeled words in this text, just tokenize the full program
        buffer_ids = sp.EncodeAsIds(js_buffer)
        subword_ids.extend(buffer_ids[: max_length - 2] if cap_length else buffer_ids)
    while start >= 0:
        # buffer: "stuff [start ptr]__LS__typedIdentifier__LE__..."
        # Tokenize stuff before the typed identifier
        buffer_ids = sp.EncodeAsIds(js_buffer[last_identifier_end:start])
        if cap_length and len(subword_ids) + len(buffer_ids) + 1 > max_length:  # +1 for </s>
            break
        subword_ids.extend(buffer_ids)

        # Tokenize typed identifier and record label
        start = start + len(TYPED_MARKER_START)
        end = js_buffer.index(TYPED_MARKER_END, start)
        assert end > start, "Zero-length identifier"
        identifier = js_buffer[start:end]
        identifier_ids = sp.EncodeAsIds(identifier)
        if cap_length and len(subword_ids) + len(identifier_ids) + 1 > max_length:  # +1 for </s>
            break
        # A segment is (label_id, label_start, label_end)
        label_id = target_to_id.get(labels[label_i], target_to_id["$any$"])
        label_segments.append((label_id, len(subword_ids), len(subword_ids) + len(identifier_ids)))
        subword_ids.extend(identifier_ids)

        start = js_buffer.find(TYPED_MARKER_START, start + 1)
        last_identifier_end = end + len(TYPED_MARKER_END)
        label_i += 1

    subword_ids.append(sp.PieceToId("</s>"))
    assert subword_ids[0] == sp.PieceToId("<s>")
    assert subword_ids[-1] == sp.PieceToId("</s>")

    return js_beautified, subword_ids, label_segments


def load_type_vocab(vocab_path):
    """Make type (target) vocab. vocab_path should contain have one type on each line."""
    id_to_target = {}
    target_to_id = {}
    with open(vocab_path, "r") as f:
        for i, line in enumerate(f):
            assert line
            tok = line.strip()
            assert tok
            id_to_target[i] = tok
            target_to_id[tok] = i
    assert len(id_to_target) == len(target_to_id)
    print(f"Loaded vocab from {vocab_path} with {len(id_to_target)} items")
    return id_to_target, target_to_id


class DeepTyperDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, type_vocab_path, sentencepiece_filepath, max_length=1024, subword_regularization_alpha=0.0, split_source_targets_by_tab=False):
        assert subword_regularization_alpha == 0.0
        self.max_length = max_length
        self.subword_regularization_alpha = subword_regularization_alpha
        self.split_source_targets_by_tab = split_source_targets_by_tab

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(sentencepiece_filepath)

        self.id_to_target, self.target_to_id = load_type_vocab(type_vocab_path)

        self.lines = []
        with open(data_path, "r") as f:
            for line in f:
                self.lines.append(line.rstrip())

    def __getitem__(self, idx):
        line = self.lines[idx]
        _, subword_ids, label_segments = _tokenize(
            line, self.sp, self.target_to_id, self.max_length,
            split_source_targets_by_tab=self.split_source_targets_by_tab)
        if self.max_length != -1:
            assert len(subword_ids) <= self.max_length
        subword_ids = torch.tensor(subword_ids, dtype=torch.long)
        label_segments = torch.tensor(label_segments, dtype=torch.long)
        return (subword_ids, label_segments)

    def __len__(self):
        return len(self.lines)


def get_collate_fn(pad_id, no_type_id):
    def collate_fn(batch):
        """Batch is a list of tuples (x, y)"""
        B = len(batch)
        X, Y = zip(*batch)
        X = pad_sequence(X, batch_first=True, padding_value=pad_id)
        L = X.size(1)

        # Make masks for each label interval
        labels = torch.zeros(B, L, dtype=torch.long)
        labels.fill_(no_type_id)
        output_attn = torch.zeros(B, L, L, dtype=torch.float)
        for i, y in enumerate(Y):
            for label_id, label_start, label_end in y:
                labels[i, label_start] = label_id
                output_attn[i, label_start, label_start:label_end] = 1.0 / (label_end.item() - label_start.item())

        return X, output_attn, labels

    return collate_fn


if __name__ == "__main__":
    # "../DeepTyper/data/test-outputs-gold.json",
    dataset = DeepTyperDataset(
        "/home/ajay/coderep/DeepTyper/data/test_projects_gold_filtered.json",
        "../DeepTyper/data/target_wl",
        "data/codesearchnet_javascript/csnjs_8k_9995p_unigram_url.model",
        max_length=-1,
        split_source_targets_by_tab=True
    )
    max_ids, max_labels = 0, 0
    for i, (subword_ids, label_segments) in enumerate(dataset):
        max_ids = max(len(subword_ids), max_ids)
        max_labels = max(len(label_segments), max_labels)
        print(f"Dataset {i} max_ids {max_ids}, max_labels {max_labels}")
