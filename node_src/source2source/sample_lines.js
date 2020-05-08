const shuffle = (a) => {
    for (let i = a.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
}

module.exports = (js_src, {line_length_pct = 0.5}) => {
    const lines = js_src.split(/[\r\n]+/);//.slice(1, -1);
    const n_lines = lines.length;
    const n_lines_to_take = Math.ceil(n_lines * line_length_pct);
    const line_range = shuffle([...Array(n_lines).keys()]).slice(0, n_lines_to_take);
    const out_lines = [];
    const sorted_sample = line_range.sort();
    for (const line_idx of sorted_sample) {
        out_lines.push(lines[line_idx]);
    }
    return out_lines.join("\n");
}
