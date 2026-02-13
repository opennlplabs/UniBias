"""
Architecture abstraction layer for UniBias.
Supports both Llama-family and GPT-2-family models.
"""

import torch


def detect_architecture(model):
    """Detect whether model is Llama-like or GPT2-like."""
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return 'llama'
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return 'gpt2'
    raise ValueError(f"Unsupported model architecture: {type(model)}")


def get_layers(model, arch):
    return model.model.layers if arch == 'llama' else model.transformer.h


def get_num_layers(model):
    if hasattr(model.config, 'num_hidden_layers'):
        return model.config.num_hidden_layers
    return model.config.n_layer


def get_num_heads(model):
    if hasattr(model.config, 'num_attention_heads'):
        return model.config.num_attention_heads
    return model.config.n_head


def get_hidden_size(model):
    if hasattr(model.config, 'hidden_size'):
        return model.config.hidden_size
    return model.config.n_embd


def get_norm(model, arch):
    return model.model.norm if arch == 'llama' else model.transformer.ln_f


def get_ffn_down_proj(model, layer_idx, arch):
    """Get FFN output projection (down_proj for Llama, mlp.c_proj for GPT-2)."""
    layers = get_layers(model, arch)
    return layers[layer_idx].mlp.down_proj if arch == 'llama' else layers[layer_idx].mlp.c_proj


def get_ffn_up_proj(model, layer_idx, arch):
    """Get FFN input projection (up_proj for Llama, mlp.c_fc for GPT-2)."""
    layers = get_layers(model, arch)
    return layers[layer_idx].mlp.up_proj if arch == 'llama' else layers[layer_idx].mlp.c_fc


def get_ffn_value_vectors(model, layer_idx, arch):
    """Get value vectors matrix with shape (intermediate_size, hidden_size).
    Each row is one value vector that can be projected to vocabulary space.
    """
    down_proj = get_ffn_down_proj(model, layer_idx, arch)
    if arch == 'llama':
        # Linear weight: (hidden_size, intermediate_size) -> transpose
        return down_proj.weight.T
    else:
        # Conv1D weight: (intermediate_size, hidden_size) -> already correct
        return down_proj.weight


def get_attn_module(model, layer_idx, arch):
    layers = get_layers(model, arch)
    return layers[layer_idx].self_attn if arch == 'llama' else layers[layer_idx].attn


def get_attn_output_proj(model, layer_idx, arch):
    """Get attention output projection (o_proj for Llama, attn.c_proj for GPT-2)."""
    layers = get_layers(model, arch)
    return layers[layer_idx].self_attn.o_proj if arch == 'llama' else layers[layer_idx].attn.c_proj


def get_attn_proj_weight_per_head(module, arch, num_heads, head_dim, hidden_size):
    """Get output projection weight reshaped as [num_heads, head_dim, hidden_size].
    This allows computing per-head contributions to the residual stream.
    """
    if arch == 'llama':
        # Linear weight: (hidden_size, hidden_size) = (out, in)
        # weight.T: (in, out) = (num_heads*head_dim, hidden_size)
        return module.weight.T.view(num_heads, head_dim, hidden_size)
    else:
        # Conv1D weight: (hidden_size, hidden_size) = (in, out)
        # = (num_heads*head_dim, hidden_size)
        return module.weight.view(num_heads, head_dim, hidden_size)


def get_vocab_size(tokenizer):
    return len(tokenizer)


def get_attn_start_layer(model):
    """Get layer index from which to start looking for biased attention heads."""
    num_layers = get_num_layers(model)
    return num_layers * 2 // 3


def setup_attention_infrastructure(model, arch):
    """One-time setup: add mask tensors and register permanent masking pre-hooks."""
    num_heads = get_num_heads(model)
    hidden_size = get_hidden_size(model)
    head_dim = hidden_size // num_heads
    num_layers = get_num_layers(model)
    layers = get_layers(model, arch)

    # Add mask tensors to each attention layer
    for layer_idx in range(num_layers):
        attn = get_attn_module(model, layer_idx, arch)
        attn.unibias_mask = torch.ones(1, num_heads, 1, 1)
        attn.unibias_head_output = None

    # Register permanent masking pre-hooks on attention output projections
    for layer_idx in range(num_layers):
        out_proj = get_attn_output_proj(model, layer_idx, arch)

        def make_mask_hook(li):
            def hook(module, input):
                attn_mod = get_attn_module(model, li, arch)
                mask = attn_mod.unibias_mask.to(input[0].device)
                if not torch.all(mask == 1):
                    x = input[0]
                    bs, sl, ed = x.shape
                    per_head = x.view(bs, sl, num_heads, head_dim).transpose(1, 2)
                    masked = per_head * mask
                    return (masked.transpose(1, 2).reshape(bs, sl, ed),)
            return hook

        out_proj.register_forward_pre_hook(make_mask_hook(layer_idx))


def register_head_capture_hooks(model, arch):
    """Register hooks that capture per-head attention outputs projected to hidden_size space.
    Returns list of hooks that should be removed after use.
    """
    num_heads = get_num_heads(model)
    hidden_size = get_hidden_size(model)
    head_dim = hidden_size // num_heads
    num_layers = get_num_layers(model)

    hooks = []
    for layer_idx in range(num_layers):
        out_proj = get_attn_output_proj(model, layer_idx, arch)

        def make_hook(li):
            def hook(module, input, output):
                x = input[0]  # [batch, seq_len, hidden_size] (merged heads)
                bs, sl, _ = x.shape
                # Un-merge heads: [batch, seq_len, num_heads, head_dim]
                per_head = x.view(bs, sl, num_heads, head_dim).transpose(1, 2)
                # [batch, num_heads, seq_len, head_dim]

                # Apply output projection per-head to get contribution in hidden_size space
                proj_weight = get_attn_proj_weight_per_head(
                    module, arch, num_heads, head_dim, hidden_size)
                # proj_weight: [num_heads, head_dim, hidden_size]

                # Per-head output: [batch, num_heads, seq_len, hidden_size]
                per_head_output = torch.einsum('bnsd,ndh->bnsh', per_head, proj_weight)

                # Save only last 10 positions to save memory
                attn = get_attn_module(model, li, arch)
                attn.unibias_head_output = per_head_output[:, :, -10:, :].detach().cpu()
            return hook

        h = out_proj.register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

    return hooks


def set_attention_masks(model, AHs_dict, debias_alpha, arch):
    """Set attention head masks (modifies mask tensor values)."""
    for layer in AHs_dict.keys():
        head_indexes = AHs_dict[layer]
        attn = get_attn_module(model, int(layer), arch)
        attn.unibias_mask[0, head_indexes, 0, 0] = debias_alpha


def remove_attention_masks(model, AHs_dict, arch):
    """Reset attention head masks to 1."""
    for layer in AHs_dict.keys():
        head_indexes = AHs_dict[layer]
        attn = get_attn_module(model, int(layer), arch)
        attn.unibias_mask[0, head_indexes, 0, 0] = 1


def find_answer_location(full_tokens, task='sst2'):
    """Find the position of the answer token in tokenized prompt (tokenizer-agnostic)."""
    if task in ('sst2', 'cr', 'mr', 'sst5'):
        marker = 'entiment:'  # End of "Sentiment:"
    elif task == 'trec':
        marker = 'Type:'  # End of "Answer Type:"
    else:
        marker = 'Answer:'

    # Join all tokens to reconstruct text
    full_text = ''.join(full_tokens)

    # Find the last occurrence of the marker (test sample, not demo)
    marker_pos = full_text.rfind(marker)
    if marker_pos == -1:
        raise ValueError(f"Could not find marker '{marker}' in tokens")

    marker_end = marker_pos + len(marker)

    # Map character position back to token index
    char_pos = 0
    for i, token in enumerate(full_tokens):
        char_pos += len(token)
        if char_pos >= marker_end:
            answer_idx = i + 1
            # Skip empty/whitespace-only tokens
            while answer_idx < len(full_tokens) and full_tokens[answer_idx].strip() == '':
                answer_idx += 1
            return answer_idx

    raise ValueError(f"Could not map marker position to token index")


def find_possible_ids_for_multi_str(arg_str_list, tokenizer):
    """Find token IDs that match label strings (vocab-size agnostic)."""
    ids_dict = {arg_str[0]: [] for arg_str in arg_str_list}
    vocab_size = len(tokenizer)

    for id in range(vocab_size):
        decoded = tokenizer.decode(id)
        if decoded:
            for arg_str_tuple in arg_str_list:
                for arg_str in arg_str_tuple:
                    decoded_lower = decoded.lower()
                    arg_lower = arg_str.lower()
                    if len(arg_lower) > 1:
                        if decoded_lower in arg_lower and arg_lower[0] == decoded_lower[0] and len(decoded_lower) > 1:
                            ids_dict[arg_str_tuple[0]].append(id)
                    else:
                        if decoded_lower in arg_lower and arg_lower[0] == decoded_lower[0]:
                            ids_dict[arg_str_tuple[0]].append(id)

    ids_list = list(ids_dict.values())
    max_len = max(len(sublist) for sublist in ids_list)
    padded_lst = [sublist + [sublist[0]] * (max_len - len(sublist)) for sublist in ids_list]
    return padded_lst
