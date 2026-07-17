# %%
import math
import functools
import sys
from pathlib import Path
from typing import Callable

import circuitsvis as cv
import einops
import numpy as np
import torch as t
import torch.nn as nn
from eindex import eindex
from IPython.display import display
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm
from transformer_lens import (
    ActivationCache,
    FactoredMatrix,
    HookedTransformer,
    HookedTransformerConfig,
    utils,
)
from transformer_lens.hook_points import HookPoint

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

# Make sure exercises are in the path
chapter = "chapter1_transformer_interp"
section = "part2_intro_to_mech_interp"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part2_intro_to_mech_interp.tests as tests
from plotly_utils import (
    hist,
    imshow,
    plot_comp_scores,
    plot_logit_attribution,
    plot_loss_difference,
)

# Saves computation time, since we don't need it for the contents of this notebook
t.set_grad_enabled(False)

MAIN = __name__ == "__main__"

# %%
print(device)
# %%
gpt2_small: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small")

# %%
# Check out GPT-2
print(f"n_layers = {gpt2_small.cfg.n_layers}")
print(f"n_heads = {gpt2_small.cfg.n_heads}")
print(f"n_ctx = {gpt2_small.cfg.n_ctx}")
# %%
model_description_text = """## Loading Models

HookedTransformer comes loaded with >40 open source GPT-style models. You can load any of them in with `HookedTransformer.from_pretrained(MODEL_NAME)`. Each model is loaded into the consistent HookedTransformer architecture, designed to be clean, consistent and interpretability-friendly.

For this demo notebook we'll look at GPT-2 Small, an 80M parameter model. To try the model out, let's find the loss on this paragraph!"""

loss = gpt2_small(model_description_text, return_type="loss")
print("Model loss:", loss)
# %%
print(gpt2_small.to_str_tokens("gpt2"))
print(gpt2_small.to_str_tokens(["gpt2", "gpt2"]))
print(gpt2_small.to_tokens("gpt2"))
print(gpt2_small.to_string([50256, 70, 457, 17]))

print(gpt2_small.to_str_tokens("The Westgate is an exquisitely high end chain of fancy schmancy hotels!"))

# %%
gpt2_small("Hello world!")
maika_text = "The Westgate is an exquisitely high end chain of fancy schmancy hotels!"
gpt2_small(maika_text)

logits: Tensor = gpt2_small(maika_text, return_type="logits")
prediction = logits.argmax(dim=-1).squeeze()[:-1]
print(prediction)

gpt2_small.to_string(prediction)
# %%
logits: Tensor = gpt2_small(model_description_text, return_type="logits")
prediction = logits.argmax(dim=-1).squeeze()[:-1]
actual = gpt2_small.to_tokens(model_description_text).squeeze()[1:]
is_correct = prediction == actual

print(f"Model accuracy: {is_correct.sum()}/{len(actual)}")
print(f"Correct tokens: {gpt2_small.to_str_tokens(prediction[is_correct])}")


# %%
gpt2_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on task-specific datasets."
gpt2_tokens = gpt2_small.to_tokens(gpt2_text)
gpt2_logits, gpt2_cache = gpt2_small.run_with_cache(gpt2_tokens, remove_batch_dim=True)

print(type(gpt2_logits), type(gpt2_cache))

# %%
attn_patterns_from_shorthand = gpt2_cache["pattern", 0]
attn_patterns_from_full_name = gpt2_cache["blocks.0.attn.hook_pattern"]

t.testing.assert_close(attn_patterns_from_shorthand, attn_patterns_from_full_name)


# %%
layer0_pattern_from_cache = gpt2_cache["pattern", 0]

q, k = gpt2_cache["q", 0], gpt2_cache["k", 0]
seq, n_head, d_head = q.shape
layer0_attn_scores = einops.einsum(q, k, "seqQ n h, seqK n h -> n seqQ seqK")
mask = t.triu(t.ones((seq, seq), dtype=t.bool), diagonal=1).to(device)
layer0_attn_scores.masked_fill_(mask, -1e9)
layer0_pattern_from_q_and_k = (layer0_attn_scores / math.sqrt(d_head)).softmax(-1)
t.testing.assert_close(layer0_pattern_from_cache, layer0_pattern_from_q_and_k)
print("Tests passed!")

# %%
print(type(gpt2_cache))
attention_pattern = gpt2_cache["pattern", 0]
print(attention_pattern.shape)
gpt2_str_tokens = gpt2_small.to_str_tokens(gpt2_text)

print("Layer 0 Head Attention Patterns:")
display(
    cv.attention.attention_patterns(
        tokens=gpt2_str_tokens,
        attention=attention_pattern,
        attention_head_names=[f"L0H{i}" for i in range(12)],
    )
)
# L0H7 looks like a previous token head!

# %%
neuron_activations_for_all_layers = t.stack([
    gpt2_cache["post", layer] for layer in range(gpt2_small.cfg.n_layers)
], dim=1)
# shape = (seq_pos, layers, neurons)

cv.activations.text_neuron_activations(
    tokens=gpt2_str_tokens,
    activations=neuron_activations_for_all_layers
)

# %%
cfg = HookedTransformerConfig(
    d_model=768,
    d_head=64,
    n_heads=12,
    n_layers=2,
    n_ctx=2048,
    d_vocab=50278,
    attention_dir="causal",
    attn_only=True,  # defaults to False
    tokenizer_name="EleutherAI/gpt-neox-20b",
    seed=398,
    use_attn_result=True,
    normalization_type=None,  # defaults to "LN", i.e. layernorm with weights & biases
    positional_embedding_type="shortformer",
)

# %%
from huggingface_hub import hf_hub_download

REPO_ID = "callummcdougall/attn_only_2L_half"
FILENAME = "attn_only_2L_half.pth"

weights_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)


# %%
model = HookedTransformer(cfg)
pretrained_weights = t.load(weights_path, map_location=device, weights_only=True)
model.load_state_dict(pretrained_weights)

# %%
text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."

logits, cache = model.run_with_cache(text, remove_batch_dim=True)

# %%
str_tokens = model.to_str_tokens(text)
for layer in range(model.cfg.n_layers):
    # Capture the attention scores
    attention_pattern = cache["pattern", layer]
    # Display them
    display(cv.attention.attention_patterns(tokens=str_tokens, attention=attention_pattern))
# Head 07 looks like a previous token attention head
# Head03 is looking at first token, as is Head14 (mostly)
# Head 011 is mostly a same token head


# %% 
cache["pattern", 0].shape
cache["pattern", 0]

# %%
cache
# %%
def current_attn_detector(cache: ActivationCache) -> list[str]:
    """
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be current-token heads
    """
    # Head where the average score given to the same token is >= 0.8

    n_layers = sum(1 for name in cache.keys() if name.endswith("attn.hook_pattern"))
    heads = []
    for layer in range(n_layers):
        pattern = cache["pattern", layer]          # [n_heads, seq, seq]
        n_heads = pattern.shape[0]
        diag_means = pattern.diagonal(dim1=-2, dim2=-1).mean(-1)  # [n_heads]
        heads.extend(
            f"{layer}.{head}"
            for head in range(n_heads)
            if diag_means[head] >= 0.35
        )
    return heads    

print("Heads attending to current token  = ", ", ".join(current_attn_detector(cache)))
# %%
def prev_attn_detector(cache: ActivationCache) -> list[str]:
    """
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be prev-token heads
    """
    n_layers = sum(1 for name in cache.keys() if name.endswith("attn.hook_pattern"))
    heads = []
    for layer in range(n_layers):
        pattern = cache["pattern", layer]          # [n_heads, seq, seq]
        n_heads = pattern.shape[0]
        sub_diag_means = pattern.diagonal(offset = -1, dim1 = -2, dim2 = -1).mean(-1)  # [n_heads]
        heads.extend(
            f"{layer}.{head}"
            for head in range(n_heads)
            if sub_diag_means[head] >= 0.35
        )
    return heads    

print("Heads attending to previous token = ", ", ".join(prev_attn_detector(cache)))

# %%
def first_attn_detector(cache: ActivationCache) -> list[str]:
    """
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be first-token heads
    """
    n_layers = sum(1 for name in cache.keys() if name.endswith("attn.hook_pattern"))
    heads = []
    for layer in range(n_layers):
        pattern = cache["pattern", layer]          # [n_heads, seq, seq]
        n_heads = pattern.shape[0]
        first_col_means = pattern[..., :, 0].mean(-1)  # [n_heads]
        heads.extend(
            f"{layer}.{head}"
            for head in range(n_heads)
            if first_col_means[head] >= 0.6
        )
    return heads  

print("Heads attending to first token    = ", ", ".join(first_attn_detector(cache)))

# %%
def generate_repeated_tokens(
    model: HookedTransformer, seq_len: int, batch_size: int = 1
) -> Int[Tensor, "batch_size full_seq_len"]:
    """
    Generates a sequence of repeated random tokens

    Outputs are:
        rep_tokens: [batch_size, 1+2*seq_len]
    """
    t.manual_seed(0)  # for reproducibility
    prefix = (t.ones(batch_size, 1) * model.tokenizer.bos_token_id).long()
    # After the prefix, append a list of seq_len tokens repeated once
    rep_tokens_half = t.randint(0, model.cfg.d_vocab, (batch_size, seq_len), dtype=t.int64)
    rep_tokens = t.cat([prefix, rep_tokens_half, rep_tokens_half], dim=-1).to(device)
    return rep_tokens


def run_and_cache_model_repeated_tokens(
    model: HookedTransformer, seq_len: int, batch_size: int = 1
) -> tuple[Tensor, Tensor, ActivationCache]:
    """
    Generates a sequence of repeated random tokens, and runs the model on it, returning (tokens,
    logits, cache). This function should use the `generate_repeated_tokens` function above.

    Outputs are:
        rep_tokens: [batch_size, 1+2*seq_len]
        rep_logits: [batch_size, 1+2*seq_len, d_vocab]
        rep_cache: The cache of the model run on rep_tokens
    """
    rep_tokens = generate_repeated_tokens(model, seq_len, batch_size)
    rep_logits, rep_cache = model.run_with_cache(rep_tokens)
    return rep_tokens, rep_logits, rep_cache


def get_log_probs(
    logits: Float[Tensor, "batch posn d_vocab"], tokens: Int[Tensor, "batch posn"]
) -> Float[Tensor, "batch posn-1"]:
    logprobs = logits.log_softmax(dim=-1)
    # We want to get logprobs[b, s, tokens[b, s+1]], in eindex syntax this looks like:
    correct_logprobs = eindex(logprobs, tokens, "b s [b s+1]")
    return correct_logprobs


seq_len = 50
batch_size = 1
(rep_tokens, rep_logits, rep_cache) = run_and_cache_model_repeated_tokens(model, seq_len, batch_size)
rep_cache.remove_batch_dim()
rep_str = model.to_str_tokens(rep_tokens)
model.reset_hooks()
log_probs = get_log_probs(rep_logits, rep_tokens).squeeze()

print(f"Performance on the first half: {log_probs[:seq_len].mean():.3f}")
print(f"Performance on the second half: {log_probs[seq_len:].mean():.3f}")

plot_loss_difference(log_probs, rep_str, seq_len)

# %%
for layer in range(model.cfg.n_layers):
    # Capture the attention scores
    attention_pattern = cache["pattern", layer]
    # Display them
    display(cv.attention.attention_patterns(tokens=rep_str, attention=attention_pattern))

# %%
def induction_attn_detector(cache: ActivationCache) -> list[str]:
    """
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be induction heads

    Remember - the tokens used to generate rep_cache are (bos_token, *rand_tokens, *rand_tokens)
    """
    attn_heads = []
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            attention_pattern = cache["pattern", layer][head]
            # take avg of (-seq_len+1)-offset elements
            seq_len = (attention_pattern.shape[-1] - 1) // 2
            score = attention_pattern.diagonal(-seq_len + 1).mean()
            if score > 0.3:
                attn_heads.append(f"{layer}.{head}")
    return attn_heads



print("Induction heads = ", ", ".join(induction_attn_detector(rep_cache)))

# %%
seq_len = 50
batch_size = 10
rep_tokens_10 = generate_repeated_tokens(model, seq_len, batch_size)

# We make a tensor to store the induction score for each head.
# We put it on the model's device to avoid needing to move things between the GPU and CPU,
# which can be slow.
induction_score_store = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)


def induction_score_hook(pattern: Float[Tensor, "batch head_index dest_pos source_pos"], hook: HookPoint):
    """
    Calculates the induction score, and stores it in the [layer, head] position of the
    `induction_score_store` tensor.
    """
    # Take the diagonal of attn paid from each dest posn to src posns (seq_len-1) tokens back
    # (This only has entries for tokens with index>=seq_len)
    induction_stripe = pattern.diagonal(dim1=-2, dim2=-1, offset=1 - seq_len)
    # Get an average score per head
    induction_score = einops.reduce(induction_stripe, "batch head_index position -> head_index", "mean")
    # Store the result.
    induction_score_store[hook.layer(), :] = induction_score


# We make a boolean filter on activation names, that's true only on attention pattern names
pattern_hook_names_filter = lambda name: name.endswith("pattern")

# Run with hooks (this is where we write to the `induction_score_store` tensor`)
model.run_with_hooks(
    rep_tokens_10,
    return_type=None,  # For efficiency, we don't need to calculate the logits
    fwd_hooks=[(pattern_hook_names_filter, induction_score_hook)],
)

# Plot the induction scores for each head in each layer
imshow(
    induction_score_store,
    labels={"x": "Head", "y": "Layer"},
    title="Induction Score by Head",
    text_auto=".2f",
    width=900,
    height=350,
)

# %%
# Generated repeated tokens in GPT-2's tokenizer
seq_len = 50
batch_size = 10
rep_tokens_10 = generate_repeated_tokens(gpt2_small, seq_len, batch_size)

induction_score_store = t.zeros((gpt2_small.cfg.n_layers, gpt2_small.cfg.n_heads), device=gpt2_small.cfg.device)

# Visualize using a hook
def visualize_pattern_hook(
    pattern: Float[Tensor, "batch head_index dest_pos source_pos"],
    hook: HookPoint,
):
    print("Layer: ", hook.layer())
    display(cv.attention.attention_patterns(tokens=gpt2_small.to_str_tokens(rep_tokens[0]), attention=pattern.mean(0)))


# YOUR CODE HERE - find induction heads in gpt2_small

# Run with hooks (this is where we write to the `induction_score_store` tensor`)
gpt2_small.run_with_hooks(
    rep_tokens_10,
    return_type=None,  # For efficiency, we don't need to calculate the logits
    fwd_hooks=[(pattern_hook_names_filter, induction_score_hook)],
)

imshow(
    induction_score_store,
    labels={"x": "Head", "y": "Layer"},
    title="Induction Score by Head",
    text_auto=".1f",
    width=700,
    height=500,
)
# %%
# Observation: heads 5.1, 5.5, 6.9, 7.2, 7.10 are all strongly induction-y.
# Confirm observation by visualizing attn patterns for layers 5 through 7:

induction_head_layers = [5, 6, 7]
fwd_hooks = [
    (utils.get_act_name("pattern", induction_head_layer), visualize_pattern_hook)
    for induction_head_layer in induction_head_layers
]
gpt2_small.run_with_hooks(
    rep_tokens,
    return_type=None,
    fwd_hooks=fwd_hooks,
)
# %%
def logit_attribution(
    embed: Float[Tensor, "seq d_model"],
    l1_results: Float[Tensor, "seq nheads d_model"],
    l2_results: Float[Tensor, "seq nheads d_model"],
    W_U: Float[Tensor, "d_model d_vocab"],
    tokens: Int[Tensor, "seq"],
) -> Float[Tensor, "seq-1 n_components"]:
    """
    Inputs:
        embed: the embeddings of the tokens (i.e. token + position embeddings)
        l1_results: the outputs of the attention heads at layer 1 (with head as one of the dims)
        l2_results: the outputs of the attention heads at layer 2 (with head as one of the dims)
        W_U: the unembedding matrix
        tokens: the token ids of the sequence

    Returns:
        Tensor of shape (seq_len-1, n_components)
        represents the concatenation (along dim=-1) of logit attributions from:
            the direct path (seq-1,1)
            layer 0 logits (seq-1, n_heads)
            layer 1 logits (seq-1, n_heads)
        so n_components = 1 + 2*n_heads
    """
    W_U_correct_tokens = W_U[:, tokens[1:]]
    # Calculate embedding direct contribution as just the first seq-1 elements of embed dotted with unembed
    resid_contribution = einops.einsum(embed[:-1], W_U_correct_tokens, "seq d_model, d_model seq -> seq")

    # Calculate l1_results direct contribution similarly
    l1_contribution = einops.einsum(l1_results[:-1], W_U_correct_tokens, "seq nheads d_model, d_model seq -> seq nheads")
    
    # Calculate l2_results direct contribution similarly
    l2_contribution = einops.einsum(l2_results[:-1], W_U_correct_tokens, "seq nheads d_model, d_model seq -> seq nheads")

    return t.concat([
        resid_contribution.unsqueeze(-1),  # [seq-1, 1]
        l1_contribution,                   # [seq-1, n_heads]
        l2_contribution,                   # [seq-1, n_heads]
    ], dim=-1)  # [seq-1, 1 + 2*n_heads]


text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."
logits, cache = model.run_with_cache(text, remove_batch_dim=True)
str_tokens = model.to_str_tokens(text)
tokens = model.to_tokens(text)

with t.inference_mode():
    embed = cache["embed"]
    l1_results = cache["result", 0]
    l2_results = cache["result", 1]
    logit_attr = logit_attribution(embed, l1_results, l2_results, model.W_U, tokens[0])
    # Uses fancy indexing to get a len(tokens[0])-1 length tensor, where the kth entry is the predicted logit for the correct k+1th token
    correct_token_logits = logits[0, t.arange(len(tokens[0]) - 1), tokens[0, 1:]]
    t.testing.assert_close(logit_attr.sum(1), correct_token_logits, atol=1e-3, rtol=0)
    print("Tests passed!")


# %%
embed = cache["embed"]
l1_results = cache["result", 0]
l2_results = cache["result", 1]
logit_attr = logit_attribution(embed, l1_results, l2_results, model.W_U, tokens.squeeze())

plot_logit_attribution(model, logit_attr, tokens, title="Logit attribution (demo prompt)")

# %%
seq_len = 50

embed = rep_cache["embed"]
l1_results = rep_cache["result", 0]
l2_results = rep_cache["result", 1]

logit_attr = logit_attribution(embed, l1_results, l2_results, model.W_U, rep_tokens.squeeze())
plot_logit_attribution(model, logit_attr, rep_tokens.squeeze(), title="Logit attribution (random induction prompt)")


# %%
def head_zero_ablation_hook(
    z: Float[Tensor, "batch seq n_heads d_head"],
    hook: HookPoint,
    head_index_to_ablate: int,
) -> None:
    z[:, :, head_index_to_ablate, :] = 0.0


def get_ablation_scores(
    model: HookedTransformer,
    tokens: Int[Tensor, "batch seq"],
    ablation_function: Callable = head_zero_ablation_hook,
) -> Float[Tensor, "n_layers n_heads"]:
    """
    Returns a tensor of shape (n_layers, n_heads) containing the increase in cross entropy loss
    from ablating the output of each head.
    """
    # Initialize an object to store the ablation scores
    ablation_scores = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)

    # Calculating loss without any ablation, to act as a baseline
    model.reset_hooks()
    seq_len = (tokens.shape[1] - 1) // 2
    logits = model(tokens, return_type="logits")
    loss_no_ablation = -get_log_probs(logits, tokens)[:, -(seq_len - 1) :].mean()

    for layer in tqdm(range(model.cfg.n_layers)):
        for head in range(model.cfg.n_heads):
            # Create a temporary hook function with the head number fixed
            temp_hook_fn = functools.partial(ablation_function, head_index_to_ablate=head)
            # Run the model with the ablation hook
            ablated_logits = model.run_with_hooks(tokens, fwd_hooks=[(utils.get_act_name("z", layer), temp_hook_fn)])
            # Calculate the loss difference (= neg correct logprobs)
            loss = -get_log_probs(ablated_logits, tokens)[:, -(seq_len - 1) :].mean()
            # Store the result, subtracting the clean loss so that a value of 0 means no loss change
            ablation_scores[layer, head] = loss - loss_no_ablation

    return ablation_scores


ablation_scores = get_ablation_scores(model, rep_tokens)
tests.test_get_ablation_scores(ablation_scores, model, rep_tokens)

# %%
# Plot the loss impact of ablating each head
imshow(
    ablation_scores,
    labels={"x": "Head", "y": "Layer", "color": "Logit diff"},
    title="Loss Difference After Ablating Heads",
    text_auto=".2f",
    width=900,
    height=350,
)

# %%
def head_mean_ablation_hook(
    z: Float[Tensor, "batch seq n_heads d_head"],
    hook: HookPoint,
    head_index_to_ablate: int,
) -> None:
    z[:, :, head_index_to_ablate, :] = z[:, :, head_index_to_ablate, :].mean(dim = 0)


rep_tokens_batch = run_and_cache_model_repeated_tokens(model, seq_len=50, batch_size=10)[0]
mean_ablation_scores = get_ablation_scores(model, rep_tokens_batch, ablation_function=head_mean_ablation_hook)

imshow(
    mean_ablation_scores,
    labels={"x": "Head", "y": "Layer", "color": "Logit diff"},
    title="Loss Difference After Ablating Heads",
    text_auto=".2f",
    width=900,
    height=350,
)


# %%
# Plot how much average attention each head is paying to each position before a token
import plotly.express as px

max_offset = 10  # how far back to look

for layer in range(model.cfg.n_layers):
    pattern = cache["pattern", layer]  # [n_heads, seq_dest, seq_src]
    n_heads, seq_len, _ = pattern.shape

    # mean_attn[head, k] = mean attention paid to position t-k
    mean_attn = t.zeros(n_heads, max_offset + 1)
    for k in range(max_offset + 1):
        # k-th lower diagonal: attention from position i to position i-k
        diag = pattern.diagonal(offset=-k, dim1=-2, dim2=-1)  # [n_heads, seq_len - k]
        mean_attn[:, k] = diag.mean(dim=-1)

    px.imshow(
        mean_attn.cpu(),
        labels=dict(x="offset (tokens back)", y="head", color="mean attn"),
        title=f"Layer {layer}: mean attention by relative position",
        color_continuous_scale="Blues",
    ).show()

# %%
def make_ambiguous_bigram_tokens(
    model: HookedTransformer, seq_len: int = 50, batch_size: int = 10, min_gap: int = 12
) -> tuple[Int[Tensor, "batch full_seq"], Int[Tensor, "batch 2"], Int[Tensor, "batch 2 2"], Int[Tensor, "batch 3"]]:
    """
    Build sequences of random tokens with two planted bigrams sharing a second token:
        (A, B) -> X   and   (C, B) -> Y
    then repeat the whole sequence.

    Returns:
        tokens: [batch, 1 + 2*seq_len] (BOS prepended)
        ambig_positions: [batch, 2] - repeated-half B positions (where model predicts X or Y)
        answer_tokens: [batch, 2, 2] - (correct, confusable) token pair per ambiguous position
        control_positions: [batch, 3] - repeated-half positions with unambiguous prefixes
    """
    prefix = (t.ones(batch_size, 1) * model.tokenizer.bos_token_id).long()
    rand = t.randint(0, model.cfg.d_vocab, (batch_size, seq_len))

    # sample 5 distinct special tokens A, B, C, X, Y per batch element
    special = t.stack([t.randperm(model.cfg.d_vocab)[:5] for _ in range(batch_size)])
    A, B, C, X, Y = special.unbind(dim=1)

    # avoid accidental extra occurrences of B, X, Y in the filler (spurious matches)
    for i in range(batch_size):
        for tok in (B[i], X[i], Y[i]):
            rand[i][rand[i] == tok] = (tok + 1) % model.cfg.d_vocab

    # plant "A B X" at pos1 and "C B Y" at pos2
    pos1 = 5
    pos2 = pos1 + min_gap
    assert pos2 + 3 < seq_len
    rand[:, pos1], rand[:, pos1 + 1], rand[:, pos1 + 2] = A, B, X
    rand[:, pos2], rand[:, pos2 + 1], rand[:, pos2 + 2] = C, B, Y

    tokens = t.concat([prefix, rand, rand], dim=-1).to(model.cfg.device)

    # repeated-half B positions: model should predict X after first B, Y after second
    ambig_positions = t.tensor(
        [[1 + seq_len + pos1 + 1, 1 + seq_len + pos2 + 1]] * batch_size
    )
    answer_tokens = t.stack([
        t.stack([X, Y], dim=-1),   # at first B: correct=X, confusable=Y
        t.stack([Y, X], dim=-1),   # at second B: correct=Y, confusable=X
    ], dim=1)  # [batch, 2, 2]

    control_positions = t.tensor([[1 + seq_len + p for p in (20, 30, 40)]] * batch_size)

    return tokens, ambig_positions, answer_tokens, control_positions

# %%
def get_logit_diff(
    logits: Float[Tensor, "batch seq d_vocab"],
    positions: Int[Tensor, "batch n_pos"],
    answer_tokens: Int[Tensor, "batch n_pos 2"],
) -> float:
    """Mean (correct - confusable) logit at the given positions."""
    answer_logits = eindex(
        logits, positions.to(logits.device), answer_tokens.to(logits.device),
        "batch [batch n_pos] [batch n_pos answer]",
    )  # [batch, n_pos, 2]
    correct, wrong = answer_logits.unbind(dim=-1)
    return (correct - wrong).mean().item()


def get_control_loss(
    logits: Float[Tensor, "batch seq d_vocab"],
    tokens: Int[Tensor, "batch seq"],
    positions: Int[Tensor, "batch n_pos"],
) -> float:
    """Mean loss at control positions (predicting the token at pos+1)."""
    log_probs = logits.log_softmax(dim=-1)
    targets = tokens.gather(1, (positions + 1).to(tokens.device))
    pos_log_probs = eindex(
        log_probs, positions.to(logits.device), targets,
        "batch [batch n_pos] [batch n_pos] -> batch n_pos",
    )
    return -pos_log_probs.mean().item()


def zero_ablation_hook(
    z: Float[Tensor, "batch seq n_heads d_head"],
    hook: HookPoint,
    head_index_to_ablate: int,
) -> None:
    z[:, :, head_index_to_ablate, :] = 0.0
# %%
tokens, ambig_positions, answer_tokens, control_positions = make_ambiguous_bigram_tokens(
    model, seq_len=50, batch_size=10
)

conditions = {
    "clean": None,
    "ablate L0H4": (0, 4),
    "ablate L0H11": (0, 11),
    "ablate L0H7": (0, 7),   # positive control - should destroy everything
}

results = {}
for name, ablation in conditions.items():
    if ablation is None:
        logits = model(tokens)
    else:
        layer, head = ablation
        logits = model.run_with_hooks(
            tokens,
            fwd_hooks=[(
                utils.get_act_name("z", layer),
                functools.partial(zero_ablation_hook, head_index_to_ablate=head),
            )],
        )
    results[name] = {
        "ambig_logit_diff": get_logit_diff(logits, ambig_positions, answer_tokens),
        "control_loss": get_control_loss(logits, tokens, control_positions),
    }
    del logits

for name, r in results.items():
    print(f"{name:15s}  ambiguous logit diff: {r['ambig_logit_diff']:+.3f}   control loss: {r['control_loss']:.3f}")
# %%
INDUCTION_LAYER, INDUCTION_HEAD = 1, 10  # from your ablation heatmap
pattern_name = utils.get_act_name("pattern", INDUCTION_LAYER)


def get_induction_attn_at_ambig(
    tokens: Int[Tensor, "batch seq"], fwd_hooks: list = []
) -> tuple[float, float]:
    """Mean attention from ambiguous positions to the correct vs confusable source token."""
    cache = {}

    def cache_hook(pattern: Tensor, hook: HookPoint) -> None:
        cache["pattern"] = pattern.detach()

    model.run_with_hooks(
        tokens, return_type=None,
        fwd_hooks=fwd_hooks + [(pattern_name, cache_hook)],
    )
    pattern = cache["pattern"][:, INDUCTION_HEAD]  # [batch, seq, seq]

    # correct source = the continuation token after the matching first-half bigram,
    # which sits at (ambig_position - seq_len); the confusable source is the other one
    seq_len = (tokens.size(1) - 1) // 2
    correct_src = (ambig_positions - seq_len + 1).to(pattern.device)  # the continuation token
    wrong_src = correct_src.flip(dims=[1])  

    attn_correct = eindex(
        pattern, ambig_positions.to(pattern.device), correct_src,
        "batch [batch n_pos] [batch n_pos] -> batch n_pos",
    )
    attn_wrong = eindex(
        pattern, ambig_positions.to(pattern.device), wrong_src,
        "batch [batch n_pos] [batch n_pos] -> batch n_pos",
    )
    return attn_correct.mean().item(), attn_wrong.mean().item()


clean_c, clean_w = get_induction_attn_at_ambig(tokens)
abl_c, abl_w = get_induction_attn_at_ambig(tokens, fwd_hooks=[(
    utils.get_act_name("z", 0),
    functools.partial(zero_ablation_hook, head_index_to_ablate=4),
)])

print(f"clean:      attn to correct src {clean_c:.3f} | confusable src {clean_w:.3f}")
print(f"H4 ablated: attn to correct src {abl_c:.3f} | confusable src {abl_w:.3f}")

# %%
