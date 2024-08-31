r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**

"""

part1_q2 = r"""
**Your answer:**

"""

part1_q3 = r"""
**Your answer:**

"""

part1_q4 = r"""
**Your answer:**


"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers["batch_size"] = 96
    hypers["h_dim"] = 512
    hypers["z_dim"] = 64
    hypers["x_sigma2"] = 0.1
    hypers["learn_rate"] = 0.0002
    hypers["betas"] = (0.5, 0.999)
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**


"""

part2_q2 = r"""
**Your answer:**


"""

part2_q3 = r"""
**Your answer:**



"""

part2_q4 = r"""
**Your answer:**


"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_transformer_encoder_hyperparams():
    hypers = dict(
        embed_dim = 0, 
        num_heads = 0,
        num_layers = 0,
        hidden_dim = 0,
        window_size = 0,
        droupout = 0.0,
        lr=0.0,
    )

    # TODO: Tweak the hyperparameters to train the transformer encoder.
    # ====== YOUR CODE: ======
    hypers = dict(
        embed_dim=256,
        num_heads=4,
        num_layers=6,
        hidden_dim=128,
        window_size=32,
        droupout=0.1,
        lr=0.00013
    )
    # ========================
    return hypers




part3_q1 = r"""
**Your answer:**
Stacking encoder layers with sliding-window attention increases the effective receptive field of each token in the input sequence.
Initially, the receptive field of each token is limited to $sliding-window-size * 0.5$ distance from it.
This means that each token can only attend to a limited number of tokens in the input sequence.
However, as you stack more layers, each layer processes the output of the previous layer,
which has already integrated information from a local context.
This allows the model to capture information from a larger context in the input sequence.
Thus, the final layer accumulates information from a progressively larger context across multiple layers,
much like how stacking CNN layers increases the receptive field in image processing.
In CNNs, stacking layers allows the network to capture patterns from larger regions of the image.
Similarly, in transformers, stacking layers allows the model to incorporate information from increasingly distant tokens,
thereby broadening the context in the final layer.
For instance, assume we have a sliding window size of w = 6, and consider a sequence of 50 tokens.
After the first layer, the token at position 10 has gained information from tokens in the range [7, 13].
In the second layer, token 10 is processed again, but this time,
token 7 has already been enriched with the context from [4, 10] and token 13 with context from [10, 16].
So, token 10 now indirectly incorporates information from [4, 16], which is outside the original window size of 6.
This process continues as we stack more layers, allowing the model to capture information from a larger context.
"""

part3_q2 = r"""
**Your answer:**
One possible variation to maintain the computational complexity of $O(nw)$ while incorporating a more global context is
to use dilated sliding-window attention.
This would be similar to dilated convolutions in CNNs. In this approach, rather than attending to consecutive tokens within the window,
the model could attend to every $d$-th token within the window, where $d$ is a dilation factor.
This would allow the model to capture information from a broader context within each layer while keeping the window size
and computational complexity roughly the same.
For instance, consider a sequence of 20 tokens and a sliding window size of 4.
Normally, each token attends to its neighboring tokens directly, e.g., the token at index 0 attends to tokens at indices 0, 1, 2, and 3.
With a dilation factor $d = 2$, the token at index 0 will instead attend to tokens at indices 0, 2, 4, and 6 within the window.
This widens the receptive field within each layer without increasing the window size or overall complexity.
By adjusting the dilation factor, the model can control the effective receptive field of each token and incorporate information from a broader context.

"""


part4_q1 = r"""
**Your answer:**


"""

part4_q2 = r"""
**Your answer:**


"""


part4_q3= r"""
**Your answer:**


"""

part4_q4 = r"""
**Your answer:**


"""

part4_q5 = r"""
**Your answer:**


"""


# ==============
