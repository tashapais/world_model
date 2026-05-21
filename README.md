# World Model

A three-stage video world model: **VideoTokenizer** → **LatentActionModel** → **DynamicsModel**.

Trained on Sonic the Hedgehog gameplay. The model learns to predict future game frames given a context window of past frames, with optional action conditioning via unsupervised latent action discovery.

## Inference

![Sonic inference — ground truth (top) vs. model predictions (bottom)](demo/inference_results_gt_vs_pred.png)

*Ground truth frames (top row) vs. model-predicted frames (bottom row). Generated autoregressively with no action conditioning.*

![Sonic inference video](demo/inference_video_no_actions_20260521_150010.gif)

## Architecture

Three independently trained stages:

1. **VideoTokenizer** — VQ-VAE that compresses frames into discrete patch tokens via a space-only transformer encoder/decoder with FSQ bottleneck
2. **LatentActionModel** — encodes consecutive frame pairs into discrete action tokens without any action labels (unsupervised); decoder reconstructs next frame from current + action
3. **DynamicsModel** — MaskGIT-style masked transformer that predicts next-frame token distributions, conditioned on context window of past token sequences and optional action tokens

## Novel Things Tried

- **AdaLN-Zero conditioning** — replaced FiLM-style feature modulation with AdaLN-Zero: LayerNorm with no learned affine params, followed by a zero-initialized linear that predicts per-token `(scale, shift, gate)` triples from the conditioning signal. The zero init means residual paths start as identity at the beginning of training, which stabilises early optimisation. Included in all three model stages.

- **Rotary Position Embeddings (RoPE)** — replaced additive sinusoidal positional encodings with RoPE: 1D rotary embeddings on the temporal attention axis and 2D rotary embeddings (factored over height and width) on the spatial axis. RoPE is applied directly to Q and K before the attention dot-product rather than added to the token values, so position information decays naturally with distance and the model generalises better to sequence lengths not seen during training.

- **Halton and cosine MaskGIT unmasking schedules** — the original MaskGIT used an exponential confidence schedule to decide how many tokens to unmask each iteration. Added a cosine schedule (unmasks more tokens in later iterations, smoother curve) and a Halton low-discrepancy sequence schedule (quasi-random ordering that avoids the clumping of pure random sampling). Switchable via `maskgit_schedule: "exp" | "cosine" | "halton"` in the inference config.

- **Windowed action attention** — the latent action encoder originally mean-pooled patch embeddings from two consecutive frames and concatenated them before projecting to the action bottleneck. Replaced this with a short self-attention layer (window size 2) over the concatenated patches, letting the model learn which spatial regions are most informative for inferring the transition rather than treating all patches equally.

- **Pre-tokenized dataset cache** — during dynamics training the video tokenizer forward pass ran on every batch even though its weights were frozen. Added a preprocessing script (`scripts/preprocess_tokens.py`) that runs the tokenizer once over the full dataset and saves integer token indices to an HDF5 file. At training time the dynamics model reads pre-computed tokens directly, cutting per-step wall time roughly in half.

- **New datasets** — added Street Fighter, Terraria, and Space Invaders to the dataset registry alongside the existing Sonic, Pong, and Zelda datasets, enabling cross-game generalisation experiments.
