# Genie 3 Reconstruction

A three-stage video world model: **VideoTokenizer** → **LatentActionModel** → **DynamicsModel**.

Trained on Sonic the Hedgehog gameplay. The model learns to predict future game frames given a context window of past frames, with optional action conditioning via unsupervised latent action discovery.

## Inference

![Sonic inference - ground truth (top) vs. model predictions (bottom)](demo/inference_results_gt_vs_pred.png)

*Ground truth frames (top row) vs. model-predicted frames (bottom row). Generated autoregressively with no action conditioning.*

![Sonic inference video](demo/inference_video.gif)

## Architecture

![Architecture diagram](demo/architecture.jpeg)

Three independently trained stages:

1. **VideoTokenizer** - VQ-VAE that compresses frames into discrete patch tokens via a space-only transformer encoder/decoder with FSQ bottleneck
2. **LatentActionModel** - encodes consecutive frame pairs into discrete action tokens without any action labels (unsupervised); decoder reconstructs next frame from current + action
3. **DynamicsModel** - MaskGIT-style masked transformer that predicts next-frame token distributions, conditioned on context window of past token sequences and optional action tokens

## Novel Things Tried

- **AdaLN-Zero conditioning** - replaced FiLM-style feature modulation with AdaLN-Zero: LayerNorm with no learned affine params, followed by a zero-initialized linear that predicts per-token `(scale, shift, gate)` triples from the conditioning signal. The zero init means residual paths start as identity at the beginning of training, which stabilises early optimisation. Included in all three model stages.

- **Rotary Position Embeddings (RoPE)** - replaced additive sinusoidal positional encodings with RoPE: 1D rotary embeddings on the temporal attention axis and 2D rotary embeddings (factored over height and width) on the spatial axis. RoPE is applied directly to Q and K before the attention dot-product rather than added to the token values, so position information decays naturally with distance and the model generalises better to sequence lengths not seen during training.

- **Halton and cosine MaskGIT unmasking schedules** - the original MaskGIT used an exponential confidence schedule to decide how many tokens to unmask each iteration. Added a cosine schedule (unmasks more tokens in later iterations, smoother curve) and a Halton low-discrepancy sequence schedule (quasi-random ordering that avoids the clumping of pure random sampling). Switchable via `maskgit_schedule: "exp" | "cosine" | "halton"` in the inference config.

- **Windowed action attention** - the latent action encoder originally mean-pooled patch embeddings from two consecutive frames and concatenated them before projecting to the action bottleneck. Replaced this with a short self-attention layer (window size 2) over the concatenated patches, letting the model learn which spatial regions are most informative for inferring the transition rather than treating all patches equally.

- **Pre-tokenized dataset cache** - during dynamics training the video tokenizer forward pass ran on every batch even though its weights were frozen. Added a preprocessing script (`scripts/preprocess_tokens.py`) that runs the tokenizer once over the full dataset and saves integer token indices to an HDF5 file. At training time the dynamics model reads pre-computed tokens directly, cutting per-step wall time roughly in half.

- **New datasets** - added Street Fighter, Terraria, and Space Invaders to the dataset registry alongside the existing Sonic, Pong, and Zelda datasets, enabling cross-game generalisation experiments.

## Walls, Tradeoffs, and Weird Failures

**Autoregressive drift is brutal.** The fully autoregressive inference mode - feeding the model's own predictions back as context - collapsed into dark, blurry frames within 5-6 steps. Each small error compounds. Teacher-forced inference (always using ground truth context) looks clean, which means the single-step prediction is working but the model hasn't learned to be robust to its own noise. The real Genie paper likely handles this with training-time scheduled sampling or other tricks not implemented here.

**`use_latest_checkpoints` picked the wrong checkpoint.** The inference script's "find latest checkpoint" logic sorts by directory timestamp, not training quality. After a debug run that used `patch_size=4`, that run's directory was the most recent, so inference loaded it and immediately crashed with a tensor size mismatch (`64 patches vs 256 patches`) deep inside the forward pass. Took a while to trace since the error surfaced in patch embedding, not in the config loader. Fixed by pointing directly at the pretrained `.pth` files.

**FSDP and AMP can't be used together.** PyTorch's FSDP handles mixed precision internally via its own `MixedPrecisionPolicy` - enabling both AMP autocast and FSDP mixed precision simultaneously causes incorrect gradient scaling. Added a config-time validator that raises an error if both are set, since the failure mode is silent corruption rather than a clean crash.

**AdaLN-Zero needs a conditioning signal at every block.** FiLM conditioning was only applied at specific points in the network. Switching to AdaLN-Zero required threading a conditioning tensor through every transformer block, which touched a lot of call signatures. The zero-init of the output projection means it's a no-op at init (good for stability), but the extra linear at every layer adds ~15% parameter count.

**Windowed action attention added complexity for uncertain gain.** The mean-pool baseline for the action encoder is lossy but simple. The windowed attention replacement gives the model more expressive power but also more ways to overfit or ignore the structure entirely. Without an ablation training run it's hard to know if it actually helps - the architectural motivation is sound but the empirical payoff is unverified.

**Pre-tokenized cache trades disk for speed, with a catch.** Caching the tokenizer outputs halves dynamics training time, but the cache is tied to one specific tokenizer checkpoint. If you retrain the tokenizer, the cache is silently stale - the dynamics model will train on tokens from the old tokenizer without any error. Would need a hash check on the tokenizer weights to make this safe.
