# vLLM gfx906-mobydick + KVarN

> **What this is:** the [ai-infos/vllm-gfx906-mobydick](https://github.com/ai-infos/vllm-gfx906-mobydick)
> fork (vLLM 0.23.1rc0, AMD **gfx906** / Vega20 — MI50 / MI60 / Radeon VII, ROCm 6.3.x)
> with **KVarN** variance-normalized KV-cache quantization patched in (branch
> `gfx906/kvarn`). KVarN is lifted from [huawei-csl/KVarN](https://github.com/huawei-csl/KVarN);
> the port is purely additive (new Triton/Python modules + small branches gated on
> `cache_dtype.startswith("kvarn_")`) — no C++/Rust changes, so it rides the fork's
> existing gfx906 PyTorch + triton-gfx906 + flash-attention-gfx906 stack.
>
> **TL;DR — KVarN is not a silver bullet on gfx906.** It is a KV-cache *capacity*
> lever (3–5× more KV at FP16-level accuracy), **not** a throughput lever on this
> hardware. On gfx906 it is typically *slower* than plain FP16 KV. Turn it on only
> when you need a large, accurate context window / high concurrent capacity and are
> willing to trade decode speed for it. See the reality check below before enabling.

## What KVarN is

A native vLLM attention backend + `--kv-cache-dtype` presets that quantize the KV
cache one fixed-size token tile at a time, in **float16 compute**:

1. **Hadamard rotation** along the channel dim (orthonormal → preserves attention
   scores; spreads per-channel outliers). Plain cached PyTorch Sylvester matrix via
   GEMM — no external `fast_hadamard_transform`.
2. **Sinkhorn-like variance normalization** (iterative row/col std-dev balancing in
   log space) to shrink quantization error before rounding.
3. **Asymmetric round-to-nearest** at low bit width; scales folded back at read time
   (keys per-channel, values per-token).

Presets (`--kv-cache-dtype`): `kvarn_k4v2_g128` (4-bit K / 2-bit V, group 128 — the
shipped default), `kvarn_k4v4_g128`, `kvarn_k4v2_g64`, `kvarn_k4v4_g64`,
`kvarn_mla_k4_g128` (MLA latent int4). **One vLLM block = one KVarN tile, so
`--block-size` must equal the group (128 or 64).**

**Why it keeps accuracy (vs TurboQuant):** the headline differentiator is that the
Hadamard+Sinkhorn recipe holds **FP16-level accuracy** while compressing to int4/int2.
TurboQuant (also wired in this fork) buys capacity but the
[vLLM TurboQuant blog](https://vllm.ai/blog/2026-05-11-turboquant) reports it gives up
**40–52% throughput** *and* costs accuracy. KVarN is built to keep accuracy — that is
its reason to exist. (Note: KVarN still *stores* the KV at int4/int2; it does not keep
an FP16 KV dtype. What it preserves is the FP16-level *accuracy*, not the bit width.)

KVarN's upstream claims (on datacenter NVIDIA GPUs): 3–5× KV capacity, **up to ~1.3×**
FP16 throughput, FP16-level accuracy, calibration-free. Read the next section for why
the throughput half of that does **not** transfer to gfx906.

## Enabling KVarN

```bash
vllm serve <model> \
  --dtype float16 \
  --kv-cache-dtype kvarn_k4v2_g128 \
  --block-size 128            # MUST equal the preset group
```

- `--dtype float16` — gfx906 has no native bf16; KVarN computes in fp16 anyway.
- Do **not** pin `--attention-backend`; the ROCm platform auto-selects the `KVARN`
  backend from the `kvarn_` dtype (you'll see `Overriding with KVARN ...` in the log).
- MLA models (e.g. GLM-4.7-Flash): use the same `kvarn_` dtype; it auto-routes to the
  MLA latent path.
- The fastest way to get a KVarN-patched image is `Dockerfile.quick` (overlay on the
  prebuilt mobydick image — no triton/flash-attn rebuild). See
  [Quick KVarN image](#-quick-kvarn-image-overlay-recommended) below.

## KVarN on gfx906 — reality check (read before enabling)

We brought this up on an **MI50 (gfx906, 16 GB)** and measured it. Findings:

**It works.** The KVARN backend registers, kernels JIT-compile, and it serves
end-to-end — verified correct OCR output on `datalab-to/chandra-ocr-2`. Accuracy is
not the problem.

**Throughput is the problem.** On chandra-ocr-2 (hybrid, 8 k context, batch 32) we saw
**~110 tok/s aggregate with KVarN vs ~200 tok/s with plain FP16 KV** (≈0.55×). That is
expected, not a misconfiguration — KVarN's `up to ~1.3×` claim has four asterisks, and
gfx906 fails all of them:

1. **Long context** (16 k–32 k). KVarN's throughput win comes from moving ~4× fewer KV
   bytes from HBM each decode step. At short context, weight reads dominate and the KV
   saving is negligible.
2. **VRAM/bandwidth-bound regime.** You must actually be hitting the KV wall. If KV
   already fits with batch headroom, KVarN only adds work.
3. **Tensor-core hardware.** The win assumes dequant is ~free next to a tensor-core
   GEMM. **gfx906 (Vega20) has no MFMA/matrix cores**, so KVarN's `tl.dot`-heavy decode
   + Hadamard/Sinkhorn/dequant runs the FMA path — the dequant overhead *is* the
   bottleneck and eats the bandwidth saving.
4. **"up to"** = best case. KVarN's own GLM-4.7-Flash MLA table shows **0.94×** (a
   throughput *loss*) at 32 k when the KV/latent is already small — even on their
   datacenter GPU.

chandra-on-MI50 sits outside all four (hybrid → only 8/32 layers hold KV; 8 k context;
no MFMA), so it lands in KVarN's worst quadrant.

**When KVarN *is* worth it on gfx906:** when your binding constraint is KV-cache
*capacity*, not speed, and you accept slower decode — e.g. you need a much longer or
higher-concurrency context than FP16 KV can fit, and you need that context to stay
accurate (where TurboQuant would cost accuracy). The lever is 3–5× capacity at
FP16-level accuracy. If you are chasing tok/s on a model whose KV already fits, leave
KVarN off and tune the non-KVarN levers (batch sizing, chunked prefill, the
GDN/linear-attn and ViT kernels).

### gfx906-specific caveats discovered during bring-up

- **`maxnreg` autotune fix.** KVarN's decode autotune space included NVIDIA-only
  `maxnreg=` configs; the gfx906 Triton fork rejects them at launch (`Keyword argument
  maxnreg ... unrecognised`), which killed EngineCore. They are now gated behind
  `not current_platform.is_rocm()` (commit on branch `gfx906/kvarn`).
- **Hybrid / Mamba models bump your block size.** For hybrid models (chandra-ocr-2 =
  Qwen3.5-VL) vLLM aligns the attention page to the (large, fixed) Mamba state page.
  Because KVarN shrinks the attention page ~4×, vLLM inflates `block_size` to match
  (e.g. **128 → 2176** for chandra), overriding your `--block-size 128`. Consequences:
  (a) you must set `--max-num-batched-tokens` ≥ that inflated block size or boot fails
  with `In Mamba cache align mode, block_size (N) must be <= max_num_batched_tokens`;
  (b) the KVarN kernels then run far off their tuned 128-tile point, hurting perf.
  Output was still correct in our test, but hybrid+KVarN is the least-tuned path.
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is a no-op here.** This ROCm
  build logs `expandable_segments not supported on this platform`. Under heavy
  concurrent multimodal prefill the OOMs land in the model's own GDN/linear-attn
  activations (not KVarN — KV usage was ~13%); the relief that actually worked was
  lowering `--max-num-batched-tokens` and `--gpu-memory-utilization`.
- **Tight single-GPU budget:** set `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` so the
  profiler doesn't over-reserve and shrink the KV pool. (On ROCm the cudagraph estimate
  is 0 anyway, so this is harmless.)

---

## Mini Install Guide for GFX906

### 🐳 Using Pre-built Docker Image (Recommended)

If you have Docker and the AMD ROCm drivers/kernel modules installed on your host system, you can totally bypass the complex manual source-build installation by using our pre-built Docker image.

```bash
# Pull the latest image (or specify a tag instead of latest, e.g. v0.19.1rc0.x)
docker pull aiinfos/vllm-gfx906-mobydick:latest

# Run the container interactively (Make sure to pass ROCm devices into the container and have your models in host /home/ as we map /home:/home; feel free to edit the command below to a safer one, without priviledged and others)
sudo docker run -it --name vllm-gfx906-mobydick -v /home:/home --network host --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add $(getent group render | cut -d: -f3) \
  --cap-add=SYS_ADMIN --volume /sys:/sys:ro --pid=host --privileged \
  --ipc=host aiinfos/vllm-gfx906-mobydick:latest
```

Once inside the container, you are all set! You can immediately start serving models (see the Quickstart example below).

---

### ⚡ Quick KVarN image (overlay, recommended)

The KVarN patch is pure Python + Triton (JIT) — **no C++/Rust/kernel changes** — so you
do **not** need to rebuild the whole stack to get a KVarN-enabled image. `Dockerfile.quick`
takes the prebuilt `aiinfos/vllm-gfx906-mobydick` image and just drops the patched
`vllm/` source on top of the installed package (`cp -r` merges, so the base image's
compiled `triton-gfx906` / `flash-attention-gfx906` / vLLM artifacts are preserved). This
takes seconds plus the one-time base-image pull, instead of the multi-hour full build.

```bash
# From a checkout of this repo (the patched source = the build context):
podman build -f Dockerfile.quick -t localhost/vllm-gfx906-mobydick-kvarn:latest .
#   docker also works:
#   docker build -f Dockerfile.quick -t vllm-gfx906-mobydick-kvarn:latest .

# Override the base image if you built/pulled a different one:
#   podman build -f Dockerfile.quick --build-arg BASE_IMAGE=<your image> -t ... .
```

Then serve with KVarN enabled (see [Enabling KVarN](#enabling-kvarn) above):

```bash
podman run --rm --device /dev/kfd --device /dev/dri --group-add keep-groups \
  --security-opt label=disable --ipc host -p 8000:8000 \
  -e HSA_OVERRIDE_GFX_VERSION=9.0.6 \
  localhost/vllm-gfx906-mobydick-kvarn:latest \
  vllm serve <model> --dtype float16 --kv-cache-dtype kvarn_k4v2_g128 --block-size 128
```

Prefer the full source build below if you need to change C++/Rust/kernels or rebase onto
a different base image.

---

### 🛠️ Manual Build from Source

If you prefer to build and install from source on your bare metal instead, follow the steps below:

### ROCm 6.3.4 & amdgpu drivers

```code
# Get the script that adds the AMD repo for 24.04 (noble)
wget https://repo.radeon.com/amdgpu-install/6.3.4/ubuntu/noble/amdgpu-install_6.3.60304-1_all.deb
sudo apt install ./amdgpu-install_6.3.60304-1_all.deb

# Install ROCm  6.3.4 including hip, rocblas, amdgpu-dkms etc (assuming the machine has already the advised compatible kernel 6.11)
sudo amdgpu-install --usecase=rocm --rocmrelease=6.3.4    

sudo usermod -aG render,video $USER

# Verify ROCm installation
rocm-smi --showproductname --showdriverversion
rocminfo


# Add iommu=pt if you later grow beyond two GPUs
# ROCm’s NCCL-/RCCL-based frameworks can hang on multi-GPU rigs unless the IOMMU is put in pass-through mode
# see https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.3.3/reference/install-faq.html#multi-gpu

sudo sed -i 's/GRUB_CMDLINE_LINUX_DEFAULT="/GRUB_CMDLINE_LINUX_DEFAULT="iommu=pt /' /etc/default/grub
sudo update-grub
sudo reboot
cat /proc/cmdline  # >>> to check: must return: "BOOT_IMAGE=... iommu=pt"

```

### vllm-gfx906-mobydick fork with its dependencies (python, torch, triton, flash-attn, etc)

```code

pyenv install 3.12.11
pyenv virtualenv 3.12.11 venv312
pyenv activate venv312

# PYTORCH 2.11.0

git clone --branch v2.11.0 --recursive https://github.com/pytorch/pytorch.git
cd pytorch

# Install Python Dependencies
pip install -r requirements.txt
pip install mkl-static mkl-include

# Hipify the Source (Convert CUDA to ROCm code)
python tools/amd_build/build_amd.py

# Build the wheel and install
export MAX_JOBS=96 # to be adjusted according to your setup to avoid OOM / freeze / crash
export USE_ROCM=1
export PYTORCH_ROCM_ARCH=gfx906
export CMAKE_PREFIX_PATH="${VIRTUAL_ENV}:${CMAKE_PREFIX_PATH}"

pip wheel --no-build-isolation -v -w dist -e . 2>&1 | tee build.log
pip install ./dist/torch*.whl


# TORCHVISION 0.26.0

# Install dependencies
sudo apt-get update && sudo apt-get install -y libpng-dev libjpeg-dev ffmpeg

# Build and Install
git clone --branch v0.26.0 https://github.com/pytorch/vision.git
cd vision
export FORCE_CUDA=1
export USE_ROCM=1
export PYTORCH_ROCM_ARCH=gfx906

python setup.py install


# TORCHAUDIO 2.11.0

# Build and Install
git clone --branch v2.11.0 https://github.com/pytorch/audio.git
cd audio
export PYTORCH_ROCM_ARCH=gfx906
export USE_ROCM=1

python setup.py install


# TRITON-GFX906 V3.6.0

git clone --branch v3.6.0+gfx906 https://github.com/ai-infos/triton-gfx906.git
cd triton-gfx906 
pip install -r python/requirements.txt
TRITON_CODEGEN_BACKENDS="amd" pip wheel --no-build-isolation -w dist . 2>&1 | tee build.log
pip install ./dist/triton-*.whl  


# FLASH-ATTENTION-GFX906 (triton backend)

git clone https://github.com/ai-infos/flash-attention-gfx906.git
cd flash-attention-gfx906
FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" python setup.py install

# VLLM-GFX906-MOBYDICK main

git clone https://github.com/ai-infos/vllm-gfx906-mobydick.git
cd vllm-gfx906-mobydick
pip install 'amdsmi>=6.3,<6.4'
pip install -r requirements/rocm.txt
pip wheel --no-build-isolation -v -w dist . 2>&1 | tee build.log
pip install ./dist/vllm-*.whl

# TRANSFORMERS (v5.7.0 or any other version <6 supporting your model)
pip install transformers==5.7.0
```

### Quickstart example (with Qwen3.5-0.8B)

```code
FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE VLLM_LOGGING_LEVEL=DEBUG vllm serve Qwen/Qwen3.5-0.8B \
  --dtype float16 \
  --kv-cache-dtype float16 \
  2>&1 | tee log.txt
```

NB: --dtype float16 is recommended to add for this gfx906 fork. If not set, vllm will take the dtype from config.json model which might be bfloat16, not natively supported on gfx906 (with potential fallback to float32, leading to slower inference)

CREDITS
-------

- https://github.com/nlzy/vllm-gfx906
- https://github.com/Said-Akbar/vllm-rocm
- https://github.com/vllm-project/vllm

---

<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

🔥 We have built a vLLM website to help you get started with vLLM. Please visit [vllm.ai](https://vllm.ai) to learn more.
For events, please visit [vllm.ai/events](https://vllm.ai/events) to join us.

---

## About

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has grown into one of the most active open-source AI projects built and maintained by a diverse community of many dozens of academic institutions and companies from over 2000 contributors.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests, chunked prefill, prefix caching
- Fast and flexible model execution with piecewise and full CUDA/HIP graphs
- Quantization: FP8, MXFP8/MXFP4, NVFP4, INT8, INT4, GPTQ/AWQ, GGUF, compressed-tensors, ModelOpt, TorchAO, and [more](https://docs.vllm.ai/en/latest/features/quantization/index.html)
- Optimized attention kernels including FlashAttention, FlashInfer, TRTLLM-GEN, FlashMLA, and Triton
- Optimized GEMM/MoE kernels for various precisions using CUTLASS, TRTLLM-GEN, CuTeDSL
- Speculative decoding including n-gram, suffix, EAGLE, DFlash
- Automatic kernel generation and graph-level transformations using torch.compile
- Disaggregated prefill, decode, and encode

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor, pipeline, data, expert, and context parallelism for distributed inference
- Streaming outputs
- Generation of structured outputs using xgrammar or guidance
- Tool calling and reasoning parsers
- OpenAI-compatible API server, plus Anthropic Messages API and gRPC support
- Efficient multi-LoRA support for dense and MoE layers
- Support for NVIDIA GPUs, AMD GPUs, and x86/ARM/PowerPC CPUs. Additionally, diverse hardware plugins such as Google TPUs, Intel Gaudi, IBM Spyre, Huawei Ascend, Rebellions NPU, Apple Silicon, MetaX GPU, and more.

vLLM seamlessly supports 200+ model architectures on Hugging Face, including:

- Decoder-only LLMs (e.g., Llama, Qwen, Gemma)
- Mixture-of-Expert LLMs (e.g., Mixtral, DeepSeek-V3, Qwen-MoE, GPT-OSS)
- Hybrid attention and state-space models (e.g., Mamba, Qwen3.5)
- Multi-modal models (e.g., LLaVA, Qwen-VL, Pixtral)
- Embedding and retrieval models (e.g., E5-Mistral, GTE, ColBERT)
- Reward and classification models (e.g., Qwen-Math)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

Install vLLM with [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`:

```bash
uv pip install vllm
```

Or [build from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source) for development.

Visit our [documentation](https://docs.vllm.ai/en/latest/) to learn more.

- [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for how to get involved.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

<!-- --8<-- [start:contact-us] -->
- For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
- For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
- For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
- For collaborations and partnerships, please contact us at [collaboration@vllm.ai](mailto:collaboration@vllm.ai)
<!-- --8<-- [end:contact-us] -->

## Media Kit

- If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit)
