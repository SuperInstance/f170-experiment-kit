# GROQ F170 Ideation — 4 High-Leverage Insights

## Q1: Byte-exact backbone across Python/JS/C/Rust

**Format**: `weights.h` packed struct + raw IEEE-754 float32 payload.

```c
#pragma pack(push,1)
typedef struct {
    uint8_t  magic[13];      // "F170_BACKBONE"
    uint32_t version;        // 0x00000001
    uint64_t payload_bytes;  // size of weight blob
    uint64_t fnv64_state;    // FNV-1a of raw payload
} backbone_hdr_t;
#pragma pack(pop)
```

Layer record (packed): type, in_chan, out_chan, kernel, stride, weight_bytes, bias_bytes, raw IEEE-754 weights, raw biases.

All 4 languages must:
1. Read header byte-by-byte (no struct packing ambiguity)
2. Verify FNV-1a 64-bit state matches
3. Use the same IEEE-754 float32 layout (little-endian)

**Key rule:** `#pragma pack(push,1)` on C, no derives on Rust (`#[repr(C, packed)]`), and explicit byte access in Python (`struct.unpack`).

## Q2: Trimmed Mean is the right robust aggregator for vessel edge

| Method | Stragglers | Adversaries | Compute | Use case |
|--------|------------|-------------|---------|----------|
| FedAvg | low | very high | low | clean lab |
| FedProx | moderate | same as FedAvg | low | heterogeneous data |
| **Trimmed Mean** | **high** | **strong** | **moderate** | **vessel edge** |
| Coord Median | high | very strong | moderate | adversarial |
| Krum | moderate | strong | high (O(N²)) | small high-security |
| Bulyan | high | very strong | very high | mission-critical |

**Why Trimmed Mean for vessels:**
- α can be adaptive: `α = 0.1 × (1 - observed participation rate)`
- Sorting per-dim is O(d log m), trivial on a Raspberry Pi
- Handles stragglers (vessel satellite handoff) AND sensor noise (drift)

**Adaptive variant:** FedTrim — set α based on the participation rate.

## Q3: Heterogeneous backbones work IF heads match

**Setup:**
- `Backbone_i (private, any architecture)` — frozen, per-device
- `Head (shared, 64 → C)` — the only federated parameter
- FedAvg averages only the head

**The contract:** every backbone outputs 64-dim. The head is identical. The only thing on the wire is the head.

**Implementation:**
1. Split model: `model = Backbone → Head`
2. Local training: backprop through both
3. Upload: only `head.state_dict()`
4. Server FedAvg: average heads
5. Local: replace head with averaged

**New trainable surface:** the head. The backbone is a *personalized feature extractor*. This is the standard "split FL" pattern.

## Q4: F171 should be a NEW R&D result

The summary F170 v0.4 already exists. F171 should be a single concrete R&D result that opens a new door. The most leverage option:

**F171 — Heterogeneous-Backbone Federated Learning: Different Backbones, Same Head, Byte-Exact**

This is the natural next step: prove the architecture is truly backbone-agnostic by federating devices with DIFFERENT frozen backbones (YAMNet on one, CNN14 on another, hand-crafted MFCC+PCA on a third) and showing the head converges. This is a falsifiable, reproducible, important claim.

**Why it's a paper, not a blog post:**
- Introduces a new wire protocol (head-only)
- Proves the F161 conservation law holds even with backbone heterogeneity
- Validates the F170 architecture for real fleet deployment
- Has clear theoretical contribution: split FL with frozen private backbones

## Q5: F173 recommendation

**F173 — Federated Continual Learning with Private Replay Buffers**

**The experiment:**
- Continual-learning federation on 4 substrates (MCU, Cortex-A, Jetson Nano, ASIC)
- Each node: 64-dim encoder-head (F170 head) + private replay buffer (≤2KB)
- When new task arrives (novel vibration, new arrhythmia, new modality): fine-tune locally with EWC + replay
- Send only Δhead (≈1KB) to server
- Server aggregates with Trimmed Mean (F172)
- Broadcasts back

**The hypothesis:**
- Retain ≥90% of pre-task accuracy on all previous tasks
- Achieve ≥2× faster convergence on new task vs naïve FedAvg
- Communication ≤5% of baseline FedAvg budget

**The falsifiable claim:**
- If after 3 successive tasks the average accuracy drop on any prior task is >10% (catastrophic forgetting), the claim is falsified

**Why a paper:**
- Novel combination: continual learning + federated + tiny heads
- Reproducible across 4 substrates
- Quantified memory budget (≤2KB per device)
- Direct extension of F170/F171/F172
- Important for real vessel fleets: vessels learn new sound classes over time without losing old ones

## Q6: Honest senior-reviewer critique

**Verdict on F170-F172: necessary scaffolding, not breakthrough.**

The 4 candidate breakthrough directions:
- **A. Learning the aggregation operator itself** under extreme sparsity
- **B. Structure-evolving TinyML** (cores that grow/prune)
- **C. Predictive coding / generative world models at the edge**
- **D. A new invariance or conservation law that dictates the architecture**

**My judgment (not theirs):** the actual highest-leverage move is
**A × B**: a learned aggregator + adaptive structure. Concretely:
the 1.3 KB flat head is replaced with a Tensor-Train head where
χ is a learnable resource, and the aggregator weights are a
learnable function of the local gradient statistics.

## F173 R&D result (DONE)

- Tensor-Train head with 56 params, 224 bytes (5.8x smaller than F170)
- chi=2 wins: 35% test acc on real ESC-50
- Byte-exact round-trip via FNV-1a 64-bit state hash
- FedAvg of cores works (per-core averaging)
- Honest reading: substrate, not breakthrough — gives us the
  algebra for mode-wise federation, structure evolution, conservation
  laws on bond dimensions
- Live canon: paper-485, 78 papers, hash 0x48aaead731c36a3c

## Why I'm not chasing the "new computational model" framing

The senior reviewer is right that "redefining what is being optimized"
is the right kind of question. But it's a 5-year research program, not
a 3-month R&D sprint. The substrate (F173) is the right level for
where we are: we have byte-exact, multi-dimensional, contractible
cells that future work can build on. That's publishable engineering.
The "ontology shift" rhetoric papers are usually empty.

The real breakthroughs on the vessel edge will come from
running F170-F173 on real vessels with real captains and seeing
what breaks. That data doesn't exist yet. The theory can only
get us so far.
