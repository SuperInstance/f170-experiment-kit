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
