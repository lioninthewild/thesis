"""
===================================================================
CBDC Metadata De-Anonymization Risk Model
Final Implementation — Nepal Context
===================================================================

Title:  Quantifying Privacy Risk in Token-Based CBDC:
        A Normalised Metadata De-Anonymization Framework for Nepal

University:  Kathmandu University, School of Science

===================================================================
MODEL OVERVIEW
===================================================================

Population:  N users (variable, default N=1,000)
Scope:       Domestic token-based CBDC under NRB jurisdiction
Platform:    Hyperledger Fabric (reference architecture)

6 Cryptographic Schemes:
    1. Plain Hash Token     — all metadata fully exposed
    2. Blind Signature      — wallet randomised, timestamp coarsened
    3. Hybrid Scheme        — blind sig wallet + ring sig amounts
    4. Ring Signature       — ring anonymity set, bucketed amounts
    5. Stealth Address      — one-time wallet per transaction
    6. ZKP Token            — zero-knowledge proofs, nullifier wallet

7 Attributes (m=7, R_max=28):
    Standard:  amount, timestamp, merchant, wallet, frequency
    Nepal:     payment_channel, transaction_zone

Excluded:  remittance_corridor
    Justification: thesis scope is domestic CBDC only.
    Cross-border remittances involve separate settlement rails
    outside NRB's domestic CBDC mandate.

===================================================================
ANALYSES
===================================================================

    A3  — Six-scheme comparison        (m=7, N=1,000)
    A5  — Population scalability       (m=7, N=100 to N=100K)
    A7  — Mitigation effectiveness     (m=7, N=1,000)
    A8  — m-Attribute sensitivity      (m=1 to m=7)
    A9  — Seed sensitivity             (30 seeds, m=7, N=1,000)
    A10 — Micro-scale pilot test       (N=10 to N=1,000)
    A11 — Lambda sensitivity           (λ=0 to λ=100)
    A12 — Worst-case scenario          (R_norm approaching 1.0)

===================================================================
ADVERSARY MODEL
===================================================================

    Type:       Passive, honest-but-curious observer
    Access:     All 7 metadata fields for every transaction
    Capability: Unlimited computation (information-theoretic)
    Goal:       Re-identify user u from transaction metadata
    Covers:     NRB insider, network eavesdropper,
                merchant coalition

===================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from itertools import combinations
import os
import warnings
warnings.filterwarnings("ignore")

# ── Output directory ──────────────────────────────────────────────
OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Plot styling ──────────────────────────────────────────────────
BG   = "white"
AX   = "white"
EDGE = "black"

def sax(ax):
    """Apply light theme styling to a matplotlib axis."""
    ax.set_facecolor(AX)
    ax.tick_params(colors="black")
    ax.spines[:].set_color(EDGE)
    ax.yaxis.label.set_color("black")
    ax.xaxis.label.set_color("black")
    ax.title.set_color("black")

# Scheme colours and line styles
COLORS = {
    "Plain Hash Token": "#FF4444",
    "Blind Signature":  "#FF8C00",
    "Hybrid Scheme":    "#FFD93D",
    "Ring Signature":   "#4CAF50",
    "Stealth Address":  "#00BCD4",
    "ZKP Token":        "#2196F3",
}
LSTYLE = {
    "Plain Hash Token": "-",
    "Blind Signature":  "--",
    "Hybrid Scheme":    "-.",
    "Ring Signature":   ":",
    "Stealth Address":  (0, (3, 1, 1, 1)),
    "ZKP Token":        (0, (5, 1)),
}

print("="*65)
print("CBDC METADATA DE-ANONYMIZATION RISK MODEL")
print("Nepal Context  |  m=7  |  R_max=28  |  6 Schemes")
print("="*65)


# ===================================================================
# SECTION 1 — CORE MODEL FUNCTIONS
# ===================================================================

def s_single(vals):
    """
    Single-attribute uniqueness score.

    s(a_i) = (1/N) * sum over all users of I_i(u)

    where I_i(u) = 1 if user u is unique on attribute a_i,
                   0 otherwise.

    A user is unique if no other user shares their attribute value.
    Returns a float in [0, 1].
    """
    n      = len(vals)
    counts = pd.Series(vals).value_counts()
    return (counts == 1).sum() / n


def s_joint(a, b):
    """
    Joint uniqueness score for two attributes.

    s(a_i, a_j) = fraction of users who are unique on the
    COMBINATION of attributes a_i and a_j together.

    A user is jointly unique if no other user shares both
    their a_i value AND their a_j value simultaneously.
    Returns a float in [0, 1].
    """
    n      = len(a)
    counts = pd.Series(list(zip(a, b))).value_counts()
    return (counts == 1).sum() / n


def lam(si, sj, sij):
    """
    Pairwise interaction coefficient.

    lambda_ij = s(a_i, a_j) / (s(a_i) * s(a_j))

    Measures amplification beyond statistical independence:
        lambda = 1  -> independent (no amplification)
        lambda > 1  -> amplified risk (combination more identifying)
        lambda < 1  -> suppressed risk (attributes redundant)

    Special case: if s(a_i) * s(a_j) = 0, return 0.0
    because the pair contributes nothing to R_raw regardless.
    Symmetry: lambda_ij = lambda_ji always holds.
    """
    d = si * sj
    return 0.0 if d < 1e-12 else sij / d


def R_raw(s, lmat, attrs):
    """
    Un-normalised risk score.

    R_raw = sum_i s(a_i)
          + sum_{i<j} lambda_ij * s(a_i) * s(a_j)

    First term:  individual attribute leakage (m terms)
    Second term: pairwise amplification (C(m,2) terms)

    The j>i condition via combinations() ensures each pair
    is counted exactly once.
    """
    R = sum(s[a] for a in attrs)
    for ai, aj in combinations(attrs, 2):
        R += lmat[ai].get(aj, 0.0) * s[ai] * s[aj]
    return R


def R_max(m):
    """
    Theoretical maximum of R_raw for m attributes.

    R_max(m) = m + C(m,2) = m + m*(m-1)/2

    Achieved when:
        s(a_i) = 1 for all i         (all users unique on every attr)
        lambda_ij = 1 for all pairs  (all pairs independent)

    Under these conditions each of the m individual terms = 1
    and each of the C(m,2) pair terms = 1 * 1 * 1 = 1.

    Proof that R_raw <= R_max:
        Each pair term = lambda_ij * s_i * s_j = s(a_i, a_j) <= 1
        Each individual term = s(a_i) <= 1
        Sum of maximums = m + C(m,2) = R_max
        Therefore R_raw <= R_max for any valid input.
    """
    return m + m * (m - 1) // 2


def pipeline(df, attrs):
    """
    Full model pipeline for a given dataframe and attribute list.

    Steps:
        1. Compute s(a_i) for each attribute
        2. Compute lambda_ij for each attribute pair
        3. Compute R_raw
        4. Compute R_norm = R_raw / R_max(m)

    Returns a dict with all intermediate and final values.
    """
    m    = len(attrs)
    s    = {a: s_single(df[a].values) for a in attrs}
    lmat = {a: {} for a in attrs}

    for ai, aj in combinations(attrs, 2):
        sij          = s_joint(df[ai].values, df[aj].values)
        l            = lam(s[ai], s[aj], sij)
        lmat[ai][aj] = lmat[aj][ai] = l

    Rr = R_raw(s, lmat, attrs)
    Rn = Rr / R_max(m)

    return {
        "s":      s,
        "lmat":   lmat,
        "R_raw":  Rr,
        "R_norm": Rn,
        "m":      m,
        "R_max":  R_max(m),
    }


# ===================================================================
# SECTION 2 — DATA GENERATION
# ===================================================================
#
# Each scheme's gen() configuration reflects the cryptographic
# properties of that scheme — how it transforms each attribute's
# value domain. This is not arbitrary — it is grounded in the
# published properties of each cryptographic construction.
#
# Nepal-specific attribute distributions (NRB-informed):
#
# payment_channel (4 values):
#   0 = mobile wallet  (60%) — eSewa/Khalti dominance
#   1 = QR code        (25%) — rapid QR adoption at merchants
#   2 = NFC            (10%) — urban smartphone users
#   3 = offline token  ( 5%) — rural low-connectivity deployment
#
# transaction_zone (5 values):
#   0 = Kathmandu valley (35%) — capital digital activity concentration
#   1 = hill urban       (20%) — Pokhara, Biratnagar, Dharan
#   2 = hill rural       (25%) — largest rural population segment
#   3 = Terai urban      (12%) — southern plains urban centres
#   4 = Terai rural      ( 8%) — southern plains rural population
#
# Excluded: remittance_corridor
#   Domestic CBDC scope only. Cross-border remittances use
#   separate settlement rails outside NRB's domestic mandate.

CHANNEL_PROBS = [0.60, 0.25, 0.10, 0.05]
ZONE_PROBS    = [0.35, 0.20, 0.25, 0.12, 0.08]


def gen(N, scheme, seed=42):
    """
    Generate synthetic CBDC transaction metadata for N users
    under a given cryptographic scheme.

    The value domain of each attribute is determined by the
    cryptographic scheme — this models how each scheme
    transforms observable metadata in practice.

    Nepal-specific attributes (payment_channel, transaction_zone)
    are generated identically across all schemes — they are
    observable at the network level regardless of which
    cryptographic scheme is deployed on the token layer.
    """
    rng = np.random.default_rng(seed)

    # Nepal-specific attributes — network-level observable
    channel = rng.choice(4, N, p=CHANNEL_PROBS)
    zone    = rng.choice(5, N, p=ZONE_PROBS)

    if scheme == "Plain Hash Token":
        # All transaction metadata fully exposed.
        # SHA-256 hash of token provides no metadata hiding.
        # Wallet = sequential address, permanently linked.
        # Timestamp = fine-grained Unix-style integer.
        # Amount = exact float value.
        amount    = np.round(rng.exponential(500, N), 2)
        timestamp = rng.integers(0, N * 10, N)
        merchant  = rng.choice(200, N)
        wallet    = np.arange(N)
        frequency = rng.integers(1, 100, N)

    elif scheme == "Blind Signature":
        # Chaum blind signature protocol:
        # - Wallet randomised (bank signs without seeing serial)
        # - Timestamp coarsened to hour of day (24 values)
        # - Amount remains exact (no range proof)
        # - Merchant granularity unchanged
        amount    = np.round(rng.exponential(500, N), 2)
        timestamp = rng.integers(0, 24, N)
        merchant  = rng.choice(200, N)
        wallet    = rng.choice(N * 10, N)
        frequency = rng.integers(1, 50, N)

    elif scheme == "Hybrid Scheme":
        # Combines Blind Signature wallet blinding with
        # Ring Signature amount bucketing.
        # Models the digital euro's proposed tiered privacy design.
        # Sits between Blind Signature and Ring Signature
        # on the risk ladder.
        amount    = np.round(rng.exponential(500, N) / 100) * 100
        timestamp = rng.integers(0, 24, N)
        merchant  = rng.choice(100, N)
        wallet    = rng.choice(N * 10, N)
        frequency = rng.integers(1, 35, N)

    elif scheme == "Ring Signature":
        # Ring anonymity set hides true signer among k members.
        # Amounts bucketed to NPR 100 increments.
        # Merchant coarsened to broad categories (20 values).
        # Wallet drawn from larger pool than Blind Signature.
        amount    = np.round(rng.exponential(500, N) / 100) * 100
        timestamp = rng.integers(0, 24, N)
        merchant  = rng.choice(20, N)
        wallet    = rng.choice(N * 20, N)
        frequency = rng.integers(1, 20, N)

    elif scheme == "Stealth Address":
        # One-time wallet address generated per transaction.
        # Wallet is completely unlinkable across transactions.
        # Critical distinction: wallet unlinkability != wallet
        # randomisation. Each transaction produces a fresh
        # address with no mathematical link to prior addresses.
        # Amounts and timestamps still fully visible (no ZKP).
        amount    = np.round(rng.exponential(500, N), 2)
        timestamp = rng.integers(0, 24, N)
        merchant  = rng.choice(20, N)
        wallet    = np.arange(N) * 997 + rng.integers(0, 100000, N)
        frequency = rng.integers(1, 15, N)

    elif scheme == "ZKP Token":
        # Zero-knowledge proofs hide exact values.
        # All attributes reduced to coarse disclosure bands.
        # Nullifier-based wallet: fresh nullifier per spend,
        # drawn from pool of size N*100.
        # Amount: 5 bands only {0, 100, 500, 1000, 5000}
        # Timestamp: day of week only (7 values)
        # Merchant: 5 broad economic sectors
        amount    = rng.choice([0, 100, 500, 1000, 5000], N)
        timestamp = rng.choice(7, N)
        merchant  = rng.choice(5, N)
        wallet    = rng.choice(N * 100, N)
        frequency = rng.choice(5, N)

    return pd.DataFrame({
        "amount":           amount,
        "timestamp":        timestamp,
        "merchant":         merchant,
        "wallet":           wallet,
        "frequency":        frequency,
        "payment_channel":  channel,
        "transaction_zone": zone,
    })


# ===================================================================
# SECTION 3 — CONFIGURATION
# ===================================================================

SCHEMES = [
    "Plain Hash Token",
    "Blind Signature",
    "Hybrid Scheme",
    "Ring Signature",
    "Stealth Address",
    "ZKP Token",
]

ATTRS_7     = ["amount", "timestamp", "merchant", "wallet",
               "frequency", "payment_channel", "transaction_zone"]
NEPAL_ATTRS = ["payment_channel", "transaction_zone"]
N_BASE      = 1000
LABELS      = [s.replace(" ", "\n") for s in SCHEMES]

# Attribute addition order for m-sensitivity analysis
# wallet first (dominant risk driver), then standard attrs,
# then Nepal-specific attrs
ATTR_ORDER  = ["wallet", "amount", "timestamp", "merchant",
               "frequency", "payment_channel", "transaction_zone"]
ATTR_LABELS = ["wallet", "amount", "timestamp", "merchant",
               "frequency", "pay.\nchannel", "tx zone"]

# Mitigation strategies — multiplicative approach
# Each value is the multiplier applied to s(a_i)
# 1.0 = no mitigation, 0.0 = perfect suppression
MITIGATIONS = {
    "No Mitigation": {
        a: 1.00 for a in ATTRS_7
    },
    "Diff. Privacy\n(Amount)": {
        **{a: 1.00 for a in ATTRS_7},
        "amount": 0.20,
    },
    "Timestamp\nSuppression": {
        **{a: 1.00 for a in ATTRS_7},
        "timestamp": 0.05,
    },
    "Wallet\nUnlinkability": {
        **{a: 1.00 for a in ATTRS_7},
        "wallet": 0.05,
    },
    "Nepal\nControls": {
        **{a: 1.00 for a in ATTRS_7},
        "payment_channel":  0.20,
        "transaction_zone": 0.15,
    },
    "DP +\nTimestamp": {
        **{a: 1.00 for a in ATTRS_7},
        "amount":    0.20,
        "timestamp": 0.05,
    },
    "Full\nStack": {
        "amount":           0.20,
        "timestamp":        0.05,
        "merchant":         0.50,
        "wallet":           0.05,
        "frequency":        0.30,
        "payment_channel":  0.20,
        "transaction_zone": 0.15,
    },
}

print(f"\nConfiguration:")
print(f"  Schemes:    {len(SCHEMES)}")
print(f"  Attributes: m={len(ATTRS_7)}")
print(f"  R_max:      {R_max(len(ATTRS_7))}")
print(f"  N_base:     {N_BASE}")
print(f"  Output:     {OUTPUT_DIR}/\n")


# ===================================================================
# ANALYSIS 3 — SIX-SCHEME COMPARISON  (m=7, N=1,000)
# ===================================================================
print("="*65)
print(f"ANALYSIS 3: Six-Scheme Comparison  "
      f"(m=7, R_max={R_max(7)}, N={N_BASE})")
print("="*65)

res3 = {}
for scheme in SCHEMES:
    df  = gen(N_BASE, scheme)
    res = pipeline(df, ATTRS_7)
    res3[scheme] = res
    print(f"\n  {scheme}")
    for a in ATTRS_7:
        tag = " [Nepal]" if a in NEPAL_ATTRS else ""
        print(f"    s({a:<22}) = {res['s'][a]:.4f}{tag}")
    print(f"  R_raw={res['R_raw']:.4f}  "
          f"R_norm={res['R_norm']:.4f}  "
          f"({res['R_norm']*100:.1f}% of maximum)")

R_norm_vals = [res3[s]["R_norm"] for s in SCHEMES]

# ── Chart A3a: R_norm bar + attribute contributions ───────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 7))
fig.patch.set_facecolor(BG)
plt.subplots_adjust(hspace=0.35, wspace=0.35)
for ax in axes:
    sax(ax)

# R_norm bar chart
bars = axes[0].bar(
    range(len(SCHEMES)), R_norm_vals,
    color=[COLORS[s] for s in SCHEMES],
    edgecolor=EDGE, linewidth=1.2, width=0.6)
axes[0].set_xticks(range(len(SCHEMES)))
axes[0].set_xticklabels(LABELS, color="white", fontsize=10)
axes[0].set_ylabel("R_norm  [0, 1]", fontsize=12)
axes[0].set_title(
    "Normalised Risk R_norm\n6 Schemes, m=7, N=1,000",
    fontsize=12, fontweight="bold", pad=8)
axes[0].set_ylim(0, 1.0)
axes[0].axhline(0.5, color="yellow", lw=1, ls=":", alpha=0.6)
for bar, val in zip(bars, R_norm_vals):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.008, f"{val:.3f}",
        ha="center", color="white",
        fontsize=9, fontweight="bold")
axes[0].grid(axis="y", alpha=0.15, color="white")

# Stacked attribute contributions
orig_colors  = ["#FF6B6B", "#FFD93D", "#6BCB77",
                "#4D96FF", "#C77DFF"]
nepal_colors = ["#80DEEA", "#A5D6A7"]
all_colors   = orig_colors + nepal_colors
s_data = np.array([
    [res3[s]["s"][a] for a in ATTRS_7]
    for s in SCHEMES])
bottom = np.zeros(len(SCHEMES))
for i, (attr, col) in enumerate(zip(ATTRS_7, all_colors)):
    label = attr + (" [Nepal]" if attr in NEPAL_ATTRS else "")
    axes[1].bar(
        range(len(SCHEMES)), s_data[:, i],
        bottom=bottom, color=col,
        label=label, edgecolor=BG, linewidth=0.4)
    bottom += s_data[:, i]
axes[1].set_xticks(range(len(SCHEMES)))
axes[1].set_xticklabels(LABELS, color="white", fontsize=10)
axes[1].set_ylabel("Cumulative s(ai)", fontsize=12)
axes[1].set_title(
    "Attribute Contributions\n([Nepal] = Nepal-specific)",
    fontsize=12, fontweight="bold", pad=8)
leg = axes[1].legend(
    title="Attribute", facecolor=AX, edgecolor=EDGE,
    labelcolor="white", title_fontsize=10,
    fontsize=7, loc="upper right")
leg.get_title().set_color("white")
axes[1].grid(axis="y", alpha=0.15, color="white")

# Nepal attributes isolated
nepal_s = np.array([
    [res3[s]["s"][a] for a in NEPAL_ATTRS]
    for s in SCHEMES])
nb = np.zeros(len(SCHEMES))
for i, (attr, col) in enumerate(zip(NEPAL_ATTRS, nepal_colors)):
    axes[2].bar(
        range(len(SCHEMES)), nepal_s[:, i],
        bottom=nb, color=col,
        label=attr, edgecolor=BG, linewidth=0.4)
    nb += nepal_s[:, i]
axes[2].set_xticks(range(len(SCHEMES)))
axes[2].set_xticklabels(LABELS, color="white", fontsize=10)
axes[2].set_ylabel("Cumulative s(ai)", fontsize=12)
axes[2].set_title(
    "Nepal-Specific Attributes\n(payment_channel + transaction_zone)",
    fontsize=12, fontweight="bold", pad=8)
leg2 = axes[2].legend(
    title="Nepal Attr.", facecolor=AX, edgecolor=EDGE,
    labelcolor="white", title_fontsize=10, fontsize=10)
leg2.get_title().set_color("white")
axes[2].grid(axis="y", alpha=0.15, color="white")

plt.suptitle(
    "Analysis 3: Six-Scheme Comparison  |  "
    "m=7 (5 standard + 2 Nepal)  |  R_max=28  |  N=1,000",
    color="white", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout(pad=2)
plt.savefig(
    f"{OUTPUT_DIR}/a3_comparison.png",
    dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("\n[Chart] a3_comparison.png saved")

# ── Chart A3b: Lambda heatmaps ────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(20, 14))
fig.patch.set_facecolor(BG)
plt.subplots_adjust(hspace=0.4, wspace=0.4)
fig.suptitle(
    "Analysis 3: λ_ij Interaction Heatmaps — All 6 Schemes  (m=7)",
    color="black", fontsize=14, fontweight="bold", y=1.01)

for ax, scheme in zip(axes.flatten(), SCHEMES):
    mat = np.zeros((len(ATTRS_7), len(ATTRS_7)))
    for i, ai in enumerate(ATTRS_7):
        for j, aj in enumerate(ATTRS_7):
            if i != j:
                mat[i][j] = min(
                    res3[scheme]["lmat"].get(
                        ai, {}).get(aj, 0.0), 100)
    ax.set_facecolor(AX)
    sns.heatmap(
        mat, ax=ax,
        xticklabels=ATTRS_7, yticklabels=ATTRS_7,
        annot=True, fmt=".1f", cmap="YlOrRd",
        linewidths=0.5, linecolor=BG,
        annot_kws={"size": 9, "color": "black"})
    ax.set_title(
        scheme, color="black",
        fontsize=12, fontweight="bold", pad=6)
    ax.tick_params(colors="black", labelsize=9)
    ax.set_xticklabels(
        ATTRS_7, color="black",
        rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(
        ATTRS_7, color="black",
        rotation=0, fontsize=9)

for ax, scheme in zip(axes.flatten(), SCHEMES):
    mat = np.zeros((len(ATTRS_7), len(ATTRS_7)))
    for i, ai in enumerate(ATTRS_7):
        for j, aj in enumerate(ATTRS_7):
            if i != j:
                mat[i][j] = min(
                    res3[scheme]["lmat"].get(
                        ai, {}).get(aj, 0.0), 100)
    ax.set_facecolor(AX)
    sns.heatmap(
        mat, ax=ax,
        xticklabels=ATTRS_7, yticklabels=ATTRS_7,
        annot=True, fmt=".1f", cmap="YlOrRd",
        linewidths=0.5, linecolor=BG,
        annot_kws={"size": 7, "color": "black"})
    ax.set_title(
        scheme, color="white",
        fontsize=12, fontweight="bold", pad=6)
    ax.tick_params(colors="white", labelsize=7)
    ax.set_xticklabels(
        ATTRS_7, color="white",
        rotation=35, ha="right", fontsize=7)
    ax.set_yticklabels(
        ATTRS_7, color="white",
        rotation=0, fontsize=7)

plt.tight_layout(pad=2)
plt.savefig(
    f"{OUTPUT_DIR}/a3_lambda.png",
    dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("[Chart] a3_lambda.png saved")


# ===================================================================
# ANALYSIS 5 — POPULATION SCALABILITY  (m=7, N=100 to N=100K)
# ===================================================================
print("\n" + "="*65)
print("ANALYSIS 5: Population Scalability  (m=7, all 6 schemes)")
print("="*65)

N_vals     = [100, 250, 500, 1000, 2000,
              5000, 10000, 25000, 50000, 100000]
scale_norm = {s: [] for s in SCHEMES}

for scheme in SCHEMES:
    for N in N_vals:
        df  = gen(N, scheme)
        res = pipeline(df, ATTRS_7)
        scale_norm[scheme].append(res["R_norm"])
    print(f"  {scheme}: "
          f"N=100→{scale_norm[scheme][0]:.3f}  "
          f"N=1K→{scale_norm[scheme][3]:.3f}  "
          f"N=100K→{scale_norm[scheme][-1]:.3f}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor(BG)
for ax in axes.flatten():
    sax(ax)

# R_norm vs N
for scheme in SCHEMES:
    axes[0, 0].plot(
        N_vals, scale_norm[scheme],
        color=COLORS[scheme], linestyle=LSTYLE[scheme],
        lw=2.5, marker="o", ms=4, label=scheme)
axes[0, 0].set_xscale("log")
axes[0, 0].set_ylim(0, 1.0)
axes[0, 0].axhline(0.5, color="yellow", lw=1, ls=":", alpha=0.6)
axes[0, 0].set_xlabel("Population N (log scale)", fontsize=12)
axes[0, 0].set_ylabel("R_norm  [0, 1]", fontsize=12)
axes[0, 0].set_title(
    "R_norm vs Population Size",
    fontsize=12, fontweight="bold", pad=8)
axes[0, 0].grid(alpha=0.15, color="white")
axes[0, 0].legend(
    facecolor=AX, edgecolor=EDGE,
    labelcolor="white", fontsize=10)

# Convergence rate
for scheme in SCHEMES:
    delta = [
        abs(scale_norm[scheme][i + 1] - scale_norm[scheme][i])
        for i in range(len(N_vals) - 1)]
    mid = [
        (N_vals[i] * N_vals[i + 1]) ** 0.5
        for i in range(len(N_vals) - 1)]
    axes[0, 1].plot(
        mid, delta,
        color=COLORS[scheme], linestyle=LSTYLE[scheme],
        lw=2.5, marker="s", ms=4, label=scheme)
axes[0, 1].set_xscale("log")
axes[0, 1].set_yscale("log")
axes[0, 1].set_xlabel("Population N (log scale)", fontsize=12)
axes[0, 1].set_ylabel("|ΔR_norm| (log scale)", fontsize=12)
axes[0, 1].set_title(
    "Convergence Rate",
    fontsize=12, fontweight="bold", pad=8)
axes[0, 1].grid(alpha=0.15, color="white")
axes[0, 1].legend(
    facecolor=AX, edgecolor=EDGE,
    labelcolor="white", fontsize=10)

# Residual risk floor at N=100K
floors = [scale_norm[s][-1] for s in SCHEMES]
bars   = axes[1, 0].bar(
    range(len(SCHEMES)), floors,
    color=[COLORS[s] for s in SCHEMES],
    edgecolor=EDGE, lw=1.2, width=0.6)
axes[1, 0].set_xticks(range(len(SCHEMES)))
axes[1, 0].set_xticklabels(LABELS, color="white", fontsize=10)
axes[1, 0].set_ylabel("R_norm at N=100,000", fontsize=12)
axes[1, 0].set_title(
    "Residual Risk Floor at Scale",
    fontsize=12, fontweight="bold", pad=8)
axes[1, 0].set_ylim(0, max(floors) * 1.35)
for bar, val in zip(bars, floors):
    axes[1, 0].text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.003, f"{val:.3f}",
        ha="center", color="white",
        fontsize=9, fontweight="bold")
axes[1, 0].grid(axis="y", alpha=0.15, color="white")

# Relative risk vs ZKP baseline
zkp_floor = scale_norm["ZKP Token"][-1]
relative  = [scale_norm[s][-1] / zkp_floor for s in SCHEMES]
bars2     = axes[1, 1].bar(
    range(len(SCHEMES)), relative,
    color=[COLORS[s] for s in SCHEMES],
    edgecolor=EDGE, lw=1.2, width=0.6)
axes[1, 1].set_xticks(range(len(SCHEMES)))
axes[1, 1].set_xticklabels(LABELS, color="white", fontsize=10)
axes[1, 1].set_ylabel("Risk relative to ZKP Token", fontsize=12)
axes[1, 1].set_title(
    "Relative Risk Floor  (ZKP = 1.0 baseline)",
    fontsize=12, fontweight="bold", pad=8)
axes[1, 1].axhline(1.0, color="cyan", lw=1, ls=":", alpha=0.6)
for bar, val in zip(bars2, relative):
    axes[1, 1].text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.04, f"{val:.1f}×",
        ha="center", color="white",
        fontsize=9, fontweight="bold")
axes[1, 1].grid(axis="y", alpha=0.15, color="white")

plt.suptitle(
    "Analysis 5: Population Scalability  |  6 Schemes  |  m=7",
    color="white", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout(pad=2)
plt.savefig(
    f"{OUTPUT_DIR}/a5_scalability.png",
    dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("[Chart] a5_scalability.png saved")


# ===================================================================
# ANALYSIS 7 — MITIGATION EFFECTIVENESS  (m=7, N=1,000)
# ===================================================================
print("\n" + "="*65)
print("ANALYSIS 7: Mitigation Effectiveness  (m=7)")
print("="*65)

mit_norm = {s: {} for s in SCHEMES}

for scheme in SCHEMES:
    df     = gen(N_BASE, scheme)
    base_s = {a: s_single(df[a].values) for a in ATTRS_7}

    for mname, mults in MITIGATIONS.items():
        s_mit = {a: base_s[a] * mults[a] for a in ATTRS_7}
        lm    = {a: {} for a in ATTRS_7}
        for ai, aj in combinations(ATTRS_7, 2):
            sij        = (s_joint(df[ai].values, df[aj].values)
                          * mults[ai] * mults[aj])
            l          = lam(s_mit[ai], s_mit[aj], sij)
            lm[ai][aj] = lm[aj][ai] = l
        Rr = R_raw(s_mit, lm, ATTRS_7)
        mit_norm[scheme][mname] = Rr / R_max(len(ATTRS_7))

    base = mit_norm[scheme]["No Mitigation"]
    full = mit_norm[scheme]["Full\nStack"]
    print(f"  {scheme}: "
          f"{base:.3f} → {full:.3f}  "
          f"({(base - full) / base * 100:.1f}% reduction)")

mit_names = list(MITIGATIONS.keys())
x         = np.arange(len(mit_names))
w         = 1.0 / (len(SCHEMES) + 1)
offsets   = np.linspace(
    -(len(SCHEMES) - 1) / 2,
     (len(SCHEMES) - 1) / 2,
    len(SCHEMES))

fig, axes = plt.subplots(1, 2, figsize=(20, 7))
fig.patch.set_facecolor(BG)
for ax in axes:
    sax(ax)

for i, scheme in enumerate(SCHEMES):
    vals = [mit_norm[scheme][m] for m in mit_names]
    axes[0].bar(
        x + offsets[i] * w, vals,
        width=w * 0.9,
        color=COLORS[scheme], label=scheme,
        edgecolor=BG, lw=0.3, alpha=0.92)
axes[0].set_xticks(x)
axes[0].set_xticklabels(
    mit_names, color="white", fontsize=10)
axes[0].set_ylabel("R_norm  [0, 1]", fontsize=12)
axes[0].set_title(
    "R_norm by Mitigation Strategy",
    fontsize=12, fontweight="bold", pad=8)
axes[0].set_ylim(0, 1.0)
axes[0].axhline(0.5, color="yellow", lw=1, ls=":", alpha=0.5)
axes[0].grid(axis="y", alpha=0.15, color="white")
axes[0].legend(
    facecolor=AX, edgecolor=EDGE,
    labelcolor="white", fontsize=10, ncol=2)

red_data = []
for scheme in SCHEMES:
    base = mit_norm[scheme]["No Mitigation"]
    row  = [
        (base - mit_norm[scheme][m]) / base * 100
        for m in mit_names[1:]]
    red_data.append(row)

red_df = pd.DataFrame(
    red_data, index=SCHEMES,
    columns=mit_names[1:])
sns.heatmap(
    red_df, ax=axes[1],
    annot=True, fmt=".1f",
    cmap="RdYlGn",
    linewidths=0.5, linecolor=BG,
    annot_kws={"size": 9, "color": "black"})
axes[1].set_title(
    "Risk Reduction (%) per Mitigation per Scheme",
    fontsize=12, fontweight="bold", pad=8)
axes[1].tick_params(colors="white", labelsize=9)
axes[1].set_xticklabels(
    axes[1].get_xticklabels(),
    color="white", rotation=25, ha="right")
axes[1].set_yticklabels(
    SCHEMES, color="white", rotation=0)

plt.suptitle(
    "Analysis 7: Mitigation Effectiveness  |  "
    "6 Schemes  |  m=7  |  Nepal Controls Included",
    color="white", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout(pad=2)
plt.savefig(
    f"{OUTPUT_DIR}/a7_mitigation.png",
    dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("[Chart] a7_mitigation.png saved")


# ===================================================================
# ANALYSIS 8 — m-ATTRIBUTE SENSITIVITY  (m=1 to m=7)
# ===================================================================
print("\n" + "="*65)
print("ANALYSIS 8: m-Attribute Sensitivity  (m=1 to m=7)")
print("="*65)

m_results = {
    s: {"R_norm": [], "R_raw": [], "R_max": []}
    for s in SCHEMES}

for scheme in SCHEMES:
    df = gen(N_BASE, scheme)
    for m_end in range(1, len(ATTR_ORDER) + 1):
        attrs_sub = ATTR_ORDER[:m_end]
        res       = pipeline(df, attrs_sub)
        m_results[scheme]["R_norm"].append(res["R_norm"])
        m_results[scheme]["R_raw"].append(res["R_raw"])
        m_results[scheme]["R_max"].append(res["R_max"])
    print(f"  {scheme}: "
          f"m=1→{m_results[scheme]['R_norm'][0]:.3f}  "
          f"m=5→{m_results[scheme]['R_norm'][4]:.3f}  "
          f"m=7→{m_results[scheme]['R_norm'][6]:.3f}")

fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.patch.set_facecolor(BG)
plt.subplots_adjust(hspace=0.35, wspace=0.35)
for ax in axes:
    sax(ax)

m_vals = list(range(1, len(ATTR_ORDER) + 1))

for scheme in SCHEMES:
    axes[0].plot(
        m_vals, m_results[scheme]["R_norm"],
        color=COLORS[scheme], linestyle=LSTYLE[scheme],
        lw=2.5, marker="o", ms=5, label=scheme)
axes[0].set_xticks(m_vals)
axes[0].set_xticklabels(
    ATTR_LABELS, color="white",
    fontsize=9, rotation=25, ha="right")
axes[0].set_ylabel("R_norm  [0, 1]", fontsize=12)
axes[0].set_title(
    "R_norm as Attributes Added\n"
    "(← standard 5  |  Nepal 2 →)",
    fontsize=12, fontweight="bold", pad=8)
axes[0].axvline(5.5, color="cyan", lw=1.5, ls="--", alpha=0.7)
axes[0].text(5.6, 0.85, "Nepal\nattrs",
             color="cyan", fontsize=10)
axes[0].set_ylim(0, 1.05)
axes[0].grid(alpha=0.15, color="white")
axes[0].legend(
    facecolor=AX, edgecolor=EDGE,
    labelcolor="white", fontsize=10)

r_max_vals = [R_max(m) for m in m_vals]
axes[1].plot(
    m_vals, r_max_vals,
    color="#4D96FF", lw=2.5, marker="o", ms=7)
for m, rmax in zip(m_vals, r_max_vals):
    axes[1].annotate(
        f"R_max={rmax}", (m, rmax),
        textcoords="offset points", xytext=(4, 5),
        color="white", fontsize=10)
axes[1].set_xticks(m_vals)
axes[1].set_xticklabels(
    ATTR_LABELS, color="white",
    fontsize=9, rotation=25, ha="right")
axes[1].set_ylabel("R_max = m + C(m,2)", fontsize=12)
axes[1].set_title(
    "R_max Growth as m Increases\n(normalisation ceiling)",
    fontsize=12, fontweight="bold", pad=8)
axes[1].axvline(5.5, color="cyan", lw=1.5, ls="--", alpha=0.7)
axes[1].grid(alpha=0.15, color="white")

for scheme in SCHEMES:
    marginal = [
        m_results[scheme]["R_norm"][i] -
        (m_results[scheme]["R_norm"][i - 1] if i > 0 else 0)
        for i in range(len(m_vals))]
    axes[2].plot(
        m_vals, marginal,
        color=COLORS[scheme], linestyle=LSTYLE[scheme],
        lw=2.5, marker="s", ms=4, label=scheme)
axes[2].set_xticks(m_vals)
axes[2].set_xticklabels(
    ATTR_LABELS, color="white",
    fontsize=9, rotation=25, ha="right")
axes[2].set_ylabel("ΔR_norm per attribute added", fontsize=12)
axes[2].set_title(
    "Marginal Risk per Attribute\n"
    "(contribution of each addition)",
    fontsize=12, fontweight="bold", pad=8)
axes[2].axvline(5.5, color="cyan", lw=1.5, ls="--", alpha=0.7)
axes[2].grid(alpha=0.15, color="white")
axes[2].legend(
    facecolor=AX, edgecolor=EDGE,
    labelcolor="white", fontsize=10)

plt.suptitle(
    "Analysis 8: m-Attribute Sensitivity  |  "
    "6 Schemes  |  Cyan = Nepal attributes begin",
    color="white", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout(pad=2)
plt.savefig(
    f"{OUTPUT_DIR}/a8_m_sensitivity.png",
    dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("[Chart] a8_m_sensitivity.png saved")


# ===================================================================
# ANALYSIS 9 — SEED SENSITIVITY  (30 seeds, m=7, N=1,000)
# ===================================================================
print("\n" + "="*65)
print("ANALYSIS 9: Seed Sensitivity  (30 seeds, m=7, N=1,000)")
print("="*65)

N_SEEDS      = 30
seed_results = {s: [] for s in SCHEMES}

for scheme in SCHEMES:
    for seed in range(N_SEEDS):
        df  = gen(N_BASE, scheme, seed=seed)
        res = pipeline(df, ATTRS_7)
        seed_results[scheme].append(res["R_norm"])
    mu  = np.mean(seed_results[scheme])
    std = np.std(seed_results[scheme])
    print(f"  {scheme}: "
          f"mean={mu:.4f}  std={std:.4f}  "
          f"CV={std / mu * 100:.1f}%  "
          f"95% CI=[{mu - 2*std:.4f}, {mu + 2*std:.4f}]")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor(BG)
for ax in axes:
    sax(ax)

means = [np.mean(seed_results[s]) for s in SCHEMES]
stds  = [np.std(seed_results[s])  for s in SCHEMES]

axes[0].bar(
    range(len(SCHEMES)), means,
    yerr=[2 * s for s in stds],
    color=[COLORS[s] for s in SCHEMES],
    edgecolor=EDGE, lw=1.2, width=0.6,
    error_kw={"ecolor": "white", "capsize": 5, "lw": 2})
axes[0].set_xticks(range(len(SCHEMES)))
axes[0].set_xticklabels(LABELS, color="white", fontsize=10)
axes[0].set_ylabel("R_norm  (mean ± 2σ)", fontsize=12)
axes[0].set_title(
    "R_norm Stability Across 30 Seeds\n(error bars = ±2σ)",
    fontsize=12, fontweight="bold", pad=8)
axes[0].set_ylim(0, 1.0)
for i, (mu, sd) in enumerate(zip(means, stds)):
    axes[0].text(
        i, mu + 2 * sd + 0.012,
        f"σ={sd:.4f}", ha="center",
        color="white", fontsize=10)
axes[0].grid(axis="y", alpha=0.15, color="white")

seed_data = [seed_results[s] for s in SCHEMES]
vp = axes[1].violinplot(
    seed_data,
    positions=range(len(SCHEMES)),
    showmeans=True, showmedians=True)
for body, scheme in zip(vp["bodies"], SCHEMES):
    body.set_facecolor(COLORS[scheme])
    body.set_alpha(0.7)
vp["cmeans"].set_color("white")
vp["cmedians"].set_color("yellow")
axes[1].set_xticks(range(len(SCHEMES)))
axes[1].set_xticklabels(LABELS, color="white", fontsize=10)
axes[1].set_ylabel("R_norm distribution", fontsize=12)
axes[1].set_title(
    "Distribution Across 30 Seeds\n(white=mean, yellow=median)",
    fontsize=12, fontweight="bold", pad=8)
axes[1].set_ylim(0, 1.0)
axes[1].grid(axis="y", alpha=0.15, color="white")

plt.suptitle(
    "Analysis 9: Seed Sensitivity  |  30 Seeds  |  m=7  |  N=1,000",
    color="white", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout(pad=2)
plt.savefig(
    f"{OUTPUT_DIR}/a9_seed_sensitivity.png",
    dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("[Chart] a9_seed_sensitivity.png saved")


# ===================================================================
# ANALYSIS 10 — MICRO-SCALE POPULATION TEST
# (Village-Level CBDC Pilot, Nepal context)
# ===================================================================
print("\n" + "="*65)
print("ANALYSIS 10: Micro-Scale Population Test")
print("(Village-Level CBDC Pilot — N=10 to N=1,000)")
print("="*65)

N_micro    = [10, 25, 50, 75, 100, 250, 500, 1000]
micro_norm = {s: [] for s in SCHEMES}

for scheme in SCHEMES:
    for N in N_micro:
        df  = gen(N, scheme)
        res = pipeline(df, ATTRS_7)
        micro_norm[scheme].append(res["R_norm"])
    print(f"  {scheme}: "
          f"N=10→{micro_norm[scheme][0]:.3f}  "
          f"N=50→{micro_norm[scheme][2]:.3f}  "
          f"N=100→{micro_norm[scheme][4]:.3f}  "
          f"N=1K→{micro_norm[scheme][7]:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor(BG)
for ax in axes:
    sax(ax)

for scheme in SCHEMES:
    axes[0].plot(
        N_micro, micro_norm[scheme],
        color=COLORS[scheme], linestyle=LSTYLE[scheme],
        lw=2.5, marker="o", ms=6, label=scheme)
axes[0].set_xticks(N_micro)
axes[0].set_xticklabels(
    [str(n) for n in N_micro],
    color="black", fontsize=9, rotation=45, ha="right")
axes[0].set_ylabel("R_norm  [0, 1]", fontsize=12)
axes[0].set_xlabel("Population N", fontsize=12)
axes[0].set_title(
    "R_norm at Micro-Scale\n(Village Pilot — Nepal VDC Context)",
    fontsize=12, fontweight="bold", pad=8)
axes[0].set_ylim(0, 1.05)
axes[0].axvline(
    50, color="yellow", lw=1.5,
    ls="--", alpha=0.7)
axes[0].text(
    52, 0.9, "N=50\nVDC pilot",
    color="yellow", fontsize=10)
axes[0].grid(alpha=0.15, color="white")
axes[0].legend(
    facecolor=AX, edgecolor=EDGE,
    labelcolor="white", fontsize=10)

vals_50 = [micro_norm[s][2] for s in SCHEMES]
bars = axes[1].bar(
    range(len(SCHEMES)), vals_50,
    color=[COLORS[s] for s in SCHEMES],
    edgecolor=EDGE, lw=1.2, width=0.6)
axes[1].set_xticks(range(len(SCHEMES)))
axes[1].set_xticklabels(LABELS, color="white", fontsize=10)
axes[1].set_ylabel("R_norm at N=50", fontsize=12)
axes[1].set_title(
    "Risk at VDC Pilot Scale\n(N=50 users)",
    fontsize=12, fontweight="bold", pad=8)
axes[1].set_ylim(0, 1.05)
for bar, val in zip(bars, vals_50):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.01, f"{val:.3f}",
        ha="center", color="white",
        fontsize=9, fontweight="bold")
axes[1].grid(axis="y", alpha=0.15, color="white")

plt.suptitle(
    "Analysis 10: Micro-Scale Population Test  |  "
    "Village-Level CBDC Pilot  |  m=7",
    color="white", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout(pad=2)
plt.savefig(
    f"{OUTPUT_DIR}/a10_microscale.png",
    dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("[Chart] a10_microscale.png saved")


# ===================================================================
# ANALYSIS 11 — LAMBDA SENSITIVITY TEST
# Effect of interaction coefficient on R_norm
# ===================================================================
print("\n" + "="*65)
print("ANALYSIS 11: Lambda Sensitivity Test  (λ = 0 to 100)")
print("="*65)

lambda_vals = [0, 0.5, 1, 2, 5, 10, 25, 50, 75, 100]

# Fix s values at Plain Hash Token baseline (seed=42, N=1000)
# amount and wallet are the dominant pair in Plain Hash
s_ph_amount   = res3["Plain Hash Token"]["s"]["amount"]
s_ph_wallet   = res3["Plain Hash Token"]["s"]["wallet"]
s_ph_others   = {
    a: res3["Plain Hash Token"]["s"][a]
    for a in ATTRS_7
    if a not in ["amount", "wallet"]
}

# Fix s values at ZKP Token baseline
s_zkp_wallet  = res3["ZKP Token"]["s"]["wallet"]
s_zkp_others  = {
    a: res3["ZKP Token"]["s"][a]
    for a in ATTRS_7
    if a != "wallet"
}


def R_norm_manual_lambda(lam_pair, s_a, s_b,
                          s_others, pair_attrs,
                          all_attrs):
    """
    Compute R_norm with manually overridden lambda
    for one specific attribute pair.

    All other lambda values remain at their natural values
    (derived from data, approximately 1.0 for independent attrs).

    Parameters:
        lam_pair    : the lambda value to set for the pair
        s_a, s_b    : s values for the two attributes in the pair
        s_others    : dict of s values for all other attributes
        pair_attrs  : list of two attribute names for the pair
        all_attrs   : full list of all attribute names
    """
    s_all = {pair_attrs[0]: s_a,
             pair_attrs[1]: s_b,
             **s_others}

    lmat = {a: {} for a in all_attrs}
    for ai, aj in combinations(all_attrs, 2):
        if set([ai, aj]) == set(pair_attrs):
            lmat[ai][aj] = lmat[aj][ai] = lam_pair
        else:
            lmat[ai][aj] = lmat[aj][ai] = 1.0

    Rr = R_raw(s_all, lmat, all_attrs)
    return Rr / R_max(len(all_attrs))


# Plain Hash: vary lambda for (amount, wallet) pair
r_norm_ph = []
for lv in lambda_vals:
    rn = R_norm_manual_lambda(
        lv, s_ph_amount, s_ph_wallet,
        s_ph_others,
        ["amount", "wallet"], ATTRS_7)
    r_norm_ph.append(rn)
    print(f"  Plain Hash  λ(amount,wallet)={lv:>6.1f}"
          f"  →  R_norm={rn:.4f}")

print()

# ZKP: vary lambda for (wallet, timestamp) pair
# wallet is dominant, timestamp is next
s_zkp_ts = s_zkp_others.get("timestamp", 0.0)
s_zkp_others_no_ts = {
    k: v for k, v in s_zkp_others.items()
    if k != "timestamp"}
r_norm_zkp = []
for lv in lambda_vals:
    rn = R_norm_manual_lambda(
        lv, s_zkp_wallet, s_zkp_ts,
        s_zkp_others_no_ts,
        ["wallet", "timestamp"], ATTRS_7)
    r_norm_zkp.append(rn)
    print(f"  ZKP Token   λ(wallet,timestamp)={lv:>6.1f}"
          f"  →  R_norm={rn:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor(BG)
for ax in axes:
    sax(ax)

# Plain Hash lambda sensitivity
axes[0].plot(
    lambda_vals, r_norm_ph,
    color=COLORS["Plain Hash Token"],
    lw=2.5, marker="o", ms=6,
    label="Plain Hash Token")
axes[0].axhline(
    r_norm_ph[2], color="yellow",
    lw=1, ls="--", alpha=0.6,
    label=f"λ=1 baseline  ({r_norm_ph[2]:.3f})")
axes[0].axhline(
    1.0, color="red",
    lw=1, ls=":", alpha=0.5,
    label="Theoretical maximum")
axes[0].set_xlabel("λ(amount, wallet)", fontsize=12)
axes[0].set_ylabel("R_norm  [0, 1]", fontsize=12)
axes[0].set_title(
    "R_norm Sensitivity to λ\n"
    "Plain Hash Token — amount × wallet pair",
    fontsize=12, fontweight="bold", pad=8)
axes[0].set_ylim(0, 1.05)
axes[0].grid(alpha=0.15, color="white")
axes[0].legend(
    facecolor=AX, edgecolor=EDGE,
    labelcolor="white", fontsize=10)

# ZKP lambda sensitivity
axes[1].plot(
    lambda_vals, r_norm_zkp,
    color=COLORS["ZKP Token"],
    lw=2.5, marker="o", ms=6,
    label="ZKP Token")
axes[1].axhline(
    r_norm_zkp[2], color="yellow",
    lw=1, ls="--", alpha=0.6,
    label=f"λ=1 baseline  ({r_norm_zkp[2]:.3f})")
axes[1].axhline(
    1.0, color="red",
    lw=1, ls=":", alpha=0.5,
    label="Theoretical maximum")
axes[1].set_xlabel("λ(wallet, timestamp)", fontsize=12)
axes[1].set_ylabel("R_norm  [0, 1]", fontsize=12)
axes[1].set_title(
    "R_norm Sensitivity to λ\n"
    "ZKP Token — wallet × timestamp pair",
    fontsize=12, fontweight="bold", pad=8)
axes[1].set_ylim(0, 1.05)
axes[1].grid(alpha=0.15, color="white")
axes[1].legend(
    facecolor=AX, edgecolor=EDGE,
    labelcolor="white", fontsize=10)

plt.suptitle(
    "Analysis 11: Lambda Sensitivity  |  "
    "Effect of Interaction Coefficient on R_norm  |  m=7",
    color="white", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout(pad=2)
plt.savefig(
    f"{OUTPUT_DIR}/a11_lambda_sensitivity.png",
    dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("\n[Chart] a11_lambda_sensitivity.png saved")


# ===================================================================
# ANALYSIS 12 — WORST-CASE SCENARIO
# R_norm approaching the theoretical maximum of 1.0
# ===================================================================
print("\n" + "="*65)
print("ANALYSIS 12: Worst-Case Scenario")
print("(R_norm approaching theoretical maximum)")
print("="*65)


def gen_worst_case(N, seed=42):
    """
    Worst-case CBDC configuration.

    Every attribute set to maximum uniqueness:
    - amount:           exact float exponential
                        (nearly every user unique)
    - timestamp:        unique integer per transaction
                        (every user unique)
    - merchant:         unique per user
                        (every user unique)
    - wallet:           sequential unique address
                        (every user unique)
    - frequency:        unique integer per user
                        (every user unique)
    - payment_channel:  unique per user
                        (unrealistic but represents worst case)
    - transaction_zone: unique per user
                        (unrealistic but represents worst case)

    This configuration represents a completely negligent CBDC
    deployment with no privacy engineering whatsoever.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "amount":           np.round(
                                rng.exponential(500, N), 2),
        "timestamp":        np.arange(N) * 1000
                            + rng.integers(0, 100, N),
        "merchant":         np.arange(N),
        "wallet":           np.arange(N),
        "frequency":        np.arange(1, N + 1),
        "payment_channel":  np.arange(N),
        "transaction_zone": np.arange(N),
    })


df_wc  = gen_worst_case(N_BASE)
res_wc = pipeline(df_wc, ATTRS_7)

print(f"\n  Worst-Case Configuration:")
for a in ATTRS_7:
    print(f"    s({a:<22}) = {res_wc['s'][a]:.4f}")
print(f"\n  R_raw  = {res_wc['R_raw']:.4f}")
print(f"  R_max  = {res_wc['R_max']}")
print(f"  R_norm = {res_wc['R_norm']:.4f}  "
      f"({res_wc['R_norm']*100:.1f}% of theoretical maximum)")
print(f"\n  Theoretical maximum = 1.0000  (100.0%)")
print(f"  Gap to maximum      = "
      f"{1.0 - res_wc['R_norm']:.4f}  "
      f"({(1.0 - res_wc['R_norm'])*100:.1f}%)")

all_labels   = SCHEMES + ["Worst\nCase", "Theoretical\nMaximum"]
all_r_norms  = ([res3[s]["R_norm"] for s in SCHEMES]
                + [res_wc["R_norm"], 1.0])
all_colors_w = ([COLORS[s] for s in SCHEMES]
                + ["#FF00FF", "#FFFFFF"])

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.patch.set_facecolor(BG)
for ax in axes:
    sax(ax)

bars = axes[0].bar(
    range(len(all_labels)), all_r_norms,
    color=all_colors_w,
    edgecolor=EDGE, lw=1.2, width=0.6)
axes[0].set_xticks(range(len(all_labels)))
axes[0].set_xticklabels(
    all_labels, color="black", fontsize=10, rotation=45, ha="right")
axes[0].set_ylabel("R_norm  [0, 1]", fontsize=12)
axes[0].set_title(
    "All Schemes vs Worst Case\nvs Theoretical Maximum",
    fontsize=12, fontweight="bold", pad=8)
axes[0].set_ylim(0, 1.15)
axes[0].axhline(
    1.0, color="red", lw=1.5,
    ls="--", alpha=0.7,
    label="Theoretical maximum (R_norm=1.0)")
for bar, val in zip(bars, all_r_norms):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.01, f"{val:.3f}",
        ha="center", color="white",
        fontsize=10, fontweight="bold")
axes[0].grid(axis="y", alpha=0.15, color="white")
axes[0].legend(
    facecolor=AX, edgecolor=EDGE,
    labelcolor="white", fontsize=10)

# s(ai) comparison — worst case vs Plain Hash vs ZKP
x_pos           = np.arange(len(ATTRS_7))
w_bar           = 0.25
schemes_compare = ["Plain Hash Token", "ZKP Token"]
colors_compare  = [
    COLORS["Plain Hash Token"],
    COLORS["ZKP Token"],
    "#FF00FF"]
labels_compare  = [
    "Plain Hash Token",
    "ZKP Token",
    "Worst Case"]

for i, (sc, col, lab) in enumerate(zip(
        schemes_compare + ["worst"],
        colors_compare,
        labels_compare)):
    if sc == "worst":
        vals = [res_wc["s"][a] for a in ATTRS_7]
    else:
        vals = [res3[sc]["s"][a] for a in ATTRS_7]
    axes[1].bar(
        x_pos + (i - 1) * w_bar, vals,
        width=w_bar * 0.9,
        color=col, label=lab,
        edgecolor=BG, lw=0.3, alpha=0.92)

axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(
    ATTRS_7, color="white",
    fontsize=10, rotation=25, ha="right")
axes[1].set_ylabel("s(ai) — attribute uniqueness", fontsize=12)
axes[1].set_title(
    "Attribute Uniqueness Comparison\n"
    "Plain Hash vs ZKP vs Worst Case",
    fontsize=12, fontweight="bold", pad=8)
axes[1].set_ylim(0, 1.1)
axes[1].grid(axis="y", alpha=0.15, color="white")
axes[1].legend(
    facecolor=AX, edgecolor=EDGE,
    labelcolor="white", fontsize=10)

plt.suptitle(
    "Analysis 12: Worst-Case Scenario  |  "
    "R_norm Approaching Theoretical Maximum  |  m=7",
    color="white", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout(pad=2)
plt.savefig(
    f"{OUTPUT_DIR}/a12_worst_case.png",
    dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("\n[Chart] a12_worst_case.png saved")


# ===================================================================
# COMBINED CHART — Scalability × Mitigation
# ===================================================================
print("\n[Building combined chart...]")

fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor(BG)
sax(ax)

mults_full = MITIGATIONS["Full\nStack"]
for scheme in SCHEMES:
    R_mit = []
    for N in N_vals:
        df     = gen(N, scheme)
        base_s = {a: s_single(df[a].values) for a in ATTRS_7}
        s_mit  = {a: base_s[a] * mults_full[a] for a in ATTRS_7}
        lm     = {a: {} for a in ATTRS_7}
        for ai, aj in combinations(ATTRS_7, 2):
            sij        = (s_joint(df[ai].values, df[aj].values)
                          * mults_full[ai] * mults_full[aj])
            lm[ai][aj] = lm[aj][ai] = lam(
                s_mit[ai], s_mit[aj], sij)
        R_mit.append(
            R_raw(s_mit, lm, ATTRS_7) / R_max(len(ATTRS_7)))

    ax.plot(
        N_vals, scale_norm[scheme],
        color=COLORS[scheme], linestyle="-",
        lw=2.5, label=f"{scheme} (no mitigation)")
    ax.plot(
        N_vals, R_mit,
        color=COLORS[scheme], linestyle="--",
        lw=1.5, alpha=0.5,
        label=f"{scheme} (full mitigation)")

ax.set_xscale("log")
ax.set_ylim(0, 1.0)
ax.axhline(0.5, color="yellow", lw=1, ls=":", alpha=0.5)
ax.set_xlabel(
    "Population N (log scale)",
    color="white", fontsize=12)
ax.set_ylabel(
    "R_norm  [0, 1]",
    color="white", fontsize=12)
ax.set_title(
    "Combined: Scheme × Scalability × Mitigation  "
    "|  6 Schemes  |  m=7",
    color="white", fontsize=14,
    fontweight="bold", pad=12)
ax.grid(alpha=0.15, color="white")
ax.legend(
    facecolor=AX, edgecolor=EDGE,
    labelcolor="white", fontsize=7, ncol=2)
plt.tight_layout(pad=2)
plt.savefig(
    f"{OUTPUT_DIR}/combined.png",
    dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("[Chart] combined.png saved")


# ===================================================================
# MASTER SUMMARY TABLE
# ===================================================================
print("\n" + "="*65)
print(f"MASTER SUMMARY  (m=7, R_max={R_max(7)}, N={N_BASE})")
print("="*65)

rows = []
for scheme in SCHEMES:
    rp    = res3[scheme]["R_norm"]
    full  = mit_norm[scheme]["Full\nStack"]
    red   = (rp - full) / rp * 100
    floor = scale_norm[scheme][-1]
    mu    = np.mean(seed_results[scheme])
    std   = np.std(seed_results[scheme])
    n50   = micro_norm[scheme][2]
    rows.append({
        "Scheme":       scheme,
        "R_norm":       round(rp,    4),
        "Full Stack":   round(full,  4),
        "Risk Red.%":   round(red,   1),
        "Floor N=100K": round(floor, 4),
        "Seed Mean":    round(mu,    4),
        "Seed Std":     round(std,   4),
        "CV%":          round(std / mu * 100, 1),
        "N=50":         round(n50,   3),
    })

summary = pd.DataFrame(rows)
print(summary.to_string(index=False))
summary.to_csv(
    f"{OUTPUT_DIR}/master_summary.csv", index=False)

print(f"\n{'='*65}")
print("ALL ANALYSES COMPLETE")
print(f"{'='*65}")
print(f"Charts saved to:  {OUTPUT_DIR}/")
print(f"Summary saved to: {OUTPUT_DIR}/master_summary.csv")
print(f"\nChart files:")
charts = [
    "a3_comparison.png",
    "a3_lambda.png",
    "a5_scalability.png",
    "a7_mitigation.png",
    "a8_m_sensitivity.png",
    "a9_seed_sensitivity.png",
    "a10_microscale.png",
    "a11_lambda_sensitivity.png",
    "a12_worst_case.png",
    "combined.png",
]
for c in charts:
    path = f"{OUTPUT_DIR}/{c}"
    exists = os.path.exists(path)
    print(f"  {'✓' if exists else '✗'}  {c}")
