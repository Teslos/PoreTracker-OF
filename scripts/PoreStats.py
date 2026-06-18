#!/usr/bin/env python3
"""
PoreStats.py — Post-processing engine for PoreTracker-OF CSV output.

Usage:
    python3 PoreStats.py <path_to_pores.csv> [--vLaser vx vy vz] [--t0 t0]

Outputs (same directory as input CSV):
    _msd.csv              per-pore MSD + instantaneous velocity (laser frame)
    _summary.csv          per-pore birth/death/lifetime/displacement/force/birth-pos
    _msd.png              per-pore MSD curves
    _trajectories.png     2-D X-Z trajectories in laser frame
    _ensemble_msd.png     ensemble-averaged MSD + anomalous diffusion fit
    _survival.png         Kaplan-Meier curve + lifetime distribution
    _correlations.png     force / size / depth vs lifetime scatter panels
    _birth_rate.png       nucleation rate vs time + birth-depth distribution
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="PoreTracker-OF post-processor")
    p.add_argument("csv", help="Path to <name>_pores.csv")
    p.add_argument("--vLaser", nargs=3, type=float, default=[0.0, 0.0, 0.0],
                   metavar=("VX", "VY", "VZ"),
                   help="Laser velocity vector (m/s)")
    p.add_argument("--t0", type=float, default=None,
                   help="Reference time for laser frame origin (default: first timestep)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Frame transformation
# ---------------------------------------------------------------------------

def to_laser_frame(df: pd.DataFrame, v_laser: np.ndarray, t0: float) -> pd.DataFrame:
    """Add Xrel, Yrel, Zrel — coordinates relative to moving laser origin."""
    dt = df["Time"] - t0
    df = df.copy()
    df["Xrel"] = df["Cx_m"] - v_laser[0] * dt
    df["Yrel"] = df["Cy_m"] - v_laser[1] * dt
    df["Zrel"] = df["Cz_m"] - v_laser[2] * dt
    return df


# ---------------------------------------------------------------------------
# Instantaneous velocity (laser frame, central differences)
# ---------------------------------------------------------------------------

def calculate_instantaneous_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Vx_ms, Vy_ms, Vz_ms, Speed_ms (m/s) via np.gradient (central differences,
    first/last points use one-sided differences).  Uses laser-frame coords when present.
    """
    df = df.copy()
    for col in ("Vx_ms", "Vy_ms", "Vz_ms", "Speed_ms"):
        df[col] = np.nan

    x_col = "Xrel" if "Xrel" in df.columns else "Cx_m"
    y_col = "Yrel" if "Yrel" in df.columns else "Cy_m"
    z_col = "Zrel" if "Zrel" in df.columns else "Cz_m"

    for pid, grp in df.groupby("PoreID"):
        if len(grp) < 2:
            continue
        g = grp.sort_values("Time")
        t  = g["Time"].values
        vx = np.gradient(g[x_col].values, t)
        vy = np.gradient(g[y_col].values, t)
        vz = np.gradient(g[z_col].values, t)
        df.loc[g.index, "Vx_ms"]   = vx
        df.loc[g.index, "Vy_ms"]   = vy
        df.loc[g.index, "Vz_ms"]   = vz
        df.loc[g.index, "Speed_ms"] = np.sqrt(vx**2 + vy**2 + vz**2)

    return df


# ---------------------------------------------------------------------------
# Per-pore MSD
# ---------------------------------------------------------------------------

def calculate_msd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-pore MSD relative to birth position in laser frame.
    MSD(t) = |r(t) - r_birth|²
    Returns DataFrame: Time, PoreID, MSD_m2, dr_m
    """
    records = []
    for pid, grp in df.groupby("PoreID"):
        if grp["IsKeyhole"].iloc[0] == 1:
            continue
        grp = grp.sort_values("Time")
        x0, y0, z0 = grp["Xrel"].iloc[0], grp["Yrel"].iloc[0], grp["Zrel"].iloc[0]

        msd = ((grp["Xrel"] - x0)**2 + (grp["Yrel"] - y0)**2 +
               (grp["Zrel"] - z0)**2)
        dr  = np.sqrt(msd)
        for t, m, d in zip(grp["Time"], msd, dr):
            records.append({"Time": t, "PoreID": pid, "MSD_m2": m, "dr_m": d})

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Ensemble-averaged MSD
# ---------------------------------------------------------------------------

def calculate_ensemble_msd(df: pd.DataFrame, n_bins: int = 50) -> pd.DataFrame:
    """
    Ensemble-averaged MSD(τ) = ⟨|r(t_birth + τ) − r_birth|²⟩ over all gas pores
    with ≥ 2 snapshots.

    Returns DataFrame: tau_s, MSD_mean_m2, MSD_std_m2, MSD_sem_m2, n_obs
    """
    x_col = "Xrel" if "Xrel" in df.columns else "Cx_m"
    y_col = "Yrel" if "Yrel" in df.columns else "Cy_m"
    z_col = "Zrel" if "Zrel" in df.columns else "Cz_m"

    all_tau, all_msd = [], []

    for pid, grp in df.groupby("PoreID"):
        if grp["IsKeyhole"].iloc[0] == 1 or len(grp) < 2:
            continue
        grp = grp.sort_values("Time")
        t0  = grp["Time"].iloc[0]
        x0, y0, z0 = grp[x_col].iloc[0], grp[y_col].iloc[0], grp[z_col].iloc[0]
        tau = (grp["Time"] - t0).values
        msd = ((grp[x_col] - x0)**2 + (grp[y_col] - y0)**2 +
               (grp[z_col] - z0)**2).values
        all_tau.extend(tau[1:])   # skip τ = 0
        all_msd.extend(msd[1:])

    if not all_tau:
        return pd.DataFrame(columns=["tau_s", "MSD_mean_m2", "MSD_std_m2",
                                     "MSD_sem_m2", "n_obs"])

    all_tau = np.array(all_tau)
    all_msd = np.array(all_msd)

    tau_min = all_tau[all_tau > 0].min() if np.any(all_tau > 0) else 1e-12
    tau_max = all_tau.max()

    if tau_max / max(tau_min, 1e-30) > 100:
        edges = np.logspace(np.log10(tau_min), np.log10(tau_max), n_bins + 1)
    else:
        edges = np.linspace(tau_min, tau_max, n_bins + 1)

    records = []
    for i in range(len(edges) - 1):
        mask = (all_tau >= edges[i]) & (all_tau < edges[i + 1])
        n = int(mask.sum())
        if n == 0:
            continue
        vals = all_msd[mask]
        records.append({
            "tau_s":       0.5 * (edges[i] + edges[i + 1]),
            "MSD_mean_m2": float(vals.mean()),
            "MSD_std_m2":  float(vals.std()),
            "MSD_sem_m2":  float(vals.std() / np.sqrt(n)),
            "n_obs":       n,
        })

    return pd.DataFrame(records)


def fit_diffusion_exponent(ensemble_df: pd.DataFrame):
    """
    Fit MSD = D · τ^α in log-log space (linear regression).
    Returns (alpha, D_eff, r²) — or (nan, nan, nan) if too few usable bins.
    """
    df = ensemble_df[(ensemble_df["tau_s"] > 0) &
                     (ensemble_df["MSD_mean_m2"] > 0)].copy()
    if len(df) < 3:
        return np.nan, np.nan, np.nan
    slope, intercept, r, *_ = scipy_stats.linregress(
        np.log10(df["tau_s"].values),
        np.log10(df["MSD_mean_m2"].values),
    )
    return float(slope), float(10 ** intercept), float(r ** 2)


# ---------------------------------------------------------------------------
# Kaplan-Meier survival estimator
# ---------------------------------------------------------------------------

def kaplan_meier(lifetimes: np.ndarray, censored: np.ndarray = None):
    """
    Kaplan-Meier product-limit estimator with Greenwood plain-linear 95% CI.

    Parameters
    ----------
    lifetimes : array-like, observed lifetimes (s)
    censored  : bool array-like, True = right-censored (pore still alive at end)

    Returns
    -------
    times, S, lower_95, upper_95 — step-function arrays starting at (0, 1)
    """
    lifetimes = np.asarray(lifetimes, dtype=float)
    n = len(lifetimes)
    if n == 0:
        return np.zeros(1), np.ones(1), np.ones(1), np.ones(1)

    if censored is None:
        censored = np.zeros(n, dtype=bool)
    censored = np.asarray(censored, dtype=bool)

    order  = np.argsort(lifetimes)
    t_sort = lifetimes[order]
    c_sort = censored[order]

    times, S_list, gw_list = [0.0], [1.0], [0.0]
    n_risk = n
    i = 0

    while i < n:
        t_cur  = t_sort[i]
        j      = i
        n_ev   = 0
        n_cens = 0
        while j < n and t_sort[j] == t_cur:
            if c_sort[j]:
                n_cens += 1
            else:
                n_ev += 1
            j += 1

        if n_ev > 0 and n_risk > 0:
            s_new  = S_list[-1] * (1.0 - n_ev / n_risk)
            denom  = n_risk * (n_risk - n_ev)
            gw_new = gw_list[-1] + (n_ev / denom if denom > 0 else 0.0)
            times.append(t_cur)
            S_list.append(s_new)
            gw_list.append(gw_new)

        n_risk -= (n_ev + n_cens)
        i = j

    times = np.array(times)
    S     = np.array(S_list)
    gw    = np.array(gw_list)
    se    = S * np.sqrt(gw)
    return times, S, np.clip(S - 1.96 * se, 0.0, 1.0), np.clip(S + 1.96 * se, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-pore lifetime, net displacement, mean volume, mean |F|, birth position."""
    rows = []
    has_birth_pos = "BirthCx" in df.columns

    for pid, grp in df.groupby("PoreID"):
        grp     = grp.sort_values("Time")
        is_kh   = bool(grp["IsKeyhole"].iloc[0])
        t_birth = grp["Time"].iloc[0]
        t_death = grp["Time"].iloc[-1]

        dx = grp["Cx_m"].iloc[-1] - grp["Cx_m"].iloc[0]
        dy = grp["Cy_m"].iloc[-1] - grp["Cy_m"].iloc[0]
        dz = grp["Cz_m"].iloc[-1] - grp["Cz_m"].iloc[0]

        mean_F = np.sqrt(grp["Fx_N"]**2 + grp["Fy_N"]**2 +
                         grp["Fz_N"]**2).mean()

        row = {
            "PoreID":        pid,
            "IsKeyhole":     int(is_kh),
            "BirthTime_s":   t_birth,
            "DeathTime_s":   t_death,
            "Lifetime_s":    t_death - t_birth,
            "NetDisp_m":     np.sqrt(dx**2 + dy**2 + dz**2),
            "MeanVolume_m3": grp["Volume_m3"].mean(),
            "MeanForce_N":   mean_F,
            "NSnapshots":    len(grp),
        }

        if has_birth_pos:
            row["BirthCx_m"] = grp["BirthCx"].iloc[0]
            row["BirthCy_m"] = grp["BirthCy"].iloc[0]
            row["BirthCz_m"] = grp["BirthCz"].iloc[0]

        rows.append(row)

    return pd.DataFrame(rows).sort_values("PoreID").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plots — existing
# ---------------------------------------------------------------------------

def plot_msd(msd_df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for pid, grp in msd_df.groupby("PoreID"):
        grp = grp.sort_values("Time")
        ax.plot(grp["Time"] * 1e3, grp["MSD_m2"] * 1e12,
                label=f"Pore {pid}", linewidth=1.2)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("MSD (µm²)")
    ax.set_title("Pore MSD — Laser Frame")
    if msd_df["PoreID"].nunique() <= 15:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def plot_trajectories(df: pd.DataFrame, out_path: Path):
    """2-D X-Z trajectories in laser frame (keyhole excluded)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = None
    for pid, grp in df.groupby("PoreID"):
        if grp["IsKeyhole"].iloc[0] == 1:
            continue
        grp = grp.sort_values("Time")
        sc = ax.scatter(grp["Xrel"] * 1e6, grp["Zrel"] * 1e6,
                        c=grp["Time"] * 1e3, cmap="viridis", s=10)
    ax.set_xlabel("X (laser frame, µm)")
    ax.set_ylabel("Z (laser frame, µm)")
    ax.set_title("Pore Trajectories — Laser Frame (colour = time, ms)")
    if sc is not None:
        fig.colorbar(sc, ax=ax).set_label("Time (ms)")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plots — ensemble MSD
# ---------------------------------------------------------------------------

def plot_ensemble_msd(ensemble_df: pd.DataFrame, alpha: float, D: float,
                      r2: float, out_path: Path):
    """
    Log-log plot of ensemble-averaged MSD(τ) with ±1 SEM band and power-law fit.
    α < 1: subdiffusive  α ≈ 1: Brownian  α > 1: superdiffusive.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    if ensemble_df.empty:
        ax.text(0.5, 0.5, "No gas-pore ensemble data", transform=ax.transAxes,
                ha="center")
        fig.tight_layout()
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)
        return

    tau  = ensemble_df["tau_s"].values * 1e6          # µs
    msd  = ensemble_df["MSD_mean_m2"].values * 1e12   # µm²
    sem  = ensemble_df["MSD_sem_m2"].values  * 1e12

    ax.fill_between(tau, np.maximum(msd - sem, 1e-6), msd + sem,
                    alpha=0.25, color="steelblue", label="±1 SEM")
    ax.loglog(tau, msd, "o-", color="steelblue", markersize=4,
              linewidth=1.5, label="Ensemble MSD")

    if np.isfinite(alpha):
        tau_fit = np.logspace(np.log10(tau.min()), np.log10(tau.max()), 200)
        D_um   = D * 1e12
        t_s_fit = tau_fit * 1e-6
        msd_fit = D_um * (tau_fit) ** alpha   # D already in µm²/µs^α after unit conversion?
        # Recompute in consistent units: D is in m²/s^α, tau in µs → convert
        msd_fit_SI = D * (tau_fit * 1e-6) ** alpha * 1e12  # µm²
        label_fit  = f"MSD ~ τ^{alpha:.2f}  (R²={r2:.3f})"
        ax.loglog(tau_fit, msd_fit_SI, "--", color="crimson",
                  linewidth=1.5, label=label_fit)

        # Add reference lines for pure diffusion regimes
        if tau.max() / tau.min() > 10:
            for exp, style, lbl in [(1.0, ":", "α=1 (Brownian)"),
                                    (2.0, "-.", "α=2 (ballistic)")]:
                ref = msd[len(msd)//2] * (tau / tau[len(tau)//2]) ** exp
                ax.loglog(tau, ref, style, color="grey", linewidth=0.8,
                          alpha=0.6, label=lbl)

    motion = ("subdiffusive" if alpha < 0.9 else
              "superdiffusive" if alpha > 1.1 else "Brownian") if np.isfinite(alpha) else ""
    title  = f"Ensemble MSD — {len(ensemble_df)} bins"
    if motion:
        title += f"  [{motion}, α={alpha:.2f}]"
    ax.set_xlabel("Lag time τ (µs)")
    ax.set_ylabel("MSD (µm²)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.6)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plots — Kaplan-Meier survival + lifetime distribution
# ---------------------------------------------------------------------------

def plot_survival(gas_summary: pd.DataFrame, t_end: float, out_path: Path):
    """
    2-panel figure:
      Left  — Kaplan-Meier step curve with 95% CI band, exponential fit overlay,
               median lifetime annotated.
      Right — Lifetime histogram with exponential PDF overlay; mean, median, 95th
               percentile marked.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── filter out zero-lifetime pores (born and died in one snapshot) ─────────
    gas = gas_summary[gas_summary["IsKeyhole"] == 0].copy()

    if gas.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No gas pore data", transform=ax.transAxes, ha="center")
        fig.tight_layout()
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)
        return

    lifetimes = gas["Lifetime_s"].values
    # Right-censor: pores still alive at t_end (DeathTime == t_end within 0.5%)
    censored  = gas["DeathTime_s"].values >= t_end * 0.995

    lt_us = lifetimes * 1e6   # µs

    # ── Left: KM curve ────────────────────────────────────────────────────────
    ax = axes[0]
    t_km, S, lo, hi = kaplan_meier(lifetimes, censored)
    t_km_us = t_km * 1e6

    ax.fill_between(t_km_us, lo, hi, step="post", alpha=0.2,
                    color="steelblue", label="95% CI")
    ax.step(t_km_us, S, where="post", color="steelblue", linewidth=2,
            label="Kaplan-Meier")

    # Exponential fit (MLE handles censoring: λ = n_deaths / Σ lifetimes)
    n_deaths = (~censored).sum()
    if n_deaths > 0:
        lam   = n_deaths / lifetimes.sum()
        t_fit = np.linspace(0, lt_us.max(), 300)
        ax.plot(t_fit, np.exp(-lam * t_fit * 1e-6), "--", color="crimson",
                linewidth=1.5, label=f"Exp fit  (λ={lam*1e-6:.3g} µs⁻¹)")

    # Median line
    median_idx = np.searchsorted(-S, -0.5)
    if 0 < median_idx < len(t_km_us):
        t_med = t_km_us[median_idx]
        ax.axvline(t_med, color="orange", linestyle=":", linewidth=1.2,
                   label=f"Median = {t_med:.1f} µs")

    ax.set_xlabel("Lifetime (µs)")
    ax.set_ylabel("Survival probability S(t)")
    ax.set_title(f"Kaplan-Meier — {len(gas)} gas pores")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)

    # ── Right: lifetime histogram ──────────────────────────────────────────────
    ax = axes[1]
    n_bins = max(20, min(60, int(np.sqrt(len(lt_us)))))
    ax.hist(lt_us, bins=n_bins, density=True, color="steelblue",
            edgecolor="white", linewidth=0.4, alpha=0.75, label="Observed")

    if n_deaths > 0:
        t_fit = np.linspace(0, lt_us.max(), 300)
        ax.plot(t_fit, lam * np.exp(-lam * t_fit * 1e-6) * 1e-6,
                "--", color="crimson", linewidth=1.5, label="Exp PDF")

    mean_lt  = lt_us.mean()
    med_lt   = np.median(lt_us)
    p95_lt   = np.percentile(lt_us, 95)
    ax.axvline(mean_lt, color="red",    linestyle="--", linewidth=1.2,
               label=f"Mean = {mean_lt:.1f} µs")
    ax.axvline(med_lt,  color="orange", linestyle=":",  linewidth=1.2,
               label=f"Median = {med_lt:.1f} µs")
    ax.axvline(p95_lt,  color="green",  linestyle="-.", linewidth=1.0,
               label=f"95th pct = {p95_lt:.1f} µs")

    ax.set_xlabel("Lifetime (µs)")
    ax.set_ylabel("Probability density (µs⁻¹)")
    ax.set_title(f"Lifetime distribution  [n={len(gas)}]")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)

    fig.suptitle("Pore Lifetime Statistics", fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plots — correlations (force / size / depth / displacement vs lifetime)
# ---------------------------------------------------------------------------

def plot_correlations(gas_summary: pd.DataFrame, out_path: Path):
    """
    2×2 scatter panels:
      (0,0) Mean volume vs lifetime
      (0,1) Net displacement vs lifetime
      (1,0) Birth depth vs lifetime
      (1,1) Mean |F| vs lifetime  (skipped if all forces are zero)
    Spearman ρ annotated on each panel.
    """
    gas = gas_summary[gas_summary["IsKeyhole"] == 0].copy()

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Pore Lifetime Correlations", fontsize=11)

    lt_us  = gas["Lifetime_s"].values  * 1e6
    vol_um = gas["MeanVolume_m3"].values * 1e18   # µm³
    disp_um = gas["NetDisp_m"].values  * 1e6      # µm

    cmap  = plt.cm.viridis
    btime = gas["BirthTime_s"].values * 1e6 if "BirthTime_s" in gas.columns else np.zeros(len(gas))
    sc_kw = dict(c=btime, cmap=cmap, s=18, alpha=0.6, linewidths=0)

    def annotate_spearman(ax, x, y):
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 5:
            return
        rho, pval = scipy_stats.spearmanr(x[mask], y[mask])
        p_str = f"p={pval:.2g}" if pval >= 0.001 else "p<0.001"
        ax.text(0.97, 0.97, f"ρ={rho:.3f}\n{p_str}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, bbox=dict(boxstyle="round,pad=0.3",
                                       facecolor="white", alpha=0.7))

    # (0,0) Volume vs lifetime
    ax = axes[0, 0]
    sc = ax.scatter(lt_us, vol_um, **sc_kw)
    ax.set_xlabel("Lifetime (µs)")
    ax.set_ylabel("Mean volume (µm³)")
    ax.set_title("Volume vs Lifetime")
    annotate_spearman(ax, lt_us, vol_um)
    fig.colorbar(sc, ax=ax, label="Birth time (µs)", pad=0.02)

    # (0,1) Net displacement vs lifetime
    ax = axes[0, 1]
    sc = ax.scatter(lt_us, disp_um, **sc_kw)
    ax.set_xlabel("Lifetime (µs)")
    ax.set_ylabel("Net displacement (µm)")
    ax.set_title("Net Displacement vs Lifetime")
    annotate_spearman(ax, lt_us, disp_um)
    fig.colorbar(sc, ax=ax, label="Birth time (µs)", pad=0.02)

    # (1,0) Birth depth (Cy) vs lifetime
    ax = axes[1, 0]
    if "BirthCy_m" in gas.columns:
        depth_um = gas["BirthCy_m"].values * 1e6
        sc = ax.scatter(lt_us, depth_um, **sc_kw)
        ax.set_ylabel("Birth depth y (µm)")
        annotate_spearman(ax, lt_us, depth_um)
        fig.colorbar(sc, ax=ax, label="Birth time (µs)", pad=0.02)
    else:
        ax.text(0.5, 0.5, "Birth position not available",
                transform=ax.transAxes, ha="center")
    ax.set_xlabel("Lifetime (µs)")
    ax.set_title("Birth Depth vs Lifetime")

    # (1,1) Mean |F| vs lifetime
    ax = axes[1, 1]
    force_N = gas["MeanForce_N"].values
    if force_N.max() > 0:
        sc = ax.scatter(lt_us, force_N, **sc_kw)
        ax.set_ylabel("Mean |F| (N)")
        annotate_spearman(ax, lt_us, force_N)
        fig.colorbar(sc, ax=ax, label="Birth time (µs)", pad=0.02)
        ax.set_title("Mean Force vs Lifetime")
    else:
        ax.text(0.5, 0.5, "Forces are zero\n(no MHD or Lorentz field)",
                transform=ax.transAxes, ha="center", va="center", fontsize=9,
                color="grey")
        ax.set_title("Mean Force vs Lifetime")
    ax.set_xlabel("Lifetime (µs)")

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plots — nucleation rate + birth-depth distribution
# ---------------------------------------------------------------------------

def plot_birth_rate(gas_summary: pd.DataFrame, t_end: float, out_path: Path):
    """
    2-panel figure:
      Left  — Pore nucleation rate vs time (births per µs bin), with cumulative
               births on a secondary axis.
      Right — Birth-depth distribution (histogram of BirthCy_m), stratified by
               three lifetime terciles so we can see if deep pores live longer.
    """
    gas = gas_summary[gas_summary["IsKeyhole"] == 0].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Pore Nucleation Analysis", fontsize=11, y=1.01)

    # ── Left: birth rate vs time ───────────────────────────────────────────────
    ax  = axes[0]
    ax2 = ax.twinx()

    if not gas.empty and "BirthTime_s" in gas.columns:
        bt_us = gas["BirthTime_s"].values * 1e6
        t_min, t_max = bt_us.min(), max(bt_us.max(), t_end * 1e6)
        n_bins = max(10, min(60, int((t_max - t_min) / 10)))   # ~10 µs per bin
        counts, edges = np.histogram(bt_us, bins=n_bins,
                                     range=(t_min, t_max))
        centers   = 0.5 * (edges[:-1] + edges[1:])
        bin_width = edges[1] - edges[0]
        rate      = counts / bin_width   # births µs⁻¹

        ax.bar(centers, rate, width=bin_width * 0.85, color="steelblue",
               alpha=0.75, label="Nucleation rate")
        ax2.plot(centers, np.cumsum(counts), "r-", linewidth=1.5,
                 label="Cumulative births")

        ax.set_ylabel("Nucleation rate (births / µs)")
        ax2.set_ylabel("Cumulative births", color="red")
        ax2.tick_params(axis="y", labelcolor="red")

    ax.set_xlabel("Time (µs)")
    ax.set_title("Pore Nucleation Rate vs Time")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)

    # ── Right: birth-depth histogram stratified by lifetime tercile ────────────
    ax = axes[1]

    if not gas.empty and "BirthCy_m" in gas.columns:
        depth_um  = gas["BirthCy_m"].values * 1e6
        lt_us     = gas["Lifetime_s"].values * 1e6

        terciles  = np.percentile(lt_us, [33, 67])
        masks     = [
            lt_us <= terciles[0],
            (lt_us > terciles[0]) & (lt_us <= terciles[1]),
            lt_us > terciles[1],
        ]
        labels    = [
            f"Short  (< {terciles[0]:.1f} µs)",
            f"Medium ({terciles[0]:.1f}–{terciles[1]:.1f} µs)",
            f"Long   (> {terciles[1]:.1f} µs)",
        ]
        colors    = ["#1f77b4", "#ff7f0e", "#2ca02c"]

        d_min, d_max = depth_um.min(), depth_um.max()
        bins = np.linspace(d_min, d_max, 30)

        for mask, lbl, col in zip(masks, labels, colors):
            if mask.sum() > 0:
                ax.hist(depth_um[mask], bins=bins, alpha=0.55,
                        color=col, label=lbl, density=True)

        ax.set_xlabel("Birth depth y (µm)")
        ax.set_ylabel("Probability density (µm⁻¹)")
        ax.set_title("Birth Depth by Lifetime Tercile")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    else:
        ax.text(0.5, 0.5, "Birth position not available",
                transform=ax.transAxes, ha="center")
        ax.set_title("Birth Depth Distribution")

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args    = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"ERROR: file not found: {csv_path}")

    stem = csv_path.with_suffix("")

    df = pd.read_csv(csv_path)
    required = {"Time", "PoreID", "IsKeyhole", "Volume_m3",
                "Cx_m", "Cy_m", "Cz_m", "Fx_N", "Fy_N", "Fz_N"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"ERROR: missing columns: {missing}")

    v_laser = np.array(args.vLaser)
    t0      = args.t0 if args.t0 is not None else df["Time"].min()
    t_end   = df["Time"].max()

    print(f"  CSV rows      : {len(df)}")
    print(f"  Unique pores  : {df['PoreID'].nunique()}")
    print(f"  Time range    : {df['Time'].min():.4g} – {t_end:.4g} s")
    print(f"  Laser velocity: {v_laser} m/s  (t0={t0:.4g} s)")

    # ── Transform to laser frame ───────────────────────────────────────────────
    df = to_laser_frame(df, v_laser, t0)

    # ── Instantaneous velocity ─────────────────────────────────────────────────
    print("  Computing instantaneous velocity ...")
    df = calculate_instantaneous_velocity(df)

    # ── Per-pore MSD ──────────────────────────────────────────────────────────
    msd_df = calculate_msd(df)

    # ── Ensemble MSD ──────────────────────────────────────────────────────────
    print("  Computing ensemble MSD ...")
    ensemble_df = calculate_ensemble_msd(df, n_bins=50)
    alpha, D, r2 = fit_diffusion_exponent(ensemble_df)
    if np.isfinite(alpha):
        motion = ("subdiffusive" if alpha < 0.9 else
                  "superdiffusive" if alpha > 1.1 else "Brownian")
        print(f"  Diffusion exponent α = {alpha:.3f}  D = {D:.3g} m²/s^α"
              f"  R² = {r2:.4f}  [{motion}]")

    # ── Summary ───────────────────────────────────────────────────────────────
    summary_df  = build_summary(df)
    gas_summary = summary_df[summary_df["IsKeyhole"] == 0].copy()

    gas_lt = gas_summary["Lifetime_s"]
    if len(gas_lt) > 0:
        print(f"  Gas pores     : {len(gas_lt)}")
        print(f"  Lifetime µs   : mean={gas_lt.mean()*1e6:.2f}"
              f"  median={gas_lt.median()*1e6:.2f}"
              f"  max={gas_lt.max()*1e6:.2f}")

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    msd_path  = Path(str(stem) + "_msd.csv")
    msd_df.to_csv(msd_path, index=False)
    print(f"  MSD CSV       : {msd_path}")

    if not ensemble_df.empty:
        ens_path = Path(str(stem) + "_ensemble_msd.csv")
        ensemble_df.to_csv(ens_path, index=False)
        print(f"  Ensemble MSD  : {ens_path}")

    summary_path = Path(str(stem) + "_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"  Summary CSV   : {summary_path}")
    print(summary_df.to_string(index=False))

    # ── Generate plots ─────────────────────────────────────────────────────────
    if not msd_df.empty:
        p = Path(str(stem) + "_msd.png")
        plot_msd(msd_df, p)
        print(f"  MSD plot      : {p}")

    p = Path(str(stem) + "_trajectories.png")
    plot_trajectories(df, p)
    print(f"  Trajectories  : {p}")

    p = Path(str(stem) + "_ensemble_msd.png")
    plot_ensemble_msd(ensemble_df, alpha, D, r2, p)
    print(f"  Ensemble MSD  : {p}")

    p = Path(str(stem) + "_survival.png")
    plot_survival(gas_summary, t_end, p)
    print(f"  Survival      : {p}")

    p = Path(str(stem) + "_correlations.png")
    plot_correlations(gas_summary, p)
    print(f"  Correlations  : {p}")

    p = Path(str(stem) + "_birth_rate.png")
    plot_birth_rate(gas_summary, t_end, p)
    print(f"  Birth rate    : {p}")


if __name__ == "__main__":
    main()
