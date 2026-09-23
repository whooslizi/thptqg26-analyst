from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from .data_cleaner import SUBJECT_DISPLAY_NAMES

DASHBOARD_THEME: Dict[str, str] = {
    "bg_color": "#0A1128",
    "panel_bg": "#101B3B",
    "bar_color": "#C56A0A",
    "accent_color": "#E67E22",
    "grid_color": "#1E293B",
    "text_color": "#F8FAFC",
    "subtext_color": "#94A3B8",
    "border_color": "#334155",
}

SHORT_SUBJECT_NAMES: Dict[str, str] = {
    "toan": "TOAN",
    "ngu_van": "VAN",
    "vat_li": "LI",
    "hoa_hoc": "HOA",
    "sinh_hoc": "SINH",
    "lich_su": "SU",
    "dia_li": "DIA",
    "gdkt_pl": "KTPL",
    "ngoai_ngu": "T_ANH",
}


def set_dashboard_style() -> None:
    plt.rcParams["figure.facecolor"] = DASHBOARD_THEME["bg_color"]
    plt.rcParams["axes.facecolor"] = DASHBOARD_THEME["bg_color"]
    plt.rcParams["savefig.facecolor"] = DASHBOARD_THEME["bg_color"]
    plt.rcParams["axes.edgecolor"] = DASHBOARD_THEME["border_color"]
    plt.rcParams["axes.labelcolor"] = DASHBOARD_THEME["text_color"]
    plt.rcParams["xtick.color"] = DASHBOARD_THEME["subtext_color"]
    plt.rcParams["ytick.color"] = DASHBOARD_THEME["text_color"]
    plt.rcParams["text.color"] = DASHBOARD_THEME["text_color"]
    plt.rcParams["font.sans-serif"] = [
        "DejaVu Sans",
        "Arial",
        "Roboto",
        "Helvetica",
    ]
    plt.rcParams["font.family"] = "sans-serif"


def plot_score_distribution(
    df: pd.DataFrame,
    subject: str,
    bins: Optional[List[float]] = None,
    color: str = "#C56A0A",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    set_dashboard_style()

    if subject not in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5,
            0.5,
            f"Subject '{subject}' not found in dataset.",
            ha="center",
            va="center",
            color="red",
        )
        return fig

    scores = df[subject].dropna()
    if len(scores) == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5,
            0.5,
            f"No valid score data for '{subject}'.",
            ha="center",
            va="center",
            color="yellow",
        )
        return fig

    display_name = SUBJECT_DISPLAY_NAMES.get(subject, subject.title())
    possible_scores = (
        np.arange(0.00, 10.25, 0.25) if bins is None else np.array(bins)
    )

    score_counts = scores.round(2).value_counts()
    freq_data = []
    for s in possible_scores:
        s_rounded = round(float(s), 2)
        count = int(score_counts.get(s_rounded, 0))
        freq_data.append((s_rounded, count))

    freq_df = pd.DataFrame(freq_data, columns=["score", "candidate_count"])
    freq_df.sort_values(by="score", ascending=True, inplace=True)

    fig, ax = plt.subplots(figsize=(12, 14), dpi=150)
    fig.patch.set_facecolor(DASHBOARD_THEME["bg_color"])
    ax.set_facecolor(DASHBOARD_THEME["bg_color"])

    y_labels = [f"{s:.2f}" for s in freq_df["score"]]
    y_pos = np.arange(len(freq_df))
    x_counts = freq_df["candidate_count"].values

    ax.barh(
        y_pos,
        x_counts,
        height=0.75,
        color=color,
        edgecolor="#9A4D00",
        linewidth=0.5,
        alpha=0.95,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        y_labels,
        fontsize=9,
        fontweight="bold",
        color=DASHBOARD_THEME["text_color"],
    )
    ax.invert_yaxis()

    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, p: f"{int(x):,}")
    )
    ax.tick_params(
        axis="x", labelsize=10, colors=DASHBOARD_THEME["subtext_color"]
    )
    ax.tick_params(axis="y", length=0)

    ax.grid(
        True,
        axis="x",
        linestyle="--",
        alpha=0.3,
        color=DASHBOARD_THEME["grid_color"],
    )
    ax.grid(
        True,
        axis="y",
        linestyle=":",
        alpha=0.2,
        color=DASHBOARD_THEME["grid_color"],
    )
    ax.set_axisbelow(True)

    ax.set_xlabel(
        "Number of candidates",
        fontsize=11,
        fontweight="bold",
        labelpad=12,
        color=DASHBOARD_THEME["subtext_color"],
    )
    ax.set_ylabel(
        "Score",
        fontsize=11,
        fontweight="bold",
        labelpad=12,
        color=DASHBOARD_THEME["subtext_color"],
    )

    chart_title = title if title else f"THPTQG 2026 — {display_name}"
    ax.set_title(
        chart_title,
        fontsize=16,
        fontweight="bold",
        pad=20,
        color=DASHBOARD_THEME["text_color"],
        loc="center",
    )

    max_count = max(x_counts) if len(x_counts) > 0 else 1
    ax.set_xlim(0, max_count * 1.15)

    top_indices = np.argsort(x_counts)[-8:]
    for idx in top_indices:
        val = x_counts[idx]
        if val > 0:
            ax.text(
                val + (max_count * 0.012),
                y_pos[idx],
                f"{val:,}",
                va="center",
                ha="left",
                fontsize=8,
                fontweight="bold",
                color=DASHBOARD_THEME["text_color"],
            )

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(DASHBOARD_THEME["border_color"])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    return fig


def plot_subject_grid_distributions(
    df: pd.DataFrame, save_path: Optional[str] = None
) -> plt.Figure:
    """Generates a 3x3 subplot grid of subject score histograms matching the reference design."""
    subjects = [
        "toan",
        "ngu_van",
        "vat_li",
        "hoa_hoc",
        "sinh_hoc",
        "lich_su",
        "dia_li",
        "gdkt_pl",
        "ngoai_ngu",
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12), dpi=150)
    fig.suptitle("Score Distribution of Subjects", fontsize=16, pad=20)

    for idx, subj in enumerate(subjects):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        short_name = SHORT_SUBJECT_NAMES.get(subj, subj.upper())

        if subj in df.columns:
            scores = df[subj].dropna()
            if not scores.empty:
                ax.hist(
                    scores,
                    bins=30,
                    color="#4682B4",
                    edgecolor="black",
                    linewidth=0.7,
                    alpha=0.85,
                )

        ax.set_title(f"Distribution of {short_name}", fontsize=12)
        ax.set_xlabel("Score", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.tick_params(axis="both", labelsize=9)
        ax.set_xlim(1.0, 10.2)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    return fig


def plot_correlation_matrix_grid(
    df: pd.DataFrame, save_path: Optional[str] = None
) -> plt.Figure:
    """Generates correlation heatmap matching reference matrix design."""
    subjects = [
        "toan",
        "ngu_van",
        "vat_li",
        "hoa_hoc",
        "sinh_hoc",
        "lich_su",
        "dia_li",
        "gdkt_pl",
        "ngoai_ngu",
    ]

    avail_subs = [s for s in subjects if s in df.columns]
    corr_df = df[avail_subs].corr()

    rename_map = {s: SHORT_SUBJECT_NAMES.get(s, s.upper()) for s in avail_subs}
    corr_renamed = corr_df.rename(columns=rename_map, index=rename_map)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)

    sns.heatmap(
        corr_renamed,
        annot=True,
        fmt=".2f",
        cmap="magma",
        vmax=1.0,
        vmin=0.15,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )

    ax.set_title(
        "Correlation Matrix of Test Results", fontsize=14, pad=15
    )
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    return fig


def plot_province_heatmap(
    prov_stats: pd.DataFrame,
    subject: str = "toan",
    save_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    set_dashboard_style()
    if prov_stats.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 16), dpi=150)
    fig.patch.set_facecolor(DASHBOARD_THEME["bg_color"])
    ax.set_facecolor(DASHBOARD_THEME["bg_color"])

    pivot_df = prov_stats.sort_values(by="mean_score", ascending=True)
    y_labels = pivot_df["province_name"].values
    means = pivot_df["mean_score"].values
    counts = pivot_df["candidate_count"].values

    y_pos = np.arange(len(pivot_df))
    ax.barh(
        y_pos,
        means,
        color="#1D4ED8",
        edgecolor="#3B82F6",
        height=0.65,
        alpha=0.9,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [
            f"{name} (N={count:,})"
            for name, count in zip(y_labels, counts)
        ],
        fontsize=8,
    )
    ax.set_xlabel(
        "Mean Subject Score",
        fontsize=10,
        fontweight="bold",
        color=DASHBOARD_THEME["subtext_color"],
    )
    ax.set_title(
        f"THPT 2026 — Average {SUBJECT_DISPLAY_NAMES.get(subject, subject.title())} Score by Province",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    for i, m in enumerate(means):
        ax.text(
            m + 0.05,
            i,
            f"{m:.2f}",
            va="center",
            fontsize=8,
            color=DASHBOARD_THEME["text_color"],
        )

    ax.set_xlim(0, 10.0)
    ax.grid(
        True,
        axis="x",
        linestyle="--",
        alpha=0.3,
        color=DASHBOARD_THEME["grid_color"],
    )

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig


def plot_correlation_heatmap(
    corr_df: pd.DataFrame, save_path: Optional[str] = None
) -> Optional[plt.Figure]:
    set_dashboard_style()
    if corr_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    fig.patch.set_facecolor(DASHBOARD_THEME["bg_color"])
    ax.set_facecolor(DASHBOARD_THEME["bg_color"])

    display_corr = corr_df.rename(
        columns=SUBJECT_DISPLAY_NAMES, index=SUBJECT_DISPLAY_NAMES
    )

    sns.heatmap(
        display_corr,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        vmax=1.0,
        vmin=0.0,
        linewidths=1.0,
        linecolor=DASHBOARD_THEME["bg_color"],
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )

    ax.set_title(
        "THPT 2026 — Subject Score Correlation Matrix",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.xticks(
        rotation=45,
        ha="right",
        fontsize=9,
        color=DASHBOARD_THEME["text_color"],
    )
    plt.yticks(rotation=0, fontsize=9, color=DASHBOARD_THEME["text_color"])

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig


def create_national_dashboard(
    df: pd.DataFrame,
    quality_report: Dict[str, Any],
    stats_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    set_dashboard_style()
    fig = plt.figure(figsize=(16, 12), dpi=150)
    fig.patch.set_facecolor(DASHBOARD_THEME["bg_color"])

    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    fig.suptitle(
        "THPTQG 2026 — NATIONAL SCORE DASHBOARD",
        fontsize=18,
        fontweight="bold",
        y=0.96,
        color=DASHBOARD_THEME["text_color"],
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(DASHBOARD_THEME["panel_bg"])
    ax1.axis("off")

    total_analyzed = quality_report.get("cleaned_records", len(df))
    coverage_pct = quality_report.get("coverage_pct", 0.0)
    coverage_lbl = quality_report.get("coverage_label", "Partial")

    kpi_text = (
        f"DATA COVERAGE METRICS\n"
        f"---------------------------\n"
        f"Analyzed Candidates : {total_analyzed:,}\n"
        f"MOET Population     : {quality_report.get('official_candidates', 1070000):,}\n"
        f"Coverage Ratio      : {coverage_pct:.2f}%\n"
        f"Dataset Status      : {coverage_lbl}"
    )
    ax1.text(
        0.08,
        0.5,
        kpi_text,
        fontsize=10,
        family="monospace",
        va="center",
        color=DASHBOARD_THEME["text_color"],
    )
    ax1.set_title(
        "Data Provenance Overview",
        fontsize=11,
        fontweight="bold",
        pad=10,
        color=DASHBOARD_THEME["accent_color"],
    )

    ax2 = fig.add_subplot(gs[0, 1])
    if "toan" in df.columns and not df["toan"].dropna().empty:
        scores = df["toan"].dropna()
        ax2.hist(
            scores, bins=20, color="#C56A0A", edgecolor="#9A4D00", alpha=0.85
        )
        ax2.set_title(
            "Mathematics Distribution",
            fontsize=11,
            fontweight="bold",
            color=DASHBOARD_THEME["text_color"],
        )
        ax2.set_xlabel("Score", fontsize=8)
        ax2.set_ylabel("Candidates", fontsize=8)
        ax2.grid(True, linestyle="--", alpha=0.2)

    ax3 = fig.add_subplot(gs[0, 2])
    if "ngu_van" in df.columns and not df["ngu_van"].dropna().empty:
        scores = df["ngu_van"].dropna()
        ax3.hist(
            scores, bins=20, color="#059669", edgecolor="#047857", alpha=0.85
        )
        ax3.set_title(
            "Literature Distribution",
            fontsize=11,
            fontweight="bold",
            color=DASHBOARD_THEME["text_color"],
        )
        ax3.set_xlabel("Score", fontsize=8)
        ax3.set_ylabel("Candidates", fontsize=8)
        ax3.grid(True, linestyle="--", alpha=0.2)

    ax4 = fig.add_subplot(gs[1, :2])
    ax4.axis("off")
    if not stats_df.empty:
        disp_cols = [
            "subject_name",
            "candidate_count",
            "mean",
            "median",
            "std",
            "pct_gte_5",
            "pct_gte_8",
            "pct_eq_10",
        ]
        cols_avail = [c for c in disp_cols if c in stats_df.columns]
        sub_stats = stats_df[cols_avail].copy()

        table_data = []
        headers = [
            "Subject",
            "Count",
            "Mean",
            "Med",
            "Std",
            "%>=5.0",
            "%>=8.0",
            "Score 10",
        ]
        for _, row in sub_stats.iterrows():
            table_data.append([
                row.get("subject_name", ""),
                f"{int(row.get('candidate_count', 0)):,}",
                f"{row.get('mean', 0.0):.2f}",
                f"{row.get('median', 0.0):.2f}",
                f"{row.get('std', 0.0):.2f}",
                f"{row.get('pct_gte_5', 0.0):.1f}%",
                f"{row.get('pct_gte_8', 0.0):.1f}%",
                f"{int(row.get('count_eq_10', 0)) if 'count_eq_10' in row else '-'}",
            ])

        t = ax4.table(
            cellText=table_data,
            colLabels=headers,
            loc="center",
            cellLoc="center",
        )
        t.auto_set_font_size(False)
        t.set_fontsize(8)
        t.scale(1.1, 1.3)
        for key, cell in t.get_celld().items():
            cell.set_facecolor(DASHBOARD_THEME["panel_bg"])
            cell.set_text_props(color=DASHBOARD_THEME["text_color"])
            cell.set_edgecolor(DASHBOARD_THEME["border_color"])

    ax5 = fig.add_subplot(gs[1, 2])
    if "ngoai_ngu" in df.columns and not df["ngoai_ngu"].dropna().empty:
        scores = df["ngoai_ngu"].dropna()
        ax5.hist(
            scores, bins=20, color="#3B82F6", edgecolor="#1D4ED8", alpha=0.85
        )
        ax5.set_title(
            "Foreign Language Distribution",
            fontsize=11,
            fontweight="bold",
            color=DASHBOARD_THEME["text_color"],
        )
        ax5.set_xlabel("Score", fontsize=8)
        ax5.grid(True, linestyle="--", alpha=0.2)

    ax6 = fig.add_subplot(gs[2, :3])
    num_cols = df.select_dtypes(include=[np.number]).columns
    avail_subs = [
        c
        for c in [
            "toan",
            "vat_li",
            "hoa_hoc",
            "sinh_hoc",
            "ngu_van",
            "lich_su",
            "dia_li",
            "ngoai_ngu",
        ]
        if c in num_cols
    ]
    if len(avail_subs) > 1:
        corr = df[avail_subs].corr()
        corr_renamed = corr.rename(
            columns=SUBJECT_DISPLAY_NAMES, index=SUBJECT_DISPLAY_NAMES
        )
        sns.heatmap(
            corr_renamed,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            ax=ax6,
            cbar=False,
            linewidths=0.5,
        )
        ax6.set_title(
            "Subject Score Correlations",
            fontsize=11,
            fontweight="bold",
            color=DASHBOARD_THEME["text_color"],
        )
        plt.setp(ax6.get_xticklabels(), rotation=30, ha="right", fontsize=8)
        plt.setp(ax6.get_yticklabels(), rotation=0, fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    return fig
