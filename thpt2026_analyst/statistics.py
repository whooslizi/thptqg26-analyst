from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from .data_cleaner import RAW_COLUMN_MAP, SUBJECT_DISPLAY_NAMES


def calculate_subject_statistics(df: pd.DataFrame) -> pd.DataFrame:
    subject_cols = [c for c in RAW_COLUMN_MAP.values() if c in df.columns]

    results = []
    for subj in subject_cols:
        scores = df[subj].dropna()
        if len(scores) == 0:
            continue

        count = len(scores)
        mean_val = float(scores.mean())
        med_val = float(scores.median())
        std_val = float(scores.std())
        min_val = float(scores.min())
        max_val = float(scores.max())
        q1_val = float(scores.quantile(0.25))
        q3_val = float(scores.quantile(0.75))
        iqr_val = q3_val - q1_val

        mode_res = stats.mode(scores, keepdims=True)
        mode_val = (
            float(mode_res.mode[0]) if len(mode_res.mode) > 0 else np.nan
        )

        pct_lt_5 = float((scores < 5.0).sum() / count * 100)
        pct_gte_5 = float((scores >= 5.0).sum() / count * 100)
        pct_gte_8 = float((scores >= 8.0).sum() / count * 100)
        pct_gte_9 = float((scores >= 9.0).sum() / count * 100)
        pct_gte_95 = float((scores >= 9.5).sum() / count * 100)
        count_10 = int((scores == 10.0).sum())
        pct_eq_10 = float(count_10 / count * 100)

        results.append({
            "subject_code": subj,
            "subject_name": SUBJECT_DISPLAY_NAMES.get(subj, subj),
            "candidate_count": count,
            "mean": round(mean_val, 3),
            "median": round(med_val, 3),
            "std": round(std_val, 3),
            "min": round(min_val, 2),
            "max": round(max_val, 2),
            "q1": round(q1_val, 3),
            "q3": round(q3_val, 3),
            "iqr": round(iqr_val, 3),
            "mode": round(mode_val, 2),
            "pct_lt_5": round(pct_lt_5, 2),
            "pct_gte_5": round(pct_gte_5, 2),
            "pct_gte_8": round(pct_gte_8, 2),
            "pct_gte_9": round(pct_gte_9, 2),
            "pct_gte_95": round(pct_gte_95, 2),
            "count_eq_10": count_10,
            "pct_eq_10": round(pct_eq_10, 4),
        })

    return pd.DataFrame(results)


def calculate_admission_blocks(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    block_defs = {
        "A00": ["toan", "vat_li", "hoa_hoc"],
        "A01": ["toan", "vat_li", "ngoai_ngu"],
        "B00": ["toan", "hoa_hoc", "sinh_hoc"],
        "C00": ["ngu_van", "lich_su", "dia_li"],
        "D01": ["toan", "ngu_van", "ngoai_ngu"],
    }

    block_df = (
        df[["sbd", "province_name"]].copy()
        if "sbd" in df.columns
        else pd.DataFrame()
    )

    block_summaries = []

    for block_code, subs in block_defs.items():
        avail_subs = [s for s in subs if s in df.columns]
        if len(avail_subs) == 3:
            comp_df = df[avail_subs].dropna()
            total_score = comp_df.sum(axis=1)
            block_df[f"total_{block_code}"] = total_score

            count = len(total_score)
            if count > 0:
                block_summaries.append({
                    "block_code": block_code,
                    "components": " + ".join([
                        SUBJECT_DISPLAY_NAMES.get(s, s) for s in subs
                    ]),
                    "candidate_count": count,
                    "mean_total": round(float(total_score.mean()), 2),
                    "median_total": round(float(total_score.median()), 2),
                    "std": round(float(total_score.std()), 2),
                    "min": round(float(total_score.min()), 2),
                    "max": round(float(total_score.max()), 2),
                    "pct_gte_24": round(
                        float((total_score >= 24.0).sum() / count * 100), 2
                    ),
                    "pct_gte_27": round(
                        float((total_score >= 27.0).sum() / count * 100), 2
                    ),
                })

    return block_df, pd.DataFrame(block_summaries)


def find_valedictorians(df: pd.DataFrame) -> pd.DataFrame:
    """Finds top scorers / valedictorians (Thủ khoa) per admission block and overall."""
    block_defs = {
        "A00": ["toan", "vat_li", "hoa_hoc"],
        "A01": ["toan", "vat_li", "ngoai_ngu"],
        "B00": ["toan", "hoa_hoc", "sinh_hoc"],
        "C00": ["ngu_van", "lich_su", "dia_li"],
        "D01": ["toan", "ngu_van", "ngoai_ngu"],
    }

    results = []

    for block, subs in block_defs.items():
        avail = [s for s in subs if s in df.columns]
        if len(avail) == 3:
            sub_df = df.dropna(subset=avail).copy()
            if not sub_df.empty:
                sub_df["total"] = sub_df[avail].sum(axis=1)
                max_score = sub_df["total"].max()
                top_candidates = sub_df[sub_df["total"] == max_score]

                for _, row in top_candidates.iterrows():
                    scores_str = ", ".join([
                        f"{SUBJECT_DISPLAY_NAMES.get(s, s)}: {row[s]:.2f}"
                        for s in avail
                    ])
                    results.append({
                        "category": f"Block {block} Valedictorian",
                        "sbd": row.get("sbd", "-"),
                        "province": row.get("province_name", "-"),
                        "total_score": round(row["total"], 2),
                        "details": scores_str,
                    })

    # Overall sum valedictorian
    subj_cols = [c for c in RAW_COLUMN_MAP.values() if c in df.columns]
    if subj_cols:
        sum_df = df.copy()
        sum_df["overall_sum"] = sum_df[subj_cols].sum(axis=1, skipna=True)
        max_sum = sum_df["overall_sum"].max()
        top_overall = sum_df[sum_df["overall_sum"] == max_sum]

        for _, row in top_overall.iterrows():
            results.append({
                "category": "Overall Top Candidate",
                "sbd": row.get("sbd", "-"),
                "province": row.get("province_name", "-"),
                "total_score": round(row["overall_sum"], 2),
                "details": f"Total across subjects taken",
            })

    return pd.DataFrame(results)


def calculate_province_statistics(
    df: pd.DataFrame, subject: str = "toan"
) -> pd.DataFrame:
    if "province_name" not in df.columns or subject not in df.columns:
        return pd.DataFrame()

    valid_df = df[["province_name", subject]].dropna()
    if valid_df.empty:
        return pd.DataFrame()

    grouped = valid_df.groupby("province_name")[subject]

    prov_stats = grouped.agg(
        candidate_count="count",
        mean_score="mean",
        median_score="median",
        std_score="std",
        q1_score=lambda x: x.quantile(0.25),
        q3_score=lambda x: x.quantile(0.75),
        high_score_count=lambda x: (x >= 8.0).sum(),
    ).reset_index()

    prov_stats["high_score_pct"] = (
        prov_stats["high_score_count"] / prov_stats["candidate_count"] * 100
    ).round(2)
    prov_stats["mean_score"] = prov_stats["mean_score"].round(3)
    prov_stats["median_score"] = prov_stats["median_score"].round(3)
    prov_stats.sort_values(by="mean_score", ascending=False, inplace=True)

    return prov_stats


def calculate_subject_correlations(df: pd.DataFrame) -> pd.DataFrame:
    subject_cols = [c for c in RAW_COLUMN_MAP.values() if c in df.columns]
    num_df = df[subject_cols].dropna(how="all")
    if num_df.empty:
        return pd.DataFrame()
    return num_df.corr(method="pearson").round(3)


def detect_score_outliers(
    df: pd.DataFrame, subject: str = "toan"
) -> Dict[str, Any]:
    if subject not in df.columns:
        return {}

    scores = df[subject].dropna()
    if len(scores) == 0:
        return {}

    q1 = float(scores.quantile(0.25))
    q3 = float(scores.quantile(0.75))
    iqr = q3 - q1
    lower_bound = max(0.0, q1 - 1.5 * iqr)
    upper_bound = min(10.0, q3 + 1.5 * iqr)

    iqr_outliers_low = (scores < lower_bound).sum()
    iqr_outliers_high = (scores > upper_bound).sum()

    z_scores = stats.zscore(scores)
    z_outliers = (np.abs(z_scores) > 3.0).sum()

    count_zero = (scores == 0.0).sum()
    count_ten = (scores == 10.0).sum()

    return {
        "subject": subject,
        "subject_name": SUBJECT_DISPLAY_NAMES.get(subject, subject),
        "total_valid": len(scores),
        "q1": round(q1, 2),
        "q3": round(q3, 2),
        "iqr": round(iqr, 2),
        "lower_bound": round(lower_bound, 2),
        "upper_bound": round(upper_bound, 2),
        "iqr_outliers_count": int(iqr_outliers_low + iqr_outliers_high),
        "z_outliers_count": int(z_outliers),
        "zero_scores_count": int(count_zero),
        "perfect_ten_count": int(count_ten),
    }
