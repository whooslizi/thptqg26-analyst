import os
import sys

sys.path.append(os.path.abspath("."))

from thpt2026_analyst import (
    clean_thpt_2026_data,
    create_national_dashboard,
    find_valedictorians,
    get_national_top_candidates,
    get_regional_top_candidates,
    load_thpt_2026_data,
    plot_correlation_matrix_grid,
    plot_score_distribution,
    plot_subject_grid_distributions,
    calculate_subject_statistics,
)


def run_pipeline() -> None:
    os.makedirs("data", exist_ok=True)
    os.makedirs("output/charts", exist_ok=True)
    os.makedirs("output/reports", exist_ok=True)

    local_path = "data/raw_diem_thpt_2026.csv"
    raw_df, _ = load_thpt_2026_data(local_path=local_path)
    clean_df, quality_report = clean_thpt_2026_data(raw_df)

    if clean_df.empty:
        return

    plot_subject_grid_distributions(
        clean_df, save_path="output/charts/subject_distributions_grid.png"
    )
    plot_correlation_matrix_grid(
        clean_df, save_path="output/charts/correlation_matrix_grid.png"
    )

    subjects = [
        "toan",
        "ngu_van",
        "ngoai_ngu",
        "vat_li",
        "hoa_hoc",
        "lich_su",
        "dia_li",
        "gdkt_pl",
    ]
    for subj in subjects:
        if subj in clean_df.columns and not clean_df[subj].dropna().empty:
            plot_score_distribution(
                df=clean_df,
                subject=subj,
                color="#C56A0A",
                save_path=f"output/charts/dist_{subj}.png",
            )

    stats_df = calculate_subject_statistics(clean_df)
    stats_df.to_csv(
        "output/reports/national_subject_statistics_2026.csv",
        index=False,
        encoding="utf-8-sig",
    )

    create_national_dashboard(
        df=clean_df,
        quality_report=quality_report,
        stats_df=stats_df,
        save_path="output/charts/national_score_dashboard_2026.png",
    )

    valedictorians_df = find_valedictorians(clean_df)
    valedictorians_df.to_csv(
        "output/reports/valedictorians.csv", index=False, encoding="utf-8-sig"
    )

    regional_top_df = get_regional_top_candidates(clean_df, top_n=5)
    regional_top_df.to_csv(
        "output/reports/top_candidates_regional.csv",
        index=False,
        encoding="utf-8-sig",
    )

    top_100_df = get_national_top_candidates(clean_df, top_n=100)
    top_100_df.to_csv(
        "output/reports/top_100_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )


if __name__ == "__main__":
    run_pipeline()
