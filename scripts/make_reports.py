from __future__ import annotations

import os

import pandas as pd

from ml_circuits.evaluation.reports import summarize_all_rc_ladder_runs
from ml_circuits.features.rc_ladder_features import RUN_CONFIGS
from ml_circuits.visualization.rc_ladder_plots import (
    ensure_dir,
    load_metrics,
    load_predictions,
    plot_bode_examples,
    plot_error_vs_frequency,
    save_group_summary,
)


def main() -> None:
    for cfg in RUN_CONFIGS:
        artifact_dir = cfg["artifact_dir"]
        leaderboard_path = os.path.join(artifact_dir, "leaderboard_seeds.csv")
        metrics_json_path = os.path.join(artifact_dir, "metrics_best_seed.json")
        plots_dir = os.path.join(artifact_dir, "plots_rc_ladder")

        if not os.path.exists(leaderboard_path):
            print(f"[WARN] Не найден файл leaderboard: {leaderboard_path}")
            continue

        if not os.path.exists(metrics_json_path):
            print(f"[WARN] Не найден файл metrics: {metrics_json_path}")
            continue

        ensure_dir(plots_dir)

        leaderboard_df = pd.read_csv(leaderboard_path)
        best_row = leaderboard_df.sort_values(
            by=["score", "test_dB_MAE", "test_phi_MAE_filtered_deg"]
        ).iloc[0]
        best_seed = int(best_row["seed"])

        predictions_csv_path = os.path.join(
            artifact_dir,
            f"predictions_test_n4_seed_{best_seed}.csv"
        )

        if not os.path.exists(predictions_csv_path):
            print(f"[WARN] Не найден файл предсказаний: {predictions_csv_path}")
            continue

        predictions_df = load_predictions(predictions_csv_path)
        _ = load_metrics(metrics_json_path)

        plot_bode_examples(
            predictions_df=predictions_df,
            output_dir=plots_dir,
            num_groups=3,
        )

        plot_error_vs_frequency(
            predictions_df=predictions_df,
            output_dir=plots_dir,
        )

        save_group_summary(
            predictions_df=predictions_df,
            output_dir=plots_dir,
        )

        print(f"[OK] {cfg['mode_name']}")
        print(f"     best_seed        = {best_seed}")
        print(f"     predictions_csv  = {predictions_csv_path}")
        print(f"     plots_dir        = {plots_dir}")

    summary_df = summarize_all_rc_ladder_runs()
    summary_df.to_csv("summary_all_runs.csv", index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(summary_df)

    print("Итоговая сводная таблица:")
    print("  summary_all_runs.csv")

    print("\nПапки с графиками по режимам:")
    for cfg in RUN_CONFIGS:
        print(f"  {cfg['mode_name']}: {os.path.join(cfg['artifact_dir'], 'plots_rc_ladder')}")


if __name__ == "__main__":
    main()
