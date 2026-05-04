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


NOTEBOOK_BODE_GROUP_IDS_BY_MODE = {
    # Эти группы соответствуют Bode-примерам из rc_ladder_model.ipynb
    # для strict black-box эксперимента.
    "strict_extrapolation": [
        "8710d35cee57",
        "84b2f5533d5e",
        "65bafebbc9ff",
    ],
}


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
        score_col = "selection_score" if "selection_score" in leaderboard_df.columns else "score"
        if score_col == "score":
            print(
                f"[WARN] {leaderboard_path} не содержит selection_score; "
                "используется старый fallback score. Перезапустите RC-ladder обучение, "
                "чтобы выбирать seed по validation."
            )
        tie_breakers = [
            c for c in [
                "val_dB_MAE_group_weighted",
                "val_phi_MAE_filtered_deg_group_weighted",
                "test_dB_MAE",
                "test_phi_MAE_filtered_deg",
            ]
            if c in leaderboard_df.columns
        ]
        best_row = leaderboard_df.sort_values(
            by=[score_col, *tie_breakers]
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
            group_ids=NOTEBOOK_BODE_GROUP_IDS_BY_MODE.get(cfg["mode_name"]),
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
