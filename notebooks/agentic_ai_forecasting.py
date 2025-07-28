#!/usr/bin/env python3
"""Agentic AI 時系列予測スクリプト（改良版・修正済み）

データ量（258行）に最適化したMOMENTによる時系列予測スクリプト

予測対象: 'agentic artificial intelligence_semantic-count'列の将来値
学習対象: 実際のデータ量に合わせた短いコンテキスト
予測期間: 現実的な範囲での予測評価
"""

import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.cuda.amp

# MOMENT関連のインポート
from momentfm import MOMENTPipeline
from momentfm.utils.utils import control_randomness
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


warnings.filterwarnings("ignore")

# 設定（データ量に合わせて調整）
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FORECAST_HORIZON = 12  # 12ヶ月予測

print(f"🔧 デバイス: {DEVICE}")
print(f"🔧 予測期間: {FORECAST_HORIZON}")


def determine_context_length(data_length: int, forecast_horizon: int) -> int:
    """データ量に基づいて適切なコンテキスト長を決定"""
    available = data_length - forecast_horizon

    # MOMENTが効率的に動作する長さの候補
    candidates = [48, 96, 192, 256, 384, 512]

    best_length = 48  # デフォルト値
    for length in candidates:
        if available >= length + 20:  # 余裕を持って20以上
            best_length = length
        else:
            break

    # 最低限必要な長さを確保
    best_length = max(best_length, 24)  # 最低2年分
    best_length = min(best_length, available - 10)  # 上限設定

    return best_length


class TimeSeriesDataset(Dataset):
    """時系列データセット（動的コンテキスト長対応）"""

    def __init__(
        self,
        data: np.ndarray,
        context_length: int,
        forecast_horizon: int = 12,
    ):
        # 単変量データとして扱う
        if data.ndim > 1:
            self.data = data[:, 0]  # 最初の列（agentic AI関連）のみ使用
        else:
            self.data = data

        self.context_length = context_length
        self.forecast_horizon = forecast_horizon

        # データ長の検証
        min_required = context_length + forecast_horizon
        if len(self.data) < min_required:
            raise ValueError(
                f"データ長が不足: 必要={min_required}, 実際={len(self.data)}",
            )

        print("📊 データセット作成完了:")
        print(f"   - データ長: {len(self.data)}")
        print(f"   - コンテキスト長: {context_length}")
        print(f"   - 予測期間: {forecast_horizon}")

    def __len__(self) -> int:
        return max(1, len(self.data) - self.context_length - self.forecast_horizon + 1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 入力データ
        input_data = self.data[idx : idx + self.context_length]
        # 予測対象データ
        target_data = self.data[
            idx + self.context_length : idx
            + self.context_length
            + self.forecast_horizon
        ]

        # MOMENTの期待入力長: 512
        moment_input_length = 512

        # 512にパディング
        padded_input = np.zeros(moment_input_length)
        if self.context_length <= moment_input_length:
            # 末尾にデータを配置（最新データを重視）
            start_idx = moment_input_length - self.context_length
            padded_input[start_idx:] = input_data
            # input_mask作成（実際のデータ部分は1、padding部分は0）
            input_mask = np.zeros(moment_input_length)
            input_mask[start_idx:] = 1.0
        else:
            # データが512より長い場合は末尾512点を使用
            padded_input = input_data[-moment_input_length:]
            input_mask = np.ones(moment_input_length)

        # MOMENTの期待形式: [channels, sequence_length]
        input_tensor = torch.tensor(padded_input, dtype=torch.float32).unsqueeze(
            0
        )  # [1, 512]
        target_tensor = torch.tensor(target_data, dtype=torch.float32).unsqueeze(
            0
        )  # [1, forecast_horizon]
        mask_tensor = torch.tensor(input_mask, dtype=torch.float32)  # [512]

        return input_tensor, target_tensor, mask_tensor


class AgenticAIForecaster:
    """Agentic AI 時系列予測クラス（改良版・修正済み）"""

    def __init__(self, data_path: str, output_dir: str = "results"):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.df: pd.DataFrame | None = None
        self.model: MOMENTPipeline | None = None
        self.results: dict = {}
        self.context_length: int | None = None

        print(f"📁 出力ディレクトリ: {self.output_dir}")

    def load_and_analyze_data(self) -> pd.DataFrame:
        """データを読み込み、適切なパラメータを決定"""
        print("📊 データ読み込み・分析中...")

        try:
            self.df = pd.read_excel(self.data_path)
            print(f"✅ データ読み込み完了: {self.df.shape}")

            # timeline列をdatetimeに変換
            if "timeline" in self.df.columns:
                self.df["timeline"] = pd.to_datetime(self.df["timeline"])
                self.df = self.df.sort_values("timeline").reset_index(drop=True)
                print(
                    f"📅 日付範囲: {self.df['timeline'].min()} ～ {self.df['timeline'].max()}",
                )

            # 適切なコンテキスト長を決定
            data_length = len(self.df)
            self.context_length = determine_context_length(
                data_length,
                FORECAST_HORIZON,
            )

            print("🎯 最適化されたパラメータ:")
            print(f"   - データ総数: {data_length}")
            print(f"   - コンテキスト長: {self.context_length}")
            print(f"   - 予測期間: {FORECAST_HORIZON}")
            print(
                f"   - 学習可能なサンプル数: {data_length - self.context_length - FORECAST_HORIZON + 1}",
            )

            return self.df

        except Exception as e:
            raise Exception(f"データ読み込みエラー: {e}")

    def prepare_data(
        self,
        end_date: str = "2025-06-01",
        target_column: str = "agentic artificial intelligence_semantic-count",
    ) -> tuple[np.ndarray, pd.DatetimeIndex]:
        """予測用データの準備"""
        if self.df is None:
            raise ValueError("先にload_and_analyze_data()を実行してください")

        # 指定日以前でフィルタリング
        end_datetime = pd.to_datetime(end_date)
        mask = self.df["timeline"] <= end_datetime
        filtered_df = self.df[mask].copy()

        if len(filtered_df) == 0:
            raise ValueError(f"指定日 {end_date} 以前にデータが存在しません")

        # 対象列の確認・選択
        if target_column not in filtered_df.columns:
            numeric_cols = [
                col
                for col in filtered_df.columns
                if col != "timeline" and pd.api.types.is_numeric_dtype(filtered_df[col])
            ]
            target_column = numeric_cols[0]
            print(f"🔄 代替列を使用: {target_column}")

        # 単変量データとして抽出
        data = filtered_df[target_column].values
        dates = filtered_df["timeline"].values

        # 欠損値の処理（新しいpandas記法を使用）
        if pd.isna(data).any():
            print("⚠️ 欠損値処理中...")
            data = pd.Series(data).ffill().bfill().values

        print(f"📈 データ準備完了: {data.shape} (終了: {end_date})")
        print(
            f"📊 統計: min={np.min(data):.1f}, max={np.max(data):.1f}, mean={np.mean(data):.1f}",
        )

        return data.reshape(-1, 1), pd.DatetimeIndex(dates)

    def initialize_model(self, forecast_horizon: int = 12) -> MOMENTPipeline:
        """MOMENTモデルの初期化（512入力長対応）"""
        print("🤖 MOMENTモデル初期化中...")

        model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large",
            model_kwargs={
                "task_name": "forecasting",
                "forecast_horizon": forecast_horizon,
                "head_dropout": 0.1,
                "weight_decay": 0,
                "freeze_encoder": True,
                "freeze_embedder": True,
                "freeze_head": False,
            },
        )

        model.init()
        model = model.to(DEVICE)

        print("✅ MOMENTモデル初期化完了（入力長: 512）")
        print("💡 短いデータは自動的に512にパディングされます")
        return model

    def train_model(
        self,
        train_data: np.ndarray,
        epochs: int = 3,
        batch_size: int = 4,
        lr: float = 1e-4,
    ) -> None:
        """モデルの訓練"""
        if self.model is None:
            raise ValueError("モデルが初期化されていません")
        if self.context_length is None:
            raise ValueError("コンテキスト長が設定されていません")

        print(f"🚀 モデル訓練開始 (epochs: {epochs})")

        # データセット作成
        dataset = TimeSeriesDataset(train_data, self.context_length, FORECAST_HORIZON)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        if len(dataset) == 0:
            raise ValueError("訓練用データセットが空です")

        print(f"📊 データセットサイズ: {len(dataset)}")

        # 訓練設定
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=lr,
        )
        scaler = torch.cuda.amp.GradScaler() if DEVICE.type == "cuda" else None

        self.model.train()

        for epoch in range(epochs):
            epoch_losses = []
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

            for batch_input, batch_target, batch_mask in progress_bar:
                # MOMENTの期待する形状: [batch_size, n_channels, 512]
                batch_input = batch_input.to(DEVICE)  # [batch_size, 1, 512]
                batch_target = batch_target.to(DEVICE)  # [batch_size, 1, forecast_len]
                batch_mask = batch_mask.to(DEVICE)  # [batch_size, 512]

                optimizer.zero_grad()

                try:
                    if scaler:
                        with torch.cuda.amp.autocast():
                            output = self.model(
                                x_enc=batch_input, input_mask=batch_mask
                            )
                            loss = criterion(output.forecast, batch_target)

                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        output = self.model(x_enc=batch_input, input_mask=batch_mask)
                        loss = criterion(output.forecast, batch_target)
                        loss.backward()
                        optimizer.step()

                    epoch_losses.append(loss.item())
                    progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

                except Exception as e:
                    print(f"❌ バッチエラー: {e}")
                    print(f"入力形状: {batch_input.shape}")
                    print(f"ターゲット形状: {batch_target.shape}")
                    print(f"マスク形状: {batch_mask.shape}")
                    print(f"入力形状: {batch_input.shape}")
                    print(f"ターゲット形状: {batch_target.shape}")
                    raise

            avg_loss = np.mean(epoch_losses) if epoch_losses else 0
            print(f"Epoch {epoch + 1} 完了 - 平均損失: {avg_loss:.4f}")

        print("✅ モデル訓練完了")

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """予測実行"""
        if self.model is None:
            raise ValueError("モデルが初期化されていません")
        if self.context_length is None:
            raise ValueError("コンテキスト長が設定されていません")

        self.model.eval()

        with torch.no_grad():
            # 単変量データとして処理
            if input_data.ndim > 1:
                input_data = input_data[:, 0]

            # 最後のcontext_length分を取得
            if len(input_data) >= self.context_length:
                input_sequence = input_data[-self.context_length :]
            else:
                input_sequence = input_data
                print(f"⚠️ データ長が不足: {len(input_data)} < {self.context_length}")

            # MOMENTの期待入力長: 512
            moment_input_length = 512

            # 512にパディング
            padded_input = np.zeros(moment_input_length)
            if len(input_sequence) <= moment_input_length:
                # 末尾にデータを配置（最新データを重視）
                start_idx = moment_input_length - len(input_sequence)
                padded_input[start_idx:] = input_sequence
                # input_mask作成（実際のデータ部分は1、padding部分は0）
                input_mask = np.zeros(moment_input_length)
                input_mask[start_idx:] = 1.0
            else:
                # データが512より長い場合は末尾512点を使用
                padded_input = input_sequence[-moment_input_length:]
                input_mask = np.ones(moment_input_length)

            # [512] → [1, 1, 512]
            input_tensor = (
                torch.tensor(padded_input, dtype=torch.float32)
                .unsqueeze(0)  # バッチ次元
                .unsqueeze(0)  # チャンネル次元
                .to(DEVICE)
            )

            # input_mask [512] → [1, 512] (バッチ次元追加)
            mask_tensor = (
                torch.tensor(input_mask, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            )

            try:
                output = self.model(x_enc=input_tensor, input_mask=mask_tensor)
                if hasattr(output.forecast, "cpu"):
                    prediction = output.forecast.cpu().numpy().squeeze()
                else:
                    prediction = output.forecast.squeeze()
                return prediction

            except Exception as e:
                print(f"❌ 予測エラー: {e}")
                print(f"入力形状: {input_tensor.shape}")
                print(f"マスク形状: {mask_tensor.shape}")
                raise

    def run_evaluation(self, max_years_back: int = 3) -> dict:
        """評価実行（データ量に応じて調整）"""
        if self.df is None:
            raise ValueError("データが読み込まれていません")
        if self.context_length is None:
            raise ValueError("コンテキスト長が設定されていません")

        # データ量に基づいて現実的な評価範囲を決定
        total_months = len(self.df)
        max_possible_years = (
            total_months - self.context_length - FORECAST_HORIZON
        ) // 12

        years_back_list = list(
            range(1, min(max_years_back + 1, max_possible_years + 1)),
        )

        if not years_back_list:
            print("⚠️ データが不足しているため、評価をスキップします")
            return {}

        print(f"🔍 評価開始: {len(years_back_list)}年分の予測を実行")
        print(f"   評価範囲: {years_back_list}")

        results = {}

        for years_back in years_back_list:
            print(f"\n📅 {years_back}年前からの予測...")

            # 訓練終了日の設定
            train_end_date = pd.to_datetime("2025-06-01") - pd.DateOffset(
                years=years_back,
            )
            train_end_str = train_end_date.strftime("%Y-%m-01")

            try:
                # データ準備
                train_data, train_dates = self.prepare_data(train_end_str)

                # データ長チェック
                required_length = self.context_length + FORECAST_HORIZON
                if len(train_data) < required_length:
                    print(
                        f"⚠️ データ長不足のためスキップ: {len(train_data)} < {required_length}",
                    )
                    continue

                # モデル初期化と訓練
                self.model = self.initialize_model(FORECAST_HORIZON)
                self.train_model(train_data, epochs=2)

                # 予測実行
                prediction = self.predict(train_data)

                # 予測期間の日付生成
                last_date = train_dates[-1]
                forecast_dates = pd.date_range(
                    start=last_date + pd.DateOffset(months=1),
                    periods=FORECAST_HORIZON,
                    freq="MS",
                )

                results[years_back] = {
                    "train_end_date": train_end_str,
                    "prediction": prediction,
                    "train_dates": train_dates,
                    "forecast_dates": forecast_dates,
                    "train_data": train_data,
                }

                print(f"✅ {years_back}年前からの予測完了")
                print(
                    f"📈 予測値範囲: {np.min(prediction):.1f} ～ {np.max(prediction):.1f}",
                )

            except Exception as e:
                print(f"❌ {years_back}年前の予測でエラー: {e}")
                continue

        self.results = results
        print(f"\n🎯 評価完了: {len(results)}/{len(years_back_list)} の予測が成功")
        return results

    def get_actual_values(
        self, start_date: pd.Timestamp, periods: int
    ) -> tuple[np.ndarray, pd.DatetimeIndex]:
        """実測値を取得する"""
        try:
            # 元データから実測値を取得
            if self.df is None:
                return np.array([]), pd.DatetimeIndex([])

            # 対象列（agentic AI関連の数値列）を取得
            target_col = None
            for col in self.df.columns[1:]:  # 最初の列は日付と仮定
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    target_col = col
                    break

            if target_col is None:
                return np.array([]), pd.DatetimeIndex([])

            # 日付列を取得
            date_col = self.df.columns[0]
            df_copy = self.df.copy()
            df_copy[date_col] = pd.to_datetime(df_copy[date_col])

            # 指定期間の実測値を抽出
            end_date = start_date + pd.DateOffset(months=periods - 1)
            mask = (df_copy[date_col] >= start_date) & (df_copy[date_col] <= end_date)
            actual_data = df_copy.loc[mask, target_col].values
            actual_dates = pd.date_range(
                start=start_date, periods=len(actual_data), freq="MS"
            )

            return actual_data, actual_dates

        except Exception as e:
            print(f"⚠️ 実測値取得エラー: {e}")
            return np.array([]), pd.DatetimeIndex([])

    def visualize_results(self, save_plots: bool = True) -> None:
        """結果の可視化（横長プロット）"""
        if not self.results:
            print("❌ 結果データがありません")
            return

        print("📊 結果可視化中...")

        # 横長プロット設定
        fig, axes = plt.subplots(
            len(self.results),
            1,
            figsize=(24, 6 * len(self.results)),
        )

        if len(self.results) == 1:
            axes = [axes]

        for idx, (years_back, result) in enumerate(self.results.items()):
            ax = axes[idx]

            # 訓練データのプロット
            train_data = result["train_data"][:, 0]
            train_dates = result["train_dates"]

            ax.plot(
                train_dates,
                train_data,
                color="steelblue",
                linewidth=2,
                label="学習データ",
                alpha=0.8,
            )

            # 予測データのプロット
            prediction = result["prediction"]
            forecast_dates = result["forecast_dates"]

            ax.plot(
                forecast_dates,
                prediction,
                color="orangered",
                linewidth=4,
                linestyle="--",
                label=f"予測 ({years_back}年前から)",
                marker="o",
                markersize=8,
                markerfacecolor="white",
                markeredgecolor="orangered",
                markeredgewidth=2,
            )

            # 実測値を取得・プロット
            train_end_date = (
                pd.to_datetime(result["train_end_date"])
                if isinstance(result["train_end_date"], str)
                else result["train_end_date"]
            )
            forecast_start = train_end_date + pd.DateOffset(months=1)
            actual_values, actual_dates = self.get_actual_values(
                forecast_start, FORECAST_HORIZON
            )
            if len(actual_values) > 0:
                # 予測期間と実測値期間を合わせる
                min_length = min(len(prediction), len(actual_values))
                if min_length > 0:
                    ax.plot(
                        forecast_dates[:min_length],
                        actual_values[:min_length],
                        color="darkgreen",
                        linewidth=3,
                        linestyle="-",
                        label=f"実測値 ({min_length}ヶ月分)",
                        marker="s",
                        markersize=6,
                        markerfacecolor="darkgreen",
                        markeredgecolor="white",
                        markeredgewidth=1,
                        alpha=0.9,
                    )

            # グラフの装飾
            ax.set_title(
                f"Agentic AI 時系列予測 - {years_back}年前から予測 (学習終了: {result['train_end_date']})",
                fontsize=18,
                fontweight="bold",
                pad=25,
            )
            ax.set_xlabel("日付", fontsize=14)
            ax.set_ylabel("セマンティック カウント", fontsize=14)
            ax.legend(fontsize=13, loc="upper left")
            ax.grid(True, alpha=0.3, linestyle=":")

            # 日付軸の書式設定
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=6))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

            # Y軸の範囲調整（実測値も含める）
            all_values = [train_data, prediction]
            if len(actual_values) > 0:
                all_values.append(actual_values)
            combined_values = np.concatenate(all_values)
            y_margin = (np.max(combined_values) - np.min(combined_values)) * 0.1
            ax.set_ylim(
                np.min(combined_values) - y_margin, np.max(combined_values) + y_margin
            )

        plt.tight_layout()

        if save_plots:
            plot_path = self.output_dir / "agentic_ai_forecasting_results.png"
            plt.savefig(
                plot_path,
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
            print(f"📈 プロット保存: {plot_path}")

        plt.show()

    def save_results(self) -> None:
        """結果をCSVファイルに保存"""
        if not self.results:
            print("❌ 保存する結果データがありません")
            return

        print("💾 結果保存中...")

        # 詳細結果の保存
        all_results = []
        for years_back, result in self.results.items():
            prediction = result["prediction"]
            forecast_dates = result["forecast_dates"]

            for i, (date, pred_value) in enumerate(
                zip(forecast_dates, prediction, strict=False),
            ):
                all_results.append({
                    "years_back": years_back,
                    "train_end_date": result["train_end_date"],
                    "forecast_date": date,
                    "month_ahead": i + 1,
                    "predicted_value": pred_value,
                })

        results_df = pd.DataFrame(all_results)
        csv_path = self.output_dir / "agentic_ai_forecasting_results.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"📁 詳細結果: {csv_path}")

        # サマリー統計の保存
        summary_stats = []
        for years_back, result in self.results.items():
            prediction = result["prediction"]
            summary_stats.append({
                "years_back": years_back,
                "train_end_date": result["train_end_date"],
                "prediction_mean": np.mean(prediction),
                "prediction_std": np.std(prediction),
                "prediction_min": np.min(prediction),
                "prediction_max": np.max(prediction),
            })

        summary_df = pd.DataFrame(summary_stats)
        summary_path = self.output_dir / "prediction_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"📊 サマリー統計: {summary_path}")


def main():
    """メイン実行関数"""
    print("🚀 Agentic AI 時系列予測（改良版・修正済み）")
    print("=" * 80)

    # 設定
    control_randomness(seed=RANDOM_SEED)
    data_path = "/home/kimoton/tsfoundations/data/ss/agentic-ai.xlsx"

    # 予測器の初期化と実行
    forecaster = AgenticAIForecaster(data_path, output_dir="results")

    try:
        # データ読み込みと分析
        forecaster.load_and_analyze_data()

        # 評価実行
        forecaster.run_evaluation(max_years_back=3)

        # 結果の可視化と保存
        forecaster.visualize_results(save_plots=True)
        forecaster.save_results()

        print("\n🎉 全ての処理が完了しました！")
        print("=" * 80)

    except Exception as e:
        print(f"❌ エラー発生: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
