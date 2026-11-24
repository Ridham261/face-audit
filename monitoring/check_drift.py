"""
Production drift detector for Face Audit System.
Compares recent API predictions against the baseline from run_test results.

Usage:
    python monitoring/check_drift.py              # check last 200 predictions
    python monitoring/check_drift.py --window 100 # check last 100
    python monitoring/check_drift.py --log-mlflow # also log to MLflow
"""
import argparse
import os
import sys
import statistics
from datetime import datetime, timezone

HERE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

# ── Baseline (from run_test on 244 images — update after each full test run) ──
BASELINE = {
    "acceptance_rate": 0.70,   # update from your latest run_test CSV
    "rejection_rate":  0.25,
    "review_rate":     0.05,
    "avg_ms":          44000,  # ms per image on CPU
}

# ── Alert thresholds (how much drift triggers an alert) ──
THRESHOLDS = {
    "acceptance_rate_drop": 0.10,  # alert if acceptance drops >10 pp
    "rejection_rate_spike": 0.15,  # alert if rejection spikes >15 pp
    "avg_ms_multiplier":    2.0,   # alert if avg time doubles
    "min_samples":          10,    # skip check if fewer than this
}

# ─────────────────────────────────────────────────────────────────────────────

def check(window: int = 200, log_mlflow: bool = False):
    from monitoring.prediction_logger import recent, total_count

    rows  = recent(window)
    total = total_count()

    print(f"\nFace Audit — Drift Check   {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'─'*55}")
    print(f"  Total logged predictions : {total}")
    print(f"  Window checked           : last {len(rows)}")

    if len(rows) < THRESHOLDS["min_samples"]:
        print(f"\n  ⚠  Only {len(rows)} samples — need {THRESHOLDS['min_samples']}+ to check drift.")
        print("  Run the API on some images first.\n")
        return

    # ── Compute window metrics ──────────────────────────────────────────────
    statuses    = [r["status"] for r in rows]
    accepted    = statuses.count("accepted")
    rejected    = statuses.count("rejected")
    review      = statuses.count("review")
    n           = len(rows)

    acc_rate  = accepted / n
    rej_rate  = rejected / n
    rev_rate  = review   / n

    times = [r["processing_ms"] for r in rows if r.get("processing_ms")]
    avg_ms = statistics.mean(times) if times else 0

    # ── Print current metrics ───────────────────────────────────────────────
    print(f"\n  {'Metric':<26} {'Current':>8}  {'Baseline':>8}  {'Delta':>8}")
    print(f"  {'─'*54}")

    def _row(label, current, baseline, fmt=".1%", higher_is_good=True):
        delta = current - baseline
        flag  = ""
        if higher_is_good and delta < 0 and abs(delta) > 0.05:
            flag = "  ⚠ DROP"
        if not higher_is_good and delta > 0 and abs(delta) > 0.05:
            flag = "  ⚠ SPIKE"
        print(f"  {label:<26} {current:{fmt}:>8}  {baseline:{fmt}:>8}  {delta:+{fmt}:>8}{flag}")

    _row("Acceptance rate",  acc_rate, BASELINE["acceptance_rate"])
    _row("Rejection rate",   rej_rate, BASELINE["rejection_rate"],  higher_is_good=False)
    _row("Review rate",      rev_rate, BASELINE["review_rate"],     higher_is_good=False)

    ms_ratio = avg_ms / max(BASELINE["avg_ms"], 1)
    ms_flag  = "  ⚠ SLOW" if ms_ratio > THRESHOLDS["avg_ms_multiplier"] else ""
    print(f"  {'Avg processing (ms)':<26} {avg_ms:>8.0f}  {BASELINE['avg_ms']:>8.0f}  {ms_ratio:>7.1f}x{ms_flag}")

    # ── Determine overall health ────────────────────────────────────────────
    alerts = []
    if BASELINE["acceptance_rate"] - acc_rate > THRESHOLDS["acceptance_rate_drop"]:
        alerts.append(f"Acceptance dropped {BASELINE['acceptance_rate']-acc_rate:.1%} below baseline")
    if rej_rate - BASELINE["rejection_rate"] > THRESHOLDS["rejection_rate_spike"]:
        alerts.append(f"Rejection spiked {rej_rate-BASELINE['rejection_rate']:.1%} above baseline")
    if ms_ratio > THRESHOLDS["avg_ms_multiplier"]:
        alerts.append(f"Processing {ms_ratio:.1f}x slower than baseline")

    print(f"\n  {'─'*54}")
    if alerts:
        print(f"  ❌ DRIFT DETECTED — {len(alerts)} alert(s):")
        for a in alerts:
            print(f"     • {a}")
    else:
        print(f"  ✅ No drift detected — metrics within thresholds")

    # ── Status breakdown ────────────────────────────────────────────────────
    print(f"\n  Status breakdown (last {n}):")
    print(f"    accepted : {accepted:3d}  ({acc_rate:.1%})")
    print(f"    rejected : {rejected:3d}  ({rej_rate:.1%})")
    print(f"    review   : {review:3d}  ({rev_rate:.1%})")

    # ── Top rejection reasons ───────────────────────────────────────────────
    from collections import Counter
    reasons = Counter(r["result_code"] for r in rows
                      if r["status"] == "rejected" and r.get("result_code"))
    if reasons:
        print(f"\n  Top rejection reasons:")
        for code, cnt in reasons.most_common(5):
            print(f"    {code:<30} {cnt:3d}  ({cnt/n:.1%})")

    print()

    # ── Optional MLflow logging ─────────────────────────────────────────────
    if log_mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri(f"sqlite:///{os.path.join(HERE, 'mlflow.db')}")
            mlflow.set_experiment("production-monitoring")
            with mlflow.start_run(run_name=f"drift_check_{datetime.now().strftime('%Y%m%d_%H%M')}"):
                mlflow.log_params({"window": n, "total_logged": total})
                mlflow.log_metrics({
                    "acceptance_rate": round(acc_rate, 4),
                    "rejection_rate":  round(rej_rate, 4),
                    "review_rate":     round(rev_rate, 4),
                    "avg_ms":          round(avg_ms, 1),
                    "drift_alerts":    len(alerts),
                })
                mlflow.set_tag("status", "DRIFT" if alerts else "OK")
            print("  Logged to MLflow → production-monitoring experiment")
        except Exception as e:
            print(f"  MLflow log skipped: {e}")

    return len(alerts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window",     type=int, default=200)
    parser.add_argument("--log-mlflow", action="store_true")
    args = parser.parse_args()
    n_alerts = check(window=args.window, log_mlflow=args.log_mlflow)
    sys.exit(1 if n_alerts else 0)
