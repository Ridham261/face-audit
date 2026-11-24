"""
Automated retraining trigger for Face Audit System.

Decision logic:
  1. Run drift check on the last 200 predictions
  2. If drift alerts > 0 (or --force), run training/run_train.py
  3. After training, check whether the new model reached Production
  4. Log the full decision (retrain / skip / promote) to MLflow

Usage:
    python retrain_trigger.py              # check drift, retrain if needed
    python retrain_trigger.py --force      # retrain regardless of drift
    python retrain_trigger.py --dry-run    # check only, don't actually retrain
    python retrain_trigger.py --window 100 # use last 100 predictions for drift
"""
import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MLFLOW_DB = f"sqlite:///{os.path.join(HERE, 'mlflow.db')}"
PYTHON    = sys.executable


def _log_decision(reason: str, retrained: bool, promoted: bool,
                  new_version: str = "", drift_alerts: int = 0,
                  map50: float = 0.0) -> None:
    """Log the retraining decision to MLflow retraining-decisions experiment."""
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_DB)
        mlflow.set_experiment("retraining-decisions")
        run_name = f"retrain_{datetime.now().strftime('%Y%m%d_%H%M')}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                "reason":      reason,
                "retrained":   str(retrained),
                "promoted":    str(promoted),
                "new_version": new_version or "none",
            })
            mlflow.log_metrics({
                "drift_alerts": drift_alerts,
                "map50":        map50,
            })
            mlflow.set_tag("outcome", "promoted" if promoted
                           else ("retrained" if retrained else "skipped"))
        print(f"Decision logged to MLflow → retraining-decisions experiment")
    except Exception as exc:
        print(f"MLflow log skipped: {exc}")


def _run(cmd: list, label: str) -> int:
    """Run a subprocess, stream output, return exit code."""
    print(f"\n{'─'*55}")
    print(f"  Running: {label}")
    print(f"{'─'*55}")
    result = subprocess.run(cmd, cwd=HERE)
    return result.returncode


def _get_latest_map50() -> float:
    """Read the most recent mAP50 from MLflow goggles-detector experiment."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        mlflow.set_tracking_uri(MLFLOW_DB)
        client = MlflowClient(tracking_uri=MLFLOW_DB)
        runs = client.search_runs(
            experiment_names=["goggles-detector"],
            order_by=["start_time DESC"],
            max_results=1,
        )
        if runs:
            return runs[0].data.metrics.get("mAP50", 0.0)
    except Exception:
        pass
    return 0.0


def _check_production_version() -> str:
    """Return current production alias version, or '' if not set."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        mlflow.set_tracking_uri(MLFLOW_DB)
        client = MlflowClient(tracking_uri=MLFLOW_DB)
        mv = client.get_model_version_by_alias("GogglesDetector", "production")
        return str(mv.version)
    except Exception:
        return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force",   action="store_true",
                        help="Retrain regardless of drift")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check drift only, do not retrain")
    parser.add_argument("--window",  type=int, default=200,
                        help="Number of recent predictions to check")
    args = parser.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n{'='*55}")
    print(f"  Face Audit — Automated Retraining Trigger")
    print(f"  {ts}")
    print(f"{'='*55}")

    # ── 1. Drift check ──────────────────────────────────────────────────────
    print("\n[1/4] Checking for production drift …")
    from monitoring.check_drift import check as drift_check
    drift_alerts = drift_check(window=args.window, log_mlflow=True)

    if drift_alerts is None:
        drift_alerts = 0

    # ── 2. Decide whether to retrain ────────────────────────────────────────
    print(f"\n[2/4] Decision")
    if args.force:
        reason = "forced"
        should_retrain = True
        print(f"  → --force flag set: retraining regardless of drift")
    elif drift_alerts > 0:
        reason = f"drift_detected ({drift_alerts} alert(s))"
        should_retrain = True
        print(f"  → {drift_alerts} drift alert(s) detected: scheduling retrain")
    else:
        reason = "no_drift"
        should_retrain = False
        print(f"  → No drift detected: skipping retrain")

    if args.dry_run:
        print(f"  → --dry-run: not retraining (would have: {should_retrain})")
        _log_decision(reason="dry_run", retrained=False, promoted=False,
                      drift_alerts=drift_alerts)
        print("\nDry-run complete.\n")
        return

    if not should_retrain:
        _log_decision(reason=reason, retrained=False, promoted=False,
                      drift_alerts=0)
        print("\nNothing to do — model is healthy.\n")
        return

    # ── 3. Run training ─────────────────────────────────────────────────────
    print(f"\n[3/4] Starting training …")
    version_before = _check_production_version()

    rc = _run([PYTHON, os.path.join(HERE, "training", "run_train.py")],
              "training/run_train.py")
    if rc != 0:
        print(f"\n  ✗ Training failed (exit {rc}) — aborting.")
        _log_decision(reason=reason + "_train_failed", retrained=False,
                      promoted=False, drift_alerts=drift_alerts)
        sys.exit(rc)

    # ── 4. Check promotion ──────────────────────────────────────────────────
    print(f"\n[4/4] Checking promotion …")
    map50 = _get_latest_map50()
    version_after = _check_production_version()

    if version_after and version_after != version_before:
        promoted = True
        print(f"  ✅ GogglesDetector v{version_after} promoted to Production "
              f"(mAP50={map50:.4f})")
    else:
        promoted = False
        print(f"  ⚠  New model NOT promoted to Production "
              f"(mAP50={map50:.4f} < threshold or no change)")

    _log_decision(reason=reason, retrained=True, promoted=promoted,
                  new_version=version_after, drift_alerts=drift_alerts,
                  map50=map50)

    print(f"\n{'='*55}")
    print(f"  Retraining complete")
    print(f"  Promoted : {'YES  → v' + version_after if promoted else 'NO'}")
    print(f"  mAP50    : {map50:.4f}")
    print(f"{'='*55}\n")

    sys.exit(0 if promoted else 2)


if __name__ == "__main__":
    main()
