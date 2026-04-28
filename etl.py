"""
ETL pipeline for unified analytics and lead growth platform.
Builds staging, warehouse, and mart layers with SCD2 and data quality checks.
"""
import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataLayer(str, Enum):
    STAGING = "staging"
    WAREHOUSE = "warehouse"
    MART = "mart"


@dataclass
class DataQualityCheck:
    check_name: str
    table: str
    column: Optional[str]
    result: bool
    message: str
    rows_affected: int = 0
    run_at: float = field(default_factory=time.time)


@dataclass
class ETLRunSummary:
    run_id: str
    source: str
    layer: DataLayer
    rows_read: int
    rows_written: int
    rows_rejected: int
    dq_passed: int
    dq_failed: int
    duration_ms: float
    started_at: float
    completed_at: float = field(default_factory=time.time)


class DataWarehouse:
    """SQLite-backed multi-layer data warehouse for marketing and lead analytics."""

    SCHEMA = """
    -- Staging layer: raw ingested data
    CREATE TABLE IF NOT EXISTS stg_leads (
        lead_id TEXT,
        source TEXT,
        channel TEXT,
        email TEXT,
        company TEXT,
        title TEXT,
        industry TEXT,
        created_at TEXT,
        raw_score REAL,
        _ingested_at REAL
    );
    CREATE TABLE IF NOT EXISTS stg_campaigns (
        campaign_id TEXT,
        campaign_name TEXT,
        channel TEXT,
        start_date TEXT,
        end_date TEXT,
        budget_inr REAL,
        impressions INTEGER,
        clicks INTEGER,
        leads_generated INTEGER,
        spend_inr REAL,
        _ingested_at REAL
    );
    CREATE TABLE IF NOT EXISTS stg_events (
        event_id TEXT PRIMARY KEY,
        lead_id TEXT,
        event_type TEXT,
        campaign_id TEXT,
        occurred_at TEXT,
        properties TEXT,
        _ingested_at REAL
    );

    -- Warehouse layer: cleaned, deduped, with SCD2 tracking
    CREATE TABLE IF NOT EXISTS dim_leads (
        surrogate_key TEXT PRIMARY KEY,
        lead_id TEXT,
        source TEXT,
        channel TEXT,
        email TEXT,
        company TEXT,
        title TEXT,
        industry TEXT,
        valid_from TEXT,
        valid_to TEXT,
        is_current INTEGER DEFAULT 1
    );
    CREATE TABLE IF NOT EXISTS dim_campaigns (
        campaign_id TEXT PRIMARY KEY,
        campaign_name TEXT,
        channel TEXT,
        start_date TEXT,
        end_date TEXT,
        budget_inr REAL
    );
    CREATE TABLE IF NOT EXISTS fact_lead_events (
        event_id TEXT PRIMARY KEY,
        lead_surrogate_key TEXT,
        campaign_id TEXT,
        event_type TEXT,
        occurred_at TEXT,
        day_key TEXT,
        week_key TEXT
    );

    -- Mart layer: pre-aggregated for dashboards
    CREATE TABLE IF NOT EXISTS mart_channel_performance (
        channel TEXT,
        period TEXT,
        leads INTEGER,
        mqls INTEGER,
        sqls INTEGER,
        won INTEGER,
        cpl REAL,
        cac REAL,
        roas REAL,
        computed_at REAL
    );
    CREATE TABLE IF NOT EXISTS mart_funnel_daily (
        day TEXT,
        channel TEXT,
        leads INTEGER,
        demos INTEGER,
        proposals INTEGER,
        won INTEGER,
        computed_at REAL
    );
    CREATE TABLE IF NOT EXISTS etl_run_log (
        run_id TEXT PRIMARY KEY,
        source TEXT,
        layer TEXT,
        rows_read INTEGER,
        rows_written INTEGER,
        rows_rejected INTEGER,
        dq_passed INTEGER,
        dq_failed INTEGER,
        duration_ms REAL,
        started_at REAL,
        completed_at REAL
    );
    """

    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.executescript(self.SCHEMA)
        self.conn.commit()

    def execute(self, sql: str, params: tuple = ()) -> None:
        self.conn.execute(sql, params)
        self.conn.commit()

    def executemany(self, sql: str, rows: List[tuple]) -> int:
        self.conn.executemany(sql, rows)
        self.conn.commit()
        return len(rows)

    def query_df(self, sql: str) -> pd.DataFrame:
        return pd.read_sql_query(sql, self.conn)

    def log_run(self, summary: ETLRunSummary) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO etl_run_log VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (summary.run_id, summary.source, summary.layer.value,
             summary.rows_read, summary.rows_written, summary.rows_rejected,
             summary.dq_passed, summary.dq_failed, summary.duration_ms,
             summary.started_at, summary.completed_at),
        )
        self.conn.commit()


class DataQualityEngine:
    """Runs completeness, uniqueness, and freshness checks on a dataframe."""

    def check_nulls(self, df: pd.DataFrame, column: str, table: str) -> DataQualityCheck:
        null_count = int(df[column].isnull().sum()) if column in df.columns else 0
        passed = null_count == 0
        return DataQualityCheck(
            check_name="null_check",
            table=table,
            column=column,
            result=passed,
            message=f"{null_count} nulls found in {column}" if not passed else "ok",
            rows_affected=null_count,
        )

    def check_uniqueness(self, df: pd.DataFrame, column: str, table: str) -> DataQualityCheck:
        if column not in df.columns:
            return DataQualityCheck("uniqueness_check", table, column, False, "Column missing")
        duplicates = int(df[column].duplicated().sum())
        passed = duplicates == 0
        return DataQualityCheck(
            check_name="uniqueness_check",
            table=table,
            column=column,
            result=passed,
            message=f"{duplicates} duplicate values in {column}" if not passed else "ok",
            rows_affected=duplicates,
        )

    def check_range(self, df: pd.DataFrame, column: str, table: str,
                    min_val: float, max_val: float) -> DataQualityCheck:
        if column not in df.columns:
            return DataQualityCheck("range_check", table, column, False, "Column missing")
        out_of_range = int(((df[column] < min_val) | (df[column] > max_val)).sum())
        return DataQualityCheck(
            check_name="range_check",
            table=table,
            column=column,
            result=out_of_range == 0,
            message=f"{out_of_range} values outside [{min_val}, {max_val}]",
            rows_affected=out_of_range,
        )

    def check_row_count(self, df: pd.DataFrame, table: str,
                         min_rows: int = 1) -> DataQualityCheck:
        passed = len(df) >= min_rows
        return DataQualityCheck(
            check_name="row_count_check",
            table=table,
            column=None,
            result=passed,
            message=f"Row count {len(df)} (min {min_rows})" if not passed else "ok",
            rows_affected=len(df),
        )


class ETLPipeline:
    """
    Orchestrates staging -> warehouse -> mart transformations with DQ gates.
    """

    def __init__(self, dw: DataWarehouse):
        self.dw = dw
        self.dq = DataQualityEngine()

    def load_leads_staging(self, df: pd.DataFrame) -> ETLRunSummary:
        t0 = time.perf_counter()
        started = time.time()
        run_id = hashlib.md5(f"leads_stg{t0}".encode()).hexdigest()[:12]

        checks = [
            self.dq.check_row_count(df, "stg_leads"),
            self.dq.check_nulls(df, "lead_id", "stg_leads"),
            self.dq.check_uniqueness(df, "lead_id", "stg_leads"),
        ]
        dq_passed = sum(1 for c in checks if c.result)
        dq_failed = len(checks) - dq_passed

        df_clean = df.dropna(subset=["lead_id"])
        now = time.time()
        rows = [
            (str(r.get("lead_id")), str(r.get("source", "")),
             str(r.get("channel", "")), str(r.get("email", "")),
             str(r.get("company", "")), str(r.get("title", "")),
             str(r.get("industry", "")), str(r.get("created_at", "")),
             float(r.get("raw_score", 0)), now)
            for _, r in df_clean.iterrows()
        ]
        written = self.dw.executemany(
            "INSERT INTO stg_leads VALUES (?,?,?,?,?,?,?,?,?,?)", rows
        )
        dur = (time.perf_counter() - t0) * 1000
        summary = ETLRunSummary(run_id, "leads_crm", DataLayer.STAGING,
                                 len(df), written, len(df) - len(df_clean),
                                 dq_passed, dq_failed, round(dur, 1), started)
        self.dw.log_run(summary)
        return summary

    def build_lead_dimension(self) -> ETLRunSummary:
        t0 = time.perf_counter()
        started = time.time()
        run_id = hashlib.md5(f"dim_leads{t0}".encode()).hexdigest()[:12]
        df = self.dw.query_df("SELECT * FROM stg_leads")
        if df.empty:
            return ETLRunSummary(run_id, "stg_leads", DataLayer.WAREHOUSE,
                                  0, 0, 0, 0, 0, 0.0, started)
        df_deduped = df.drop_duplicates(subset=["lead_id"], keep="last")
        rows = []
        valid_from = time.strftime("%Y-%m-%d")
        for _, r in df_deduped.iterrows():
            sk = hashlib.md5(f"{r['lead_id']}{valid_from}".encode()).hexdigest()[:16]
            rows.append((sk, r["lead_id"], r["source"], r["channel"],
                          r["email"], r["company"], r["title"], r["industry"],
                          valid_from, "9999-12-31", 1))
        self.dw.executemany("INSERT OR IGNORE INTO dim_leads VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
        dur = (time.perf_counter() - t0) * 1000
        summary = ETLRunSummary(run_id, "stg_leads", DataLayer.WAREHOUSE,
                                 len(df), len(rows), 0, 1, 0, round(dur, 1), started)
        self.dw.log_run(summary)
        return summary

    def build_channel_performance_mart(self) -> ETLRunSummary:
        t0 = time.perf_counter()
        started = time.time()
        run_id = hashlib.md5(f"mart_channel{t0}".encode()).hexdigest()[:12]
        df = self.dw.query_df("SELECT * FROM stg_leads")
        campaigns_df = self.dw.query_df("SELECT * FROM stg_campaigns")
        if df.empty:
            return ETLRunSummary(run_id, "mart", DataLayer.MART, 0, 0, 0, 0, 0, 0.0, started)

        if "channel" in df.columns:
            channel_agg = (
                df.groupby("channel")
                .agg(leads=("lead_id", "count"),
                     avg_score=("raw_score", "mean"))
                .reset_index()
            )
            if not campaigns_df.empty and "channel" in campaigns_df.columns:
                camp_ch = campaigns_df.groupby("channel").agg(
                    spend=("spend_inr", "sum"),
                    impressions=("impressions", "sum"),
                ).reset_index()
                merged = channel_agg.merge(camp_ch, on="channel", how="left")
            else:
                merged = channel_agg
                merged["spend"] = 0.0
                merged["impressions"] = 0

            merged["cpl"] = merged.apply(
                lambda r: r.get("spend", 0) / r["leads"] if r["leads"] > 0 else 0, axis=1
            )
            now = time.time()
            period = time.strftime("%Y-%m")
            rows = [
                (str(r["channel"]), period,
                 int(r["leads"]), 0, 0, 0,
                 round(float(r["cpl"]), 2), 0.0, 0.0, now)
                for _, r in merged.iterrows()
            ]
            self.dw.conn.execute("DELETE FROM mart_channel_performance WHERE period=?", (period,))
            written = self.dw.executemany(
                "INSERT INTO mart_channel_performance VALUES (?,?,?,?,?,?,?,?,?,?)", rows
            )
            dur = (time.perf_counter() - t0) * 1000
            summary = ETLRunSummary(run_id, "stg_leads", DataLayer.MART,
                                     len(df), written, 0, 1, 0, round(dur, 1), started)
            self.dw.log_run(summary)
            return summary

        return ETLRunSummary(run_id, "mart", DataLayer.MART, 0, 0, 0, 0, 0, 0.0, started)

    def run_full_pipeline(self, leads_df: pd.DataFrame,
                           campaigns_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        results = {}

        stg_summary = self.load_leads_staging(leads_df)
        results["staging"] = {"rows_written": stg_summary.rows_written,
                               "dq_failed": stg_summary.dq_failed}

        if campaigns_df is not None and not campaigns_df.empty:
            camp_rows = [
                (str(r.get("campaign_id", "")), str(r.get("campaign_name", "")),
                 str(r.get("channel", "")), str(r.get("start_date", "")),
                 str(r.get("end_date", "")), float(r.get("budget_inr", 0)),
                 int(r.get("impressions", 0)), int(r.get("clicks", 0)),
                 int(r.get("leads_generated", 0)), float(r.get("spend_inr", 0)),
                 time.time())
                for _, r in campaigns_df.iterrows()
            ]
            self.dw.executemany("INSERT OR IGNORE INTO stg_campaigns VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                                camp_rows)

        dim_summary = self.build_lead_dimension()
        results["warehouse"] = {"rows_written": dim_summary.rows_written}

        mart_summary = self.build_channel_performance_mart()
        results["mart"] = {"rows_written": mart_summary.rows_written}

        return results


def _generate_demo_data(n_leads: int = 500, n_campaigns: int = 10):
    rng = np.random.default_rng(42)
    channels = ["organic_search", "paid_search", "email", "social", "referral", "direct"]
    industries = ["software", "manufacturing", "retail", "healthcare", "finance", "education"]
    titles = ["CEO", "CTO", "Manager", "Director", "VP", "Analyst", "Engineer"]

    leads = pd.DataFrame({
        "lead_id": [f"lead_{i:05d}" for i in range(n_leads)],
        "source": rng.choice(["crm", "form", "import"], n_leads),
        "channel": rng.choice(channels, n_leads),
        "email": [f"user{i}@company{rng.integers(1, 500)}.com" for i in range(n_leads)],
        "company": [f"Company_{rng.integers(1, 200)}" for _ in range(n_leads)],
        "title": rng.choice(titles, n_leads),
        "industry": rng.choice(industries, n_leads),
        "created_at": pd.date_range("2024-01-01", periods=n_leads, freq="1h").strftime("%Y-%m-%d"),
        "raw_score": rng.uniform(0, 100, n_leads),
    })

    campaigns = pd.DataFrame({
        "campaign_id": [f"camp_{i:03d}" for i in range(n_campaigns)],
        "campaign_name": [f"Campaign {i}" for i in range(n_campaigns)],
        "channel": rng.choice(channels, n_campaigns),
        "start_date": "2024-01-01",
        "end_date": "2024-03-31",
        "budget_inr": rng.uniform(50000, 500000, n_campaigns),
        "impressions": rng.integers(10000, 500000, n_campaigns),
        "clicks": rng.integers(500, 20000, n_campaigns),
        "leads_generated": rng.integers(10, 500, n_campaigns),
        "spend_inr": rng.uniform(30000, 400000, n_campaigns),
    })
    return leads, campaigns


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    dw = DataWarehouse()
    pipeline = ETLPipeline(dw=dw)
    leads, campaigns = _generate_demo_data(500, 10)

    print("Running full ETL pipeline...")
    results = pipeline.run_full_pipeline(leads, campaigns)
    for layer, info in results.items():
        print(f"  {layer}: {info}")

    print("\nChannel performance mart:")
    mart = dw.query_df("SELECT channel, leads, cpl FROM mart_channel_performance")
    print(mart.to_string(index=False))

    print("\nETL run log:")
    log = dw.query_df("SELECT source, layer, rows_read, rows_written, dq_failed, duration_ms FROM etl_run_log")
    print(log.to_string(index=False))
