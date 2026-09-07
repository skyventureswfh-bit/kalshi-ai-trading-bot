"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { API_BASE_URL } from "../../../lib/api";
import type {
  LiveTradeDecisionFeedPayload,
  LiveTradeDecisionRecord
} from "../../../lib/types";

const APPROVED_NOTE = "APPROVED_FOR_MANUAL_EXECUTION";
const REJECTED_NOTE = "REJECTED_BY_OPERATOR";

type OperatorAction = "approve" | "reject";

function isPending(record: LiveTradeDecisionRecord): boolean {
  const text = [record.decision, record.status, record.summary, record.rationale]
    .filter(Boolean)
    .join(" ")
    .toUpperCase();
  return text.includes("PENDING MANUAL APPROVAL");
}

function operatorState(record: LiveTradeDecisionRecord): "approved" | "rejected" | "pending" {
  const note = (record.feedback?.notes || "").toUpperCase();
  if (note.includes(APPROVED_NOTE)) {
    return "approved";
  }
  if (note.includes(REJECTED_NOTE)) {
    return "rejected";
  }
  return "pending";
}

function formatPrice(value: number | null): string {
  if (value === null || value === undefined) {
    return "n/a";
  }
  return value >= 0 && value <= 1 ? `${(value * 100).toFixed(1)}c` : value.toFixed(3);
}

function formatConfidence(value: number | null): string {
  if (value === null || value === undefined) {
    return "n/a";
  }
  return value >= 0 && value <= 1 ? `${(value * 100).toFixed(1)}%` : value.toFixed(2);
}

async function loadFeed(): Promise<LiveTradeDecisionFeedPayload> {
  const response = await fetch(`${API_BASE_URL}/api/live-trade/decisions?limit=50`, {
    cache: "no-store"
  });
  if (!response.ok) {
    throw new Error(`Decision feed failed (${response.status})`);
  }
  return (await response.json()) as LiveTradeDecisionFeedPayload;
}

async function submitOperatorAction(
  record: LiveTradeDecisionRecord,
  action: OperatorAction
): Promise<void> {
  const response = await fetch(
    `${API_BASE_URL}/api/live-trade/decisions/${encodeURIComponent(record.id)}/feedback`,
    {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        feedback: action === "approve" ? "up" : "down",
        notes: action === "approve" ? APPROVED_NOTE : REJECTED_NOTE,
        source: "manual-approval-queue"
      })
    }
  );

  if (!response.ok) {
    let detail = `Operator action failed (${response.status})`;
    try {
      const payload = (await response.json()) as { error?: string; message?: string };
      detail = payload.message || payload.error || detail;
    } catch {
      // Keep the status-based fallback message.
    }
    throw new Error(detail);
  }
}

export default function ManualApprovalPage() {
  const [feed, setFeed] = useState<LiveTradeDecisionFeedPayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const next = await loadFeed();
      setFeed(next);
      setError(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to load approval queue.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => {
      void refresh();
    }, 10_000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  const pendingRecords = useMemo(
    () => (feed?.decisions || []).filter(isPending),
    [feed]
  );

  const act = async (record: LiveTradeDecisionRecord, action: OperatorAction) => {
    setBusyId(record.id);
    setError(null);
    try {
      await submitOperatorAction(record, action);
      await refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Operator action failed.");
    } finally {
      setBusyId(null);
    }
  };

  return (
    <main className="mx-auto max-w-6xl px-4 py-8 sm:px-6 lg:px-8">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.26em] text-slate-400">
            Beast Manual
          </p>
          <h1 className="mt-2 text-3xl font-semibold text-steel">Manual approval queue</h1>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">
            Beast can hunt and prepare candidates continuously. Approve records your operator decision;
            Reject removes it from consideration. This screen never sends an exchange order.
          </p>
        </div>
        <div className="flex gap-2">
          <Link
            href="/live-trade"
            className="rounded-full border border-slate-200 bg-white px-4 py-2 text-sm font-semibold text-steel"
          >
            Back to live trade
          </Link>
          <button
            type="button"
            onClick={() => void refresh()}
            className="rounded-full border border-slate-200 bg-white px-4 py-2 text-sm font-semibold text-steel"
          >
            Refresh
          </button>
        </div>
      </div>

      <div className="mt-6 rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
        Approval is an operator record only. Existing Beast safety gates remain unchanged, and no real-money order is placed from this page.
      </div>

      {error ? (
        <div className="mt-4 rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
          {error}
        </div>
      ) : null}

      <div className="mt-6 flex flex-wrap gap-3 text-sm text-slate-500">
        <span className="rounded-full border border-slate-200 bg-white px-3 py-1.5">
          Pending found: {pendingRecords.length}
        </span>
        <span className="rounded-full border border-slate-200 bg-white px-3 py-1.5">
          Feed rows: {feed?.decisions.length ?? 0}
        </span>
        <span className="rounded-full border border-slate-200 bg-white px-3 py-1.5">
          Worker: {feed?.heartbeat.workerStatus ?? "unknown"}
        </span>
      </div>

      {loading ? <p className="mt-8 text-sm text-slate-500">Loading approval queue...</p> : null}

      {!loading && pendingRecords.length === 0 ? (
        <div className="mt-8 rounded-[24px] border border-dashed border-slate-200 bg-slate-50 p-8 text-center">
          <h2 className="text-lg font-semibold text-steel">No pending candidate right now</h2>
          <p className="mt-2 text-sm text-slate-500">
            The queue will populate when a Beast candidate reaches the manual-approval hold.
          </p>
        </div>
      ) : null}

      <div className="mt-6 grid gap-4 lg:grid-cols-2">
        {pendingRecords.map((record) => {
          const state = operatorState(record);
          const busy = busyId === record.id;
          return (
            <article key={record.id} className="rounded-[24px] border border-slate-200 bg-white p-5 shadow-sm">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <div className="flex flex-wrap gap-2 text-xs font-semibold uppercase tracking-[0.18em]">
                    <span className="rounded-full bg-amber-100 px-3 py-1 text-amber-800">Pending approval</span>
                    {record.side ? (
                      <span className="rounded-full bg-slate-100 px-3 py-1 text-slate-700">{record.side}</span>
                    ) : null}
                    <span className="rounded-full bg-slate-100 px-3 py-1 text-slate-700">
                      {record.runtimeMode || "mode unknown"}
                    </span>
                  </div>
                  <h2 className="mt-3 text-lg font-semibold text-steel">
                    {record.title || record.marketId || record.eventTicker || "Unlabeled candidate"}
                  </h2>
                  <p className="mt-1 text-sm text-slate-500">{record.marketId || "No market ticker"}</p>
                </div>
                <span
                  className={`rounded-full px-3 py-1 text-xs font-semibold ${
                    state === "approved"
                      ? "bg-emerald-100 text-emerald-800"
                      : state === "rejected"
                        ? "bg-rose-100 text-rose-800"
                        : "bg-slate-100 text-slate-700"
                  }`}
                >
                  {state === "approved" ? "Approved" : state === "rejected" ? "Rejected" : "Awaiting Jeff"}
                </span>
              </div>

              {record.summary ? <p className="mt-4 text-sm font-medium text-steel">{record.summary}</p> : null}
              {record.rationale && record.rationale !== record.summary ? (
                <p className="mt-3 text-sm leading-6 text-slate-600">{record.rationale}</p>
              ) : null}

              <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
                <div className="rounded-xl bg-slate-50 p-3">
                  <p className="text-xs uppercase tracking-wide text-slate-400">Confidence</p>
                  <p className="mt-1 font-semibold text-steel">{formatConfidence(record.confidence)}</p>
                </div>
                <div className="rounded-xl bg-slate-50 p-3">
                  <p className="text-xs uppercase tracking-wide text-slate-400">Limit</p>
                  <p className="mt-1 font-semibold text-steel">{formatPrice(record.metrics.limitPrice)}</p>
                </div>
                <div className="rounded-xl bg-slate-50 p-3">
                  <p className="text-xs uppercase tracking-wide text-slate-400">Quantity</p>
                  <p className="mt-1 font-semibold text-steel">{record.metrics.quantity ?? "n/a"}</p>
                </div>
                <div className="rounded-xl bg-slate-50 p-3">
                  <p className="text-xs uppercase tracking-wide text-slate-400">Edge</p>
                  <p className="mt-1 font-semibold text-steel">
                    {record.metrics.edge === null ? "n/a" : record.metrics.edge.toFixed(3)}
                  </p>
                </div>
              </div>

              <div className="mt-5 flex flex-wrap gap-3">
                {record.marketId ? (
                  <Link
                    href={`/markets/${record.marketId}`}
                    className="rounded-full border border-slate-200 bg-white px-4 py-2 text-sm font-semibold text-steel"
                  >
                    Review market
                  </Link>
                ) : null}
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => void act(record, "approve")}
                  className="rounded-full bg-emerald-700 px-5 py-2 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {busy ? "Saving..." : "Approve"}
                </button>
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => void act(record, "reject")}
                  className="rounded-full bg-rose-700 px-5 py-2 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:opacity-50"
                >
                  Reject
                </button>
              </div>
            </article>
          );
        })}
      </div>
    </main>
  );
}
