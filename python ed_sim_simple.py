# ed_sim_simple.py
# Discrete-event M/M/c simulation for a Hospital ED
# Standard-library only. (Optional plots if matplotlib is installed.)
# Usage: python ed_sim_simple.py

import random
import heapq
import statistics
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

# --------------- core simulation ---------------

@dataclass(order=True)
class Event:
    time: float
    kind: str = field(compare=False)   # "arrival" or "departure"
    pid: int = field(compare=False, default=-1)
    server_idx: Optional[int] = field(compare=False, default=None)

def exp_sample(mean: float) -> float:
    return random.expovariate(1.0 / mean) if mean > 0 else 0.0

def run_simulation(
    sim_time_min: int = 24 * 60,
    mean_interarrival: float = 8.0,
    mean_service: float = 30.0,
    n_servers: int = 3,
    seed: int = 42
) -> Dict[str, float]:
    """
    Returns a dict with: totals, averages (wait, service, LOS), utilization, and queue stats.
    """
    random.seed(seed)

    clock = 0.0
    next_pid = 0
    event_q: List[Event] = []
    queue: List[Tuple[float, int]] = []  # (arrival_time, pid)
    servers_busy = [False] * n_servers
    busy_time = [0.0] * n_servers

    # per-patient times
    arrival: Dict[int, float]  = {}
    start:   Dict[int, float]  = {}
    service: Dict[int, float]  = {}
    depart:  Dict[int, float]  = {}

    # schedule first arrival
    first_arrival = exp_sample(mean_interarrival)
    heapq.heappush(event_q, Event(first_arrival, "arrival", next_pid))
    arrival[next_pid] = first_arrival
    next_pid += 1

    while event_q and clock <= sim_time_min:
        ev = heapq.heappop(event_q)
        clock = ev.time

        if ev.kind == "arrival":
            pid = ev.pid

            # schedule next arrival
            ia = exp_sample(mean_interarrival)
            t_next = clock + ia
            if t_next <= sim_time_min:
                heapq.heappush(event_q, Event(t_next, "arrival", next_pid))
                arrival[next_pid] = t_next
                next_pid += 1

            # find an idle server
            free_idx = next((i for i, busy in enumerate(servers_busy) if not busy), None)
            if free_idx is None:
                queue.append((clock, pid))
            else:
                st = exp_sample(mean_service)
                service[pid] = st
                start[pid] = clock
                depart_time = clock + st
                heapq.heappush(event_q, Event(depart_time, "departure", pid, free_idx))
                servers_busy[free_idx] = True

        else:  # departure
            pid = ev.pid
            sidx = ev.server_idx
            depart[pid] = clock
            if sidx is not None:
                busy_time[sidx] += service.get(pid, 0.0)
                servers_busy[sidx] = False

            # start next patient (if any)
            if queue:
                arr_time, next_in_line = queue.pop(0)
                st = exp_sample(mean_service)
                service[next_in_line] = st
                start[next_in_line] = clock
                heapq.heappush(event_q, Event(clock + st, "departure", next_in_line, sidx))
                servers_busy[sidx] = True

    # metrics
    completed = [pid for pid in depart]
    n_completed = len(completed)

    avg_wait = statistics.mean([(start[pid] - arrival[pid]) for pid in completed]) if completed else 0.0
    avg_service = statistics.mean([service[pid] for pid in completed]) if completed else 0.0
    avg_los = statistics.mean([(depart[pid] - arrival[pid]) for pid in completed]) if completed else 0.0
    overall_util = sum(busy_time) / (n_servers * max(sim_time_min, 1.0))
    total_arrivals = max(arrival.keys(), default=-1) + 1  # number of pids created

    return dict(
        n_servers=n_servers,
        mean_interarrival=mean_interarrival,
        mean_service=mean_service,
        sim_time_min=sim_time_min,
        total_arrivals=total_arrivals,
        completed=n_completed,
        avg_wait_min=avg_wait,
        avg_service_min=avg_service,
        avg_los_min=avg_los,
        overall_util=overall_util
    )

# --------------- convenience runner ---------------

def run_all_scenarios():
    scenarios = [
        ("Baseline (3 doctors)",                 dict(n_servers=3, mean_interarrival=8.0,      mean_service=30.0)),
        ("Add staff (5 doctors)",                dict(n_servers=5, mean_interarrival=8.0,      mean_service=30.0)),
        ("Faster service (3 docs, 20% faster)",  dict(n_servers=3, mean_interarrival=8.0,      mean_service=24.0)),
        ("High demand (3 docs, +25% arrivals)",  dict(n_servers=3, mean_interarrival=6.0,      mean_service=30.0)),
    ]

    print("\nED Simulation (24h) — Results\n")
    header = (
        f"{'Scenario':38} {'Srv':>3} {'Arr':>4} {'Done':>4} "
        f"{'AvgWait':>8} {'AvgSrv':>7} {'AvgLOS':>7} {'Util':>6}"
    )
    print(header)
    print("-" * len(header))

    results = []
    for name, p in scenarios:
        r = run_simulation(
            sim_time_min=24*60,
            mean_interarrival=p["mean_interarrival"],
            mean_service=p["mean_service"],
            n_servers=p["n_servers"],
            seed=42,
        )
        results.append((name, r))
        print(f"{name:38} {r['n_servers']:>3} {r['total_arrivals']:>4} {r['completed']:>4} "
              f"{r['avg_wait_min']:8.2f} {r['avg_service_min']:7.2f} {r['avg_los_min']:7.2f} {r['overall_util']:6.3f}")
    return results

# --------------- optional plotting (if matplotlib present) ---------------

def save_simple_plots(results: List[Tuple[str, Dict[str, float]]], outdir: str = "ed_outputs"):
    try:
        import os
        import matplotlib.pyplot as plt
        os.makedirs(outdir, exist_ok=True)

        # bar plots
        labels = [name for name, _ in results]
        waits  = [r["avg_wait_min"] for _, r in results]
        los    = [r["avg_los_min"] for _, r in results]
        util   = [r["overall_util"] for _, r in results]

        def barplot(vals, title, ylabel, fname):
            fig, ax = plt.subplots(figsize=(8, 4.5))
            x = range(len(vals))
            ax.bar(x, vals)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=20, ha="right")
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, fname))
            plt.close(fig)

        barplot(waits, "Average Wait by Scenario", "Minutes", "avg_wait.png")
        barplot(los,   "Average LOS by Scenario",  "Minutes", "avg_los.png")
        barplot(util,  "Overall Utilization",      "Fraction", "util.png")

        print(f"\nSaved plots to ./{outdir}/  (avg_wait.png, avg_los.png, util.png)")
    except Exception as e:
        # Silently skip plotting if matplotlib is not installed or any error occurs
        print(f"\n(Plots not saved: {e})")

# --------------- main ---------------

if __name__ == "__main__":
    _results = run_all_scenarios()
    save_simple_plots(_results)
