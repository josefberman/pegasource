"""
PCAP analysis report generator.

Runs all detectors and summarises findings into a single dict (or JSON file).
"""

from __future__ import annotations

import json
import datetime
from pathlib import Path
from typing import Any

from .stats import protocol_distribution, top_talkers, conversation_table
from .patterns import (
    detect_port_scan,
    detect_beaconing,
    detect_dns_anomalies,
    detect_large_transfers,
    find_unrecognized_protocols,
)


def generate_report(
    packets: list,
    output_path: str | Path | None = None,
    *,
    port_scan_threshold: int = 20,
    beacon_min_occurrences: int = 5,
    large_transfer_bytes: float = 1e6,
    entropy_threshold: float = 3.8,
) -> dict[str, Any]:
    """Run all PCAP analyses and return a unified report.

    Parameters
    ----------
    packets : list[Packet]
        Output of :func:`~pegasource.pcap.read_pcap`.
    output_path : str or Path or None
        If given, write the report as JSON to this file.
    port_scan_threshold : int
        Passed to :func:`~pegasource.pcap.patterns.detect_port_scan`.
    beacon_min_occurrences : int
        Passed to :func:`~pegasource.pcap.patterns.detect_beaconing`.
    large_transfer_bytes : float
        Passed to :func:`~pegasource.pcap.patterns.detect_large_transfers`.
    entropy_threshold : float
        Passed to :func:`~pegasource.pcap.patterns.find_unrecognized_protocols`.

    Returns
    -------
    dict
        Top-level keys:
        ``generated_at``, ``total_packets``, ``protocol_distribution``,
        ``top_talkers``, ``conversations``, ``port_scans``, ``beaconing``,
        ``dns_anomalies``, ``large_transfers``, ``unrecognized_protocols``.
    """
    proto_dist = protocol_distribution(packets)
    talkers_df = top_talkers(packets, n=10)
    conv_df = conversation_table(packets)

    report: dict[str, Any] = {
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "total_packets": len(packets),
        "protocol_distribution": proto_dist,
        "top_talkers": talkers_df.to_dict(orient="records"),
        "conversations": conv_df.head(50).to_dict(orient="records"),
        "port_scans": detect_port_scan(packets, threshold=port_scan_threshold),
        "beaconing": detect_beaconing(packets, min_occurrences=beacon_min_occurrences),
        "dns_anomalies": detect_dns_anomalies(packets),
        "large_transfers": detect_large_transfers(packets, bytes_threshold=large_transfer_bytes),
        "unrecognized_protocols": find_unrecognized_protocols(
            packets, entropy_threshold=entropy_threshold
        ),
    }

    # Pretty-print summary to stdout
    _print_summary(report)

    if output_path is not None:
        output_path = Path(output_path)
        # Convert timestamps and other non-serialisable objects
        output_path.write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )

    return report


def _print_summary(report: dict) -> None:
    """Print a human-readable summary to stdout."""
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  PCAP Report — {report['generated_at']}")
    print(sep)
    print(f"  Total packets : {report['total_packets']:,}")
    print(f"  Protocols     : {report['protocol_distribution']}")
    print(f"\n  ⚠  Port scans detected     : {len(report['port_scans'])}")
    print(f"  ⚠  Beaconing patterns      : {len(report['beaconing'])}")
    print(f"  ⚠  DNS anomalies           : {len(report['dns_anomalies'])}")
    print(f"  ⚠  Large transfers         : {len(report['large_transfers'])}")
    print(f"  ⚠  Unrecognized protocols  : {len(report['unrecognized_protocols'])}")
    print(sep + "\n")
