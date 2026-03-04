"""
pegasource.pcap — PCAP analysis: reader, statistics, pattern detection, reporting.

Quick start::

    from pegasource.pcap import read_pcap, generate_report

    packets = read_pcap("capture.pcap")
    report  = generate_report(packets)
    print(report)
"""

from .reader import read_pcap, packet_summary
from .stats import protocol_distribution, top_talkers, conversation_table
from .patterns import (
    detect_port_scan,
    detect_beaconing,
    detect_dns_anomalies,
    detect_large_transfers,
    find_unrecognized_protocols,
)
from .report import generate_report

__all__ = [
    "read_pcap",
    "packet_summary",
    "protocol_distribution",
    "top_talkers",
    "conversation_table",
    "detect_port_scan",
    "detect_beaconing",
    "detect_dns_anomalies",
    "detect_large_transfers",
    "find_unrecognized_protocols",
    "generate_report",
]
