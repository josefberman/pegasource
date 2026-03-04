"""
PCAP file reader.

Uses scapy for parsing. pyshark can be used as a fallback when scapy
is unavailable, but scapy is strongly preferred for offline operation.
"""

from __future__ import annotations

import datetime
import os
from pathlib import Path
from typing import List

import pandas as pd

try:
    from scapy.all import rdpcap, Packet as ScapyPacket  # type: ignore
    _HAS_SCAPY = True
except ImportError:  # pragma: no cover
    _HAS_SCAPY = False


def read_pcap(path: str | Path) -> list:
    """Load a PCAP or PCAPng file and return a list of scapy packets.

    Parameters
    ----------
    path : str or Path
        Path to the ``.pcap`` / ``.pcapng`` file.

    Returns
    -------
    list[scapy.packet.Packet]
        Ordered list of parsed packets.

    Raises
    ------
    ImportError
        If scapy is not installed.
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PCAP file not found: {path}")

    if not _HAS_SCAPY:
        raise ImportError(
            "scapy is required to read PCAP files. "
            "Install it with: pip install scapy"
        )

    return list(rdpcap(str(path)))


def packet_summary(packets: list) -> pd.DataFrame:
    """Build a per-packet summary DataFrame.

    Parameters
    ----------
    packets : list[Packet]
        Output of :func:`read_pcap`.

    Returns
    -------
    pd.DataFrame
        Columns: ``time``, ``src``, ``dst``, ``proto``, ``sport``,
        ``dport``, ``length``, ``flags``, ``info``.
    """
    from scapy.layers.inet import IP, TCP, UDP, ICMP  # type: ignore
    from scapy.layers.inet6 import IPv6  # type: ignore

    rows = []
    for pkt in packets:
        row: dict = {
            "time": float(pkt.time) if hasattr(pkt, "time") else None,
            "src": None,
            "dst": None,
            "proto": None,
            "sport": None,
            "dport": None,
            "length": len(pkt),
            "flags": None,
            "info": pkt.summary(),
        }

        if pkt.haslayer(IP):
            ip = pkt[IP]
            row["src"] = ip.src
            row["dst"] = ip.dst
            row["proto"] = "IPv4"
        elif pkt.haslayer(IPv6):
            ip6 = pkt[IPv6]
            row["src"] = ip6.src
            row["dst"] = ip6.dst
            row["proto"] = "IPv6"

        if pkt.haslayer(TCP):
            tcp = pkt[TCP]
            row["proto"] = "TCP"
            row["sport"] = tcp.sport
            row["dport"] = tcp.dport
            row["flags"] = str(tcp.flags)
        elif pkt.haslayer(UDP):
            udp = pkt[UDP]
            row["proto"] = "UDP"
            row["sport"] = udp.sport
            row["dport"] = udp.dport
        elif pkt.haslayer(ICMP):
            row["proto"] = "ICMP"

        rows.append(row)

    df = pd.DataFrame(rows)
    if "time" in df.columns and df["time"].notna().any():
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df
