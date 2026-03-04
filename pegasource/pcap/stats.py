"""
PCAP statistics: protocol distribution, top talkers, conversation table.
"""

from __future__ import annotations

from collections import defaultdict
from typing import List

import pandas as pd


def protocol_distribution(packets: list) -> dict[str, int]:
    """Count packets by protocol.

    Parameters
    ----------
    packets : list[Packet]
        Output of :func:`~pegasource.pcap.read_pcap`.

    Returns
    -------
    dict[str, int]
        Protocol name → packet count, sorted descending.
    """
    from scapy.layers.inet import IP, TCP, UDP, ICMP  # type: ignore
    from scapy.layers.inet6 import IPv6  # type: ignore
    from scapy.layers.dns import DNS  # type: ignore

    counts: dict[str, int] = defaultdict(int)
    for pkt in packets:
        if pkt.haslayer(DNS):
            counts["DNS"] += 1
        elif pkt.haslayer(TCP):
            counts["TCP"] += 1
        elif pkt.haslayer(UDP):
            counts["UDP"] += 1
        elif pkt.haslayer(ICMP):
            counts["ICMP"] += 1
        elif pkt.haslayer(IPv6):
            counts["IPv6"] += 1
        elif pkt.haslayer(IP):
            counts["IP (other)"] += 1
        else:
            counts["Other"] += 1

    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def top_talkers(
    packets: list,
    n: int = 10,
    by: str = "bytes",
) -> pd.DataFrame:
    """Identify the sources that sent the most traffic.

    Parameters
    ----------
    packets : list[Packet]
        Output of :func:`~pegasource.pcap.read_pcap`.
    n : int
        Number of top talkers to return.
    by : str
        Sort criterion — ``"bytes"`` or ``"packets"``.

    Returns
    -------
    pd.DataFrame
        Columns: ``src``, ``packets``, ``bytes``.
    """
    from scapy.layers.inet import IP  # type: ignore
    from scapy.layers.inet6 import IPv6  # type: ignore

    stats: dict[str, dict] = defaultdict(lambda: {"packets": 0, "bytes": 0})
    for pkt in packets:
        src = None
        if pkt.haslayer(IP):
            src = pkt[IP].src
        elif pkt.haslayer(IPv6):
            src = pkt[IPv6].src
        if src:
            stats[src]["packets"] += 1
            stats[src]["bytes"] += len(pkt)

    df = pd.DataFrame(
        [{"src": k, **v} for k, v in stats.items()]
    )
    if df.empty:
        return df
    sort_col = "bytes" if by == "bytes" else "packets"
    return df.sort_values(sort_col, ascending=False).head(n).reset_index(drop=True)


def conversation_table(packets: list) -> pd.DataFrame:
    """Build a 5-tuple flow summary (conversations).

    Parameters
    ----------
    packets : list[Packet]

    Returns
    -------
    pd.DataFrame
        Columns: ``src_ip``, ``dst_ip``, ``proto``, ``sport``, ``dport``,
        ``packets``, ``bytes``, ``start_time``, ``end_time``, ``duration_s``.
    """
    from scapy.layers.inet import IP, TCP, UDP  # type: ignore
    from scapy.layers.inet6 import IPv6  # type: ignore

    flows: dict[tuple, dict] = {}
    for pkt in packets:
        src_ip = dst_ip = proto = sport = dport = None
        if pkt.haslayer(IP):
            src_ip, dst_ip = pkt[IP].src, pkt[IP].dst
        elif pkt.haslayer(IPv6):
            src_ip, dst_ip = pkt[IPv6].src, pkt[IPv6].dst
        else:
            continue

        if pkt.haslayer(TCP):
            layer = pkt[TCP]
            proto, sport, dport = "TCP", layer.sport, layer.dport
        elif pkt.haslayer(UDP):
            layer = pkt[UDP]
            proto, sport, dport = "UDP", layer.sport, layer.dport
        else:
            proto, sport, dport = "IP", None, None

        # Canonicalize direction so A→B and B→A share one entry
        key = tuple(sorted([(src_ip, sport), (dst_ip, dport)])) + (proto,)

        t = float(pkt.time)
        if key not in flows:
            flows[key] = {
                "src_ip": src_ip, "dst_ip": dst_ip,
                "proto": proto, "sport": sport, "dport": dport,
                "packets": 0, "bytes": 0,
                "start_time": t, "end_time": t,
            }
        f = flows[key]
        f["packets"] += 1
        f["bytes"] += len(pkt)
        f["start_time"] = min(f["start_time"], t)
        f["end_time"] = max(f["end_time"], t)

    df = pd.DataFrame(list(flows.values()))
    if not df.empty:
        df["duration_s"] = df["end_time"] - df["start_time"]
        df["start_time"] = pd.to_datetime(df["start_time"], unit="s", utc=True)
        df["end_time"] = pd.to_datetime(df["end_time"], unit="s", utc=True)
        df = df.sort_values("bytes", ascending=False).reset_index(drop=True)
    return df
