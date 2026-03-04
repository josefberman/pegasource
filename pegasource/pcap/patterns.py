"""
Anomaly and pattern detectors for PCAP packet lists.

All detectors work purely offline — no external lookups are performed.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WELL_KNOWN_PORTS: frozenset[int] = frozenset({
    20, 21, 22, 23, 25, 53, 67, 68, 69, 80, 88, 110, 111,
    119, 123, 135, 137, 138, 139, 143, 161, 162, 179, 194, 389,
    443, 445, 465, 514, 515, 587, 631, 636, 853, 989, 990, 993,
    995, 1080, 1194, 1433, 1521, 1723, 3306, 3389, 4444, 4500,
    5060, 5061, 5432, 5900, 6379, 6881, 8080, 8443, 9090, 9200,
    27017,
})


def _shannon_entropy(data: bytes) -> float:
    """Return the Shannon entropy (bits per byte) of *data*."""
    if not data:
        return 0.0
    freq: dict[int, int] = defaultdict(int)
    for b in data:
        freq[b] += 1
    n = len(data)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def _ip_layers(pkt):
    """Return (src_ip, dst_ip) or (None, None)."""
    try:
        from scapy.layers.inet import IP   # type: ignore
        from scapy.layers.inet6 import IPv6  # type: ignore
        if pkt.haslayer(IP):
            return pkt[IP].src, pkt[IP].dst
        if pkt.haslayer(IPv6):
            return pkt[IPv6].src, pkt[IPv6].dst
    except Exception:
        pass
    return None, None


# ---------------------------------------------------------------------------
# Public detectors
# ---------------------------------------------------------------------------

def detect_port_scan(
    packets: list,
    threshold: int = 20,
    window_s: float = 60.0,
    syn_only: bool = True,
) -> list[dict[str, Any]]:
    """Detect potential port scans.

    A host is flagged when it contacts more than *threshold* distinct
    destination ports on a single destination IP within *window_s* seconds.

    Parameters
    ----------
    packets : list[Packet]
    threshold : int
        Minimum number of distinct dest-ports to flag as a scan.
    window_s : float
        Rolling time window in seconds.
    syn_only : bool
        When True, only TCP SYN packets (no ACK, no RST) are considered.

    Returns
    -------
    list[dict]
        Each entry: ``src``, ``dst``, ``ports_contacted``, ``packet_count``.
    """
    from scapy.layers.inet import TCP  # type: ignore

    # src → dst → list of (time, dport)
    records: dict = defaultdict(lambda: defaultdict(list))

    for pkt in packets:
        src, dst = _ip_layers(pkt)
        if src is None:
            continue
        if not pkt.haslayer(TCP):
            continue
        tcp = pkt[TCP]
        if syn_only:
            # SYN set, ACK/RST not set
            flags = str(tcp.flags)
            if "S" not in flags or "A" in flags or "R" in flags:
                continue
        records[src][dst].append((float(pkt.time), tcp.dport))

    results = []
    for src, dsts in records.items():
        for dst, events in dsts.items():
            events.sort()
            # Sliding window
            left = 0
            for right in range(len(events)):
                while events[right][0] - events[left][0] > window_s:
                    left += 1
                window_events = events[left : right + 1]
                ports = {e[1] for e in window_events}
                if len(ports) >= threshold:
                    results.append({
                        "src": src,
                        "dst": dst,
                        "ports_contacted": sorted(ports),
                        "packet_count": len(window_events),
                        "window_start": events[left][0],
                        "window_end": events[right][0],
                    })
                    break  # one alert per src/dst pair

    return results


def detect_beaconing(
    packets: list,
    min_occurrences: int = 5,
    jitter_tolerance_s: float = 2.0,
) -> list[dict[str, Any]]:
    """Detect periodic (beaconing) connections between host pairs.

    Hosts that communicate at regular intervals (like C2 check-ins) are
    flagged when the standard deviation of inter-packet intervals is low
    relative to the mean.

    Parameters
    ----------
    packets : list[Packet]
    min_occurrences : int
        Minimum number of connections between a host pair to consider.
    jitter_tolerance_s : float
        Maximum allowed standard deviation of inter-arrival times (seconds).

    Returns
    -------
    list[dict]
        Each entry: ``src``, ``dst``, ``dport``, ``mean_interval_s``,
        ``std_interval_s``, ``occurrences``.
    """
    from scapy.layers.inet import TCP, UDP  # type: ignore

    # (src, dst, dport) → sorted list of timestamps
    flows: dict[tuple, list] = defaultdict(list)

    for pkt in packets:
        src, dst = _ip_layers(pkt)
        if src is None:
            continue
        dport = None
        if pkt.haslayer(TCP):
            dport = pkt[TCP].dport
        elif pkt.haslayer(UDP):
            dport = pkt[UDP].dport
        if dport is not None:
            flows[(src, dst, dport)].append(float(pkt.time))

    results = []
    for (src, dst, dport), times in flows.items():
        if len(times) < min_occurrences:
            continue
        times_sorted = sorted(times)
        intervals = [
            t2 - t1 for t1, t2 in zip(times_sorted, times_sorted[1:])
        ]
        if not intervals:
            continue
        mean_iv = sum(intervals) / len(intervals)
        if mean_iv <= 0:
            continue
        variance = sum((x - mean_iv) ** 2 for x in intervals) / len(intervals)
        std_iv = variance ** 0.5

        if std_iv <= jitter_tolerance_s:
            results.append({
                "src": src,
                "dst": dst,
                "dport": dport,
                "mean_interval_s": round(mean_iv, 3),
                "std_interval_s": round(std_iv, 3),
                "occurrences": len(times_sorted),
            })

    return sorted(results, key=lambda r: r["std_interval_s"])


def detect_dns_anomalies(packets: list) -> list[dict[str, Any]]:
    """Detect suspicious DNS activity.

    Checks for:
    * High NXDOMAIN rate (> 50 % of responses)
    * Unusually long or subdomain-heavy query names (potential tunnelling)
    * High Shannon entropy in query labels (potential data exfiltration)

    Parameters
    ----------
    packets : list[Packet]

    Returns
    -------
    list[dict]
        Each entry: ``type``, ``description``, and additional context fields.
    """
    from scapy.layers.dns import DNS, DNSQR, DNSRR  # type: ignore

    total_responses = 0
    nxdomain_count = 0
    long_queries: list[dict] = []
    high_entropy: list[dict] = []

    for pkt in packets:
        if not pkt.haslayer(DNS):
            continue
        dns = pkt[DNS]

        # Responses
        qr = int(dns.qr) if dns.qr is not None else -1
        if qr == 1:  # response
            total_responses += 1
            if dns.rcode == 3:  # NXDOMAIN
                nxdomain_count += 1

        # Queries
        qdcount = int(dns.qdcount) if dns.qdcount is not None else 0
        if qr == 0 and qdcount > 0:
            try:
                qname = dns.qd.qname.decode("utf-8", errors="ignore").rstrip(".")
            except Exception:
                continue

            # Long query name
            if len(qname) > 100:
                long_queries.append({"qname": qname, "length": len(qname)})

            # High-entropy subdomains (potential exfiltration)
            labels = qname.split(".")
            if labels:
                first_label = labels[0]
                ent = _shannon_entropy(first_label.encode())
                if ent > 3.5 and len(first_label) > 12:
                    high_entropy.append({
                        "qname": qname,
                        "label": first_label,
                        "entropy": round(ent, 3),
                    })

    results: list[dict] = []

    # NXDOMAIN anomaly
    if total_responses > 0 and nxdomain_count / total_responses > 0.5:
        results.append({
            "type": "HIGH_NXDOMAIN_RATE",
            "description": (
                f"NXDOMAIN rate is "
                f"{nxdomain_count / total_responses:.1%} "
                f"({nxdomain_count}/{total_responses} responses)"
            ),
            "nxdomain_count": nxdomain_count,
            "total_responses": total_responses,
        })

    for q in long_queries:
        results.append({
            "type": "LONG_DNS_QUERY",
            "description": f"Unusually long DNS query ({q['length']} chars)",
            **q,
        })

    for q in high_entropy:
        results.append({
            "type": "HIGH_ENTROPY_DNS_LABEL",
            "description": (
                f"High-entropy DNS label detected "
                f"(entropy={q['entropy']:.2f}) — possible DNS tunnel"
            ),
            **q,
        })

    return results


def detect_large_transfers(
    packets: list,
    bytes_threshold: float = 1e6,
) -> list[dict[str, Any]]:
    """Identify host pairs that transferred unusually large amounts of data.

    Parameters
    ----------
    packets : list[Packet]
    bytes_threshold : float
        Minimum total bytes between a host pair to flag.

    Returns
    -------
    list[dict]
        Each entry: ``src``, ``dst``, ``total_bytes``, ``packet_count``.
    """
    flows: dict[tuple, dict] = defaultdict(lambda: {"bytes": 0, "packets": 0})

    for pkt in packets:
        src, dst = _ip_layers(pkt)
        if src is None:
            continue
        key = (src, dst)
        flows[key]["bytes"] += len(pkt)
        flows[key]["packets"] += 1

    return [
        {"src": k[0], "dst": k[1], "total_bytes": v["bytes"], "packet_count": v["packets"]}
        for k, v in sorted(flows.items(), key=lambda x: -x[1]["bytes"])
        if v["bytes"] >= bytes_threshold
    ]


def find_unrecognized_protocols(
    packets: list,
    entropy_threshold: float = 3.8,
    min_payload_bytes: int = 32,
) -> list[dict[str, Any]]:
    """Find flows that use unknown ports and carry high-entropy payloads.

    High entropy combined with unknown ports may indicate encrypted traffic
    on non-standard ports, command-and-control channels, or custom protocols.

    Parameters
    ----------
    packets : list[Packet]
    entropy_threshold : float
        Shannon entropy above which payload is considered suspicious.
    min_payload_bytes : int
        Minimum payload length to analyse.

    Returns
    -------
    list[dict]
        Each entry: ``src``, ``dst``, ``sport``, ``dport``, ``proto``,
        ``mean_entropy``, ``packet_count``, ``total_bytes``.
    """
    from scapy.layers.inet import TCP, UDP  # type: ignore
    from scapy.packet import Raw  # type: ignore

    flows: dict[tuple, dict] = defaultdict(
        lambda: {"entropies": [], "packets": 0, "bytes": 0}
    )

    for pkt in packets:
        src, dst = _ip_layers(pkt)
        if src is None:
            continue

        proto = sport = dport = None
        if pkt.haslayer(TCP):
            tcp = pkt[TCP]
            proto, sport, dport = "TCP", tcp.sport, tcp.dport
        elif pkt.haslayer(UDP):
            udp = pkt[UDP]
            proto, sport, dport = "UDP", udp.sport, udp.dport
        else:
            continue

        # Skip well-known ports
        if sport in _WELL_KNOWN_PORTS or dport in _WELL_KNOWN_PORTS:
            continue

        payload = bytes(pkt[Raw]) if pkt.haslayer(Raw) else b""
        if len(payload) >= min_payload_bytes:
            ent = _shannon_entropy(payload)
            key = (src, dst, sport, dport, proto)
            flows[key]["entropies"].append(ent)
            flows[key]["packets"] += 1
            flows[key]["bytes"] += len(pkt)

    results = []
    for (src, dst, sport, dport, proto), v in flows.items():
        if not v["entropies"]:
            continue
        mean_ent = sum(v["entropies"]) / len(v["entropies"])
        if mean_ent >= entropy_threshold:
            results.append({
                "src": src,
                "dst": dst,
                "sport": sport,
                "dport": dport,
                "proto": proto,
                "mean_entropy": round(mean_ent, 3),
                "packet_count": v["packets"],
                "total_bytes": v["bytes"],
            })

    return sorted(results, key=lambda r: -r["mean_entropy"])
