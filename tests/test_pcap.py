"""
Tests for pegasource.pcap — reader, stats, and pattern detectors.

Uses scapy to synthesise packets in memory (no .pcap file needed).
"""

import time
import pytest

try:
    from scapy.all import (  # type: ignore
        Ether, IP, IPv6, TCP, UDP, ICMP, DNS, DNSQR, DNSRR, Raw, Packet,
        wrpcap, IP, conf
    )
    from scapy.packet import NoPayload
    HAS_SCAPY = True
except ImportError:
    HAS_SCAPY = False

pytestmark = pytest.mark.skipif(not HAS_SCAPY, reason="scapy not installed")


# ---------------------------------------------------------------------------
# Helpers — build synthetic packet lists
# ---------------------------------------------------------------------------

def _tcp_pkt(src, dst, sport, dport, flags="S", t=None, payload=b""):
    pkt = IP(src=src, dst=dst) / TCP(sport=sport, dport=dport, flags=flags)
    if payload:
        pkt = pkt / Raw(load=payload)
    if t is not None:
        pkt.time = t
    else:
        pkt.time = time.time()
    return pkt


def _udp_pkt(src, dst, sport, dport, t=None, payload=b""):
    pkt = IP(src=src, dst=dst) / UDP(sport=sport, dport=dport)
    if payload:
        pkt = pkt / Raw(load=payload)
    if t is not None:
        pkt.time = t
    else:
        pkt.time = time.time()
    return pkt


def _dns_query(src, qname, rcode=0, t=None):
    pkt = (
        IP(src=src, dst="8.8.8.8")
        / UDP(sport=5353, dport=53)
        / DNS(rd=1, qd=DNSQR(qname=qname))
    )
    pkt.time = t or time.time()
    return pkt


def _dns_response(src, qname, rcode=0, t=None):
    pkt = (
        IP(src="8.8.8.8", dst=src)
        / UDP(sport=53, dport=5353)
        / DNS(qr=1, rd=1, ra=1, rcode=rcode, qd=DNSQR(qname=qname))
    )
    pkt.time = t or time.time()
    return pkt


# ---------------------------------------------------------------------------
# Stats tests
# ---------------------------------------------------------------------------

class TestProtocolDistribution:
    def test_tcp_counted(self):
        from pegasource.pcap.stats import protocol_distribution
        pkts = [_tcp_pkt("1.1.1.1", "2.2.2.2", 1234, 80) for _ in range(5)]
        dist = protocol_distribution(pkts)
        assert "TCP" in dist
        assert dist["TCP"] == 5

    def test_mixed_protocols(self):
        from pegasource.pcap.stats import protocol_distribution
        pkts = (
            [_tcp_pkt("1.1.1.1", "2.2.2.2", 1234, 80)] * 3
            + [_udp_pkt("1.1.1.1", "2.2.2.2", 1111, 53)] * 2
        )
        dist = protocol_distribution(pkts)
        assert dist.get("TCP", 0) >= 3


class TestTopTalkers:
    def test_returns_dataframe(self):
        from pegasource.pcap.stats import top_talkers
        import pandas as pd
        pkts = [_tcp_pkt("10.0.0.1", "10.0.0.2", 1000 + i, 80) for i in range(5)]
        df = top_talkers(pkts)
        assert isinstance(df, pd.DataFrame)

    def test_top_by_packets(self):
        from pegasource.pcap.stats import top_talkers
        pkts = (
            [_tcp_pkt("10.0.0.1", "10.0.0.2", 1000, 80)] * 10
            + [_tcp_pkt("10.0.0.2", "10.0.0.3", 2000, 443)] * 3
        )
        df = top_talkers(pkts, by="packets")
        assert df.iloc[0]["src"] == "10.0.0.1"


class TestConversationTable:
    def test_returns_dataframe(self):
        from pegasource.pcap.stats import conversation_table
        pkts = [_tcp_pkt("1.1.1.1", "2.2.2.2", 5000, 80, t=float(i)) for i in range(5)]
        df = conversation_table(pkts)
        assert len(df) >= 1

    def test_duration_computed(self):
        from pegasource.pcap.stats import conversation_table
        pkts = [_tcp_pkt("1.1.1.1", "2.2.2.2", 5000, 80, t=float(i)) for i in range(5)]
        df = conversation_table(pkts)
        assert (df["duration_s"] >= 0).all()


# ---------------------------------------------------------------------------
# Pattern detector tests
# ---------------------------------------------------------------------------

class TestPortScan:
    def _syn_burst(self, src, dst, n_ports=25, base_time=0.0):
        return [
            _tcp_pkt(src, dst, 10000 + i, 1000 + i, flags="S", t=base_time + i * 0.1)
            for i in range(n_ports)
        ]

    def test_detects_scan(self):
        from pegasource.pcap.patterns import detect_port_scan
        pkts = self._syn_burst("10.0.0.1", "192.168.1.1", n_ports=30)
        results = detect_port_scan(pkts, threshold=20, window_s=60.0)
        assert len(results) > 0
        assert results[0]["src"] == "10.0.0.1"
        assert results[0]["dst"] == "192.168.1.1"

    def test_no_scan_below_threshold(self):
        from pegasource.pcap.patterns import detect_port_scan
        pkts = self._syn_burst("10.0.0.1", "192.168.1.1", n_ports=5)
        results = detect_port_scan(pkts, threshold=20)
        assert len(results) == 0


class TestBeaconing:
    def _periodic_pkts(self, src, dst, interval=5.0, n=10):
        return [
            _tcp_pkt(src, dst, 50000, 443, t=float(i) * interval)
            for i in range(n)
        ]

    def test_detects_beaconing(self):
        from pegasource.pcap.patterns import detect_beaconing
        pkts = self._periodic_pkts("10.1.1.1", "1.2.3.4", interval=10.0, n=15)
        results = detect_beaconing(pkts, min_occurrences=5, jitter_tolerance_s=1.0)
        assert len(results) > 0
        assert results[0]["src"] == "10.1.1.1"

    def test_no_beaconing_random(self):
        from pegasource.pcap.patterns import detect_beaconing
        import random
        rng = random.Random(42)
        pkts = [
            _tcp_pkt("10.0.0.1", "1.2.3.4", 50000, 443, t=rng.uniform(0, 300))
            for _ in range(20)
        ]
        results = detect_beaconing(pkts, min_occurrences=5, jitter_tolerance_s=0.5)
        assert len(results) == 0


class TestDNSAnomalies:
    def test_nxdomain_rate(self):
        from pegasource.pcap.patterns import detect_dns_anomalies
        pkts = (
            [_dns_response("10.0.0.1", "example.com", rcode=3)] * 8  # NXDOMAIN
            + [_dns_response("10.0.0.1", "google.com", rcode=0)] * 2  # OK
        )
        results = detect_dns_anomalies(pkts)
        assert any(r["type"] == "HIGH_NXDOMAIN_RATE" for r in results)

    def test_high_entropy_label(self):
        from pegasource.pcap.patterns import detect_dns_anomalies
        # Base64-like label with high entropy
        long_label = "aB3xY9zQm2pLwR7vKn4dJs6tHcFe0gU1" * 1
        pkts = [_dns_query("10.0.0.5", f"{long_label}.evil.com")]
        results = detect_dns_anomalies(pkts)
        assert any(r["type"] == "HIGH_ENTROPY_DNS_LABEL" for r in results)


class TestLargeTransfers:
    def test_detects_large_transfer(self):
        from pegasource.pcap.patterns import detect_large_transfers
        payload = b"X" * 60000
        pkts = [
            _tcp_pkt("10.0.0.1", "10.0.0.2", 9000, 80, payload=payload)
            for _ in range(20)
        ]
        results = detect_large_transfers(pkts, bytes_threshold=100_000)
        assert len(results) > 0
        assert results[0]["src"] == "10.0.0.1"


class TestUnrecognizedProtocols:
    def test_high_entropy_unknown_port(self):
        from pegasource.pcap.patterns import find_unrecognized_protocols
        # High-entropy payload on a non-standard port
        import os
        payload = os.urandom(200)  # random bytes → high entropy
        pkts = [
            _tcp_pkt("10.0.0.1", "10.0.0.2", 49000, 31337, payload=payload)
            for _ in range(5)
        ]
        results = find_unrecognized_protocols(pkts, entropy_threshold=3.0, min_payload_bytes=10)
        assert len(results) > 0

    def test_well_known_port_ignored(self):
        from pegasource.pcap.patterns import find_unrecognized_protocols
        import os
        payload = os.urandom(200)
        pkts = [
            _tcp_pkt("10.0.0.1", "10.0.0.2", 49000, 443, payload=payload)
        ]
        results = find_unrecognized_protocols(pkts, entropy_threshold=3.0, min_payload_bytes=10)
        assert len(results) == 0
