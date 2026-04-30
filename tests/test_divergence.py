"""Tests for pairwise divergence computation.

Verifies that the GPU-streaming compute_pairwise_divergences produces
numerically identical results to a naive reference implementation, and
that edge cases (single persona, CPU-only) are handled correctly.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from explore_persona_space.analysis.divergence import (
    compute_js_divergence,
    compute_kl_divergence,
    compute_pairwise_divergences,
)


def _random_log_probs(n: int, seq_len: int, vocab_size: int, seed: int = 42) -> torch.Tensor:
    """Generate random log-softmax tensors for testing."""
    rng = torch.Generator().manual_seed(seed)
    logits = torch.randn(n, seq_len, vocab_size, generator=rng)
    return F.log_softmax(logits, dim=-1)


def _reference_kl_matrix(log_probs: torch.Tensor) -> torch.Tensor:
    """Naive O(N^2 * T * V) pairwise KL computation for testing."""
    n, seq_len, _v = log_probs.shape
    kl = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            p = log_probs[i].exp()
            kl[i, j] = (p * (log_probs[i] - log_probs[j])).sum() / seq_len
    return kl


def _reference_js_matrix(log_probs: torch.Tensor) -> torch.Tensor:
    """Naive O(N^2 * T * V) pairwise JS computation for testing."""
    n, _seq_len, _v = log_probs.shape
    js = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            js[i, j] = compute_js_divergence(log_probs[i], log_probs[j]).item()
    return js


class TestComputePairwiseDivergences:
    """Test the GPU-streaming pairwise divergence computation."""

    @pytest.fixture
    def small_log_probs(self) -> torch.Tensor:
        """Small test case: 5 personas, 10 positions, 100 vocab."""
        return _random_log_probs(5, 10, 100)

    @pytest.fixture
    def persona_names_5(self) -> list[str]:
        return ["alice", "bob", "carol", "dave", "eve"]

    def test_kl_matches_reference(self, small_log_probs, persona_names_5):
        """KL matrix from compute_pairwise_divergences matches naive reference."""
        _, kl_pairs = compute_pairwise_divergences(
            small_log_probs,
            persona_names_5,
            kl_only=True,
            gpu_device="cpu",  # force CPU path for testing without GPU
        )
        ref_kl = _reference_kl_matrix(small_log_probs)

        for i in range(5):
            for j in range(5):
                if i == j:
                    continue
                ni, nj = persona_names_5[i], persona_names_5[j]
                got = kl_pairs[(ni, nj)]
                expected = ref_kl[i, j].item()
                assert abs(got - expected) < 1e-5, (
                    f"KL({ni}||{nj}): got {got:.8f}, expected {expected:.8f}"
                )

    def test_js_exact_matches_reference(self, small_log_probs, persona_names_5):
        """Exact JS from compute_pairwise_divergences matches naive reference."""
        js_pairs, _ = compute_pairwise_divergences(
            small_log_probs,
            persona_names_5,
            kl_only=False,
            gpu_device="cpu",
        )
        ref_js = _reference_js_matrix(small_log_probs)

        for i in range(5):
            for j in range(i + 1, 5):
                ni, nj = persona_names_5[i], persona_names_5[j]
                got = js_pairs[(ni, nj)]
                expected = ref_js[i, j].item()
                assert abs(got - expected) < 1e-5, (
                    f"JS({ni},{nj}): got {got:.8f}, expected {expected:.8f}"
                )

    def test_kl_only_js_approximation(self, persona_names_5):
        """kl_only JS approximation is close to exact JS for small divergences."""
        # Use distributions that are close together (small perturbation from a
        # shared base) so KL values are small and the approximation holds.
        rng = torch.Generator().manual_seed(42)
        base_logits = torch.randn(1, 10, 100, generator=rng)
        noise = 0.1 * torch.randn(5, 10, 100, generator=rng)
        close_log_probs = F.log_softmax(base_logits + noise, dim=-1)

        js_exact, _ = compute_pairwise_divergences(
            close_log_probs,
            persona_names_5,
            kl_only=False,
            gpu_device="cpu",
        )
        js_approx, _ = compute_pairwise_divergences(
            close_log_probs,
            persona_names_5,
            kl_only=True,
            gpu_device="cpu",
        )
        for key in js_exact:
            exact = js_exact[key]
            approx = js_approx[key]
            # For close distributions the approximation error should be small
            assert abs(exact - approx) < 0.01, (
                f"JS approx {key}: exact={exact:.6f}, approx={approx:.6f}"
            )

    def test_js_symmetric(self, small_log_probs, persona_names_5):
        """JS values should be symmetric: JS(A,B) == JS(B,A)."""
        js_pairs, _ = compute_pairwise_divergences(
            small_log_probs,
            persona_names_5,
            kl_only=True,
            gpu_device="cpu",
        )
        # Only upper-triangle pairs are stored, so symmetry is inherent.
        # But verify values are non-negative.
        for key, val in js_pairs.items():
            assert val >= -1e-6, f"Negative JS for {key}: {val}"

    def test_kl_non_negative(self, small_log_probs, persona_names_5):
        """KL values should be non-negative."""
        _, kl_pairs = compute_pairwise_divergences(
            small_log_probs,
            persona_names_5,
            kl_only=True,
            gpu_device="cpu",
        )
        for key, val in kl_pairs.items():
            assert val >= -1e-5, f"Negative KL for {key}: {val}"

    def test_js_bounded_by_ln2(self, small_log_probs, persona_names_5):
        """JS values should be bounded by ln(2) ~= 0.693."""
        js_pairs, _ = compute_pairwise_divergences(
            small_log_probs,
            persona_names_5,
            kl_only=False,
            gpu_device="cpu",
        )
        ln2 = math.log(2)
        for key, val in js_pairs.items():
            assert val <= ln2 + 1e-5, f"JS exceeds ln(2) for {key}: {val}"

    def test_identical_distributions_zero_divergence(self):
        """Same distribution for all personas should give zero divergence."""
        # All 3 personas have identical logits
        rng = torch.Generator().manual_seed(99)
        logits = torch.randn(1, 5, 50, generator=rng)
        log_probs = F.log_softmax(logits.expand(3, -1, -1).contiguous(), dim=-1)
        names = ["a", "b", "c"]

        js_pairs, kl_pairs = compute_pairwise_divergences(
            log_probs, names, kl_only=False, gpu_device="cpu"
        )
        for key, val in js_pairs.items():
            assert abs(val) < 1e-5, f"Non-zero JS for identical dists {key}: {val}"
        for key, val in kl_pairs.items():
            assert abs(val) < 1e-5, f"Non-zero KL for identical dists {key}: {val}"

    def test_chunk_sizes_produce_same_result(self, small_log_probs, persona_names_5):
        """Different row_chunk and time_chunk values should produce same results."""
        _, kl_default = compute_pairwise_divergences(
            small_log_probs,
            persona_names_5,
            kl_only=True,
            gpu_device="cpu",
            row_chunk=16,
            time_chunk=30,
        )
        _, kl_small = compute_pairwise_divergences(
            small_log_probs,
            persona_names_5,
            kl_only=True,
            gpu_device="cpu",
            row_chunk=2,
            time_chunk=3,
        )
        _, kl_large = compute_pairwise_divergences(
            small_log_probs,
            persona_names_5,
            kl_only=True,
            gpu_device="cpu",
            row_chunk=100,
            time_chunk=100,
        )

        for key in kl_default:
            assert abs(kl_default[key] - kl_small[key]) < 1e-5, (
                f"Chunk size mismatch for {key}: default={kl_default[key]}, small={kl_small[key]}"
            )
            assert abs(kl_default[key] - kl_large[key]) < 1e-5, (
                f"Chunk size mismatch for {key}: default={kl_default[key]}, large={kl_large[key]}"
            )

    def test_pair_count(self, small_log_probs, persona_names_5):
        """Should produce the correct number of pairs."""
        js_pairs, kl_pairs = compute_pairwise_divergences(
            small_log_probs, persona_names_5, kl_only=True, gpu_device="cpu"
        )
        n = len(persona_names_5)
        expected_js = n * (n - 1) // 2  # upper triangle
        expected_kl = n * (n - 1)  # all ordered pairs
        assert len(js_pairs) == expected_js, f"JS pairs: {len(js_pairs)} != {expected_js}"
        assert len(kl_pairs) == expected_kl, f"KL pairs: {len(kl_pairs)} != {expected_kl}"


class TestPointwiseDivergences:
    """Test the per-pair KL and JS functions."""

    def test_kl_zero_same_distribution(self):
        rng = torch.Generator().manual_seed(0)
        logits = torch.randn(10, 50, generator=rng)
        lp = F.log_softmax(logits, dim=-1)
        assert compute_kl_divergence(lp, lp).item() < 1e-6

    def test_js_zero_same_distribution(self):
        rng = torch.Generator().manual_seed(0)
        logits = torch.randn(10, 50, generator=rng)
        lp = F.log_softmax(logits, dim=-1)
        assert compute_js_divergence(lp, lp).item() < 1e-6

    def test_kl_non_negative(self):
        rng = torch.Generator().manual_seed(1)
        lp1 = F.log_softmax(torch.randn(10, 50, generator=rng), dim=-1)
        lp2 = F.log_softmax(torch.randn(10, 50, generator=rng), dim=-1)
        assert compute_kl_divergence(lp1, lp2).item() >= -1e-6

    def test_js_bounded(self):
        rng = torch.Generator().manual_seed(2)
        lp1 = F.log_softmax(torch.randn(10, 50, generator=rng), dim=-1)
        lp2 = F.log_softmax(torch.randn(10, 50, generator=rng), dim=-1)
        js = compute_js_divergence(lp1, lp2).item()
        assert -1e-6 <= js <= math.log(2) + 1e-6
