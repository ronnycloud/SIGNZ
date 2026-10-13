"""
Evaluation Metrics
===================
Comprehensive metrics for gesture recognition (classification) and
sign-to-sentence translation (NLG), plus system performance metrics.

Metrics are grouped into three categories:
    1. Classification: Accuracy, Precision, Recall, F1, Top-K Accuracy
    2. Translation: WER (Word Error Rate), BLEU-4
    3. System: Latency, FPS, Model size
"""

import time
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


class MetricsCalculator:
    """Calculates and aggregates all evaluation metrics."""

    # ---- Classification Metrics ----

    @staticmethod
    def classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None,
        average: str = "macro",
    ) -> Dict[str, float]:
        """Compute standard classification metrics.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth class indices, shape (N,).
        y_pred : np.ndarray
            Predicted class indices, shape (N,).
        labels : list of str, optional
            Class names for the classification report.
        average : str
            Averaging strategy for multi-class metrics.

        Returns
        -------
        dict
            Keys: accuracy, precision, recall, f1
        """
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
            "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
            "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
        }

    @staticmethod
    def top_k_accuracy(
        y_true: np.ndarray,
        y_probs: np.ndarray,
        k: int = 5,
    ) -> float:
        """Compute top-K accuracy from probability distributions.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth class indices, shape (N,).
        y_probs : np.ndarray
            Predicted probability distributions, shape (N, C).
        k : int
            Number of top predictions to consider.

        Returns
        -------
        float
            Fraction of samples where true class is in top-K predictions.
        """
        top_k_preds = np.argsort(y_probs, axis=1)[:, -k:]
        correct = sum(
            y_true[i] in top_k_preds[i] for i in range(len(y_true))
        )
        return correct / len(y_true)

    @staticmethod
    def per_class_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
    ) -> str:
        """Generate a full per-class classification report.

        Returns
        -------
        str
            Formatted classification report.
        """
        return classification_report(
            y_true, y_pred, target_names=class_names, zero_division=0
        )

    @staticmethod
    def confusion_mat(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        """Compute confusion matrix."""
        return confusion_matrix(y_true, y_pred)

    # ---- Translation Metrics ----

    @staticmethod
    def word_error_rate(reference: str, hypothesis: str) -> float:
        """Compute Word Error Rate (WER) between two sentences.

        WER = (S + D + I) / N
        where S = substitutions, D = deletions, I = insertions,
              N = number of words in reference.

        Uses dynamic programming (Levenshtein distance on words).
        """
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()

        n = len(ref_words)
        m = len(hyp_words)

        # Edge cases
        if n == 0:
            return 1.0 if m > 0 else 0.0

        # DP table
        dp = np.zeros((n + 1, m + 1), dtype=np.int32)
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if ref_words[i - 1] == hyp_words[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],      # Deletion
                        dp[i][j - 1],      # Insertion
                        dp[i - 1][j - 1],  # Substitution
                    )

        return dp[n][m] / n

    @staticmethod
    def batch_wer(references: List[str], hypotheses: List[str]) -> float:
        """Compute average WER over a batch of sentence pairs."""
        wers = [
            MetricsCalculator.word_error_rate(ref, hyp)
            for ref, hyp in zip(references, hypotheses)
        ]
        return np.mean(wers)

    @staticmethod
    def bleu_score(reference: str, hypothesis: str, max_n: int = 4) -> float:
        """Compute BLEU score (up to n-gram) between two sentences.

        A simplified BLEU implementation. For publication, use sacrebleu.

        Parameters
        ----------
        reference : str
            Reference sentence.
        hypothesis : str
            Generated sentence.
        max_n : int
            Maximum n-gram order.

        Returns
        -------
        float
            BLEU score in [0, 1].
        """
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()

        if len(hyp_words) == 0:
            return 0.0

        # Brevity penalty
        bp = min(1.0, np.exp(1 - len(ref_words) / max(len(hyp_words), 1)))

        # N-gram precisions
        precisions = []
        for n in range(1, max_n + 1):
            ref_ngrams = defaultdict(int)
            hyp_ngrams = defaultdict(int)

            for i in range(len(ref_words) - n + 1):
                ngram = tuple(ref_words[i:i + n])
                ref_ngrams[ngram] += 1

            for i in range(len(hyp_words) - n + 1):
                ngram = tuple(hyp_words[i:i + n])
                hyp_ngrams[ngram] += 1

            clipped = sum(
                min(count, ref_ngrams.get(ngram, 0))
                for ngram, count in hyp_ngrams.items()
            )
            total = max(len(hyp_words) - n + 1, 1)
            precisions.append(clipped / total if total > 0 else 0)

        # Geometric mean of precisions (with smoothing for zero precisions)
        log_precisions = [
            np.log(max(p, 1e-10)) for p in precisions
        ]
        avg_log_precision = np.mean(log_precisions)

        return bp * np.exp(avg_log_precision)

    # ---- System Metrics ----

    @staticmethod
    def measure_latency(func, *args, n_runs: int = 100, **kwargs) -> Dict[str, float]:
        """Measure inference latency of a callable.

        Returns
        -------
        dict
            Keys: mean_ms, std_ms, min_ms, max_ms, fps
        """
        # Warmup
        for _ in range(5):
            func(*args, **kwargs)

        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            func(*args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)

        times = np.array(times)
        mean = np.mean(times)
        return {
            "mean_ms": mean,
            "std_ms": np.std(times),
            "min_ms": np.min(times),
            "max_ms": np.max(times),
            "fps": 1000.0 / mean if mean > 0 else 0,
        }

    @staticmethod
    def format_report(
        classification: Dict[str, float],
        top5: float,
        wer: float,
        bleu: float,
        latency: Dict[str, float],
    ) -> str:
        """Format a complete evaluation report as a string."""
        lines = [
            "=" * 60,
            "MTAN-SLR Evaluation Report",
            "=" * 60,
            "",
            "--- Classification Metrics ---",
            f"  Accuracy:      {classification['accuracy']:.4f}",
            f"  Precision:     {classification['precision']:.4f}",
            f"  Recall:        {classification['recall']:.4f}",
            f"  F1 Score:      {classification['f1']:.4f}",
            f"  Top-5 Acc:     {top5:.4f}",
            "",
            "--- Translation Metrics ---",
            f"  WER:           {wer:.4f}",
            f"  BLEU-4:        {bleu:.4f}",
            "",
            "--- System Metrics ---",
            f"  Latency (mean): {latency['mean_ms']:.2f} ms",
            f"  Latency (std):  {latency['std_ms']:.2f} ms",
            f"  FPS:            {latency['fps']:.1f}",
            "=" * 60,
        ]
        return "\n".join(lines)


def evaluate_model(model, dataloader, device):
    """Evaluate a PyTorch model over a dataloader.

    Returns
    -------
    dict
        Keys: accuracy, precision, recall, f1, top5
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    import torch

    with torch.no_grad():
        for seqs, mask, labels in dataloader:
            seqs, mask = seqs.to(device), mask.to(device)
            logits = model(seqs, mask=mask)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.append(probs)

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.vstack(all_probs) if all_probs else np.zeros((0, 1))

    metrics = MetricsCalculator.classification_metrics(y_true, y_pred)
    if len(y_probs) > 0 and y_probs.shape[1] >= 5:
        metrics["top5"] = MetricsCalculator.top_k_accuracy(y_true, y_probs, k=5)
    else:
        metrics["top5"] = metrics["accuracy"]

    return metrics
