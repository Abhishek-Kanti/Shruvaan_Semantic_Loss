import re

class Probator:
    def __init__(self, logger=None):
        self.logger = logger

    def _ngram_overlap(self, text1, text2, n=3):
        ngrams1 = [text1[i:i+n] for i in range(len(text1)-n+1)]
        ngrams2 = [text2[i:i+n] for i in range(len(text2)-n+1)]
        set1, set2 = set(ngrams1), set(ngrams2)
        if not set1 or not set2:
            return 0.0
        return len(set1 & set2) / len(set1 | set2)

    def _prefix_leakage(self, text1, text2, prefix_len=5):
        return 1.0 if text1[:prefix_len] == text2[:prefix_len] else 0.0

    def _inject_probe(self, plaintext, ciphertext):
        for word in plaintext.split():
            if word in ciphertext:
                return 1.0
        return 0.0

    def _scaffold_probe(self, plaintext, ciphertext):
        digits_plain = sum(c.isdigit() for c in plaintext)
        digits_cipher = sum(c.isdigit() for c in ciphertext)
        return 1.0 if digits_plain == digits_cipher else 0.5

    def probe(self, plaintext, ciphertext):
        results = {
            "L_ngram": self._ngram_overlap(plaintext, ciphertext),
            "L_prefix": self._prefix_leakage(plaintext, ciphertext),
            "L_inject": self._inject_probe(plaintext, ciphertext),
            "L_scaffold": self._scaffold_probe(plaintext, ciphertext)
        }
        results["Rprob"] = sum(results.values()) / len(results)

        if self.logger:
            self.logger.log("probator", "probe_results", results)

        return results


def run_probator(plaintext_dict, ciphertext_dict, logger=None, history_logger=None):
    plaintext_str = str(plaintext_dict)
    ciphertext_str = str(ciphertext_dict)
    return Probator(logger).probe(plaintext_str, ciphertext_str)
