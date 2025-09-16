import re
from typing import List, Dict

SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')
CITE = re.compile(r'\[(\d+)\]')

class Ledger:
    """
    φ: sentence -> set(evidence_ids). See docs/MATH.md (Provenance ledger).
    """
    def __init__(self):
        self.sentences: List[str] = []
        self.citations: Dict[int, List[int]] = {}   # sent_idx -> [evidence_id,...]

    def record_section(self, text: str):
        base_idx = len(self.sentences)
        for s in SENT_SPLIT.split(text.strip()):
            if not s: 
                continue
            ev_ids = [int(x) for x in CITE.findall(s)]
            self.sentences.append(s)
            self.citations[base_idx] = ev_ids
            base_idx += 1

    def mcr(self) -> float:
        if not self.sentences:
            return 0.0
        missing = sum(1 for idx in range(len(self.sentences)) if not self.citations.get(idx))
        return missing / len(self.sentences)
