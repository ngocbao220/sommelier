from __future__ import annotations

from pipeline_debug.contracts import Segment


def detect_overlaps(segments: list[Segment], threshold_seconds: float) -> list[dict]:
    overlaps: list[dict] = []
    ordered = sorted(segments, key=lambda item: (item.start, item.end))
    for i, first in enumerate(ordered):
        for second in ordered[i + 1:]:
            if second.start >= first.end:
                break
            overlap_start = max(first.start, second.start)
            overlap_end = min(first.end, second.end)
            duration = overlap_end - overlap_start
            if duration >= threshold_seconds:
                overlaps.append(
                    {
                        "seg1": first.index,
                        "seg2": second.index,
                        "speaker1": first.speaker,
                        "speaker2": second.speaker,
                        "overlap_start": round(overlap_start, 3),
                        "overlap_end": round(overlap_end, 3),
                        "overlap_duration": round(duration, 3),
                    }
                )
    return overlaps

