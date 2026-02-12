from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum

class SegmentState(Enum):
    WARMUP = "WARMUP"
    UNRELIABLE = "UNRELIABLE"
    ANALYZED = "ANALYZED"

class EventCategory(Enum):
    RHYTHM = "RHYTHM"
    ECTOPY = "ECTOPY"

class DisplayState(Enum):
    DISPLAYED = "DISPLAYED"
    HIDDEN = "HIDDEN"

@dataclass
class Event:
    """
    Represents a single arrhythmia occurrence or significant ECG event.
    
    This object enables:
    - Individual highlighting in the dashboard.
    - Targeted retraining by identifying specific events.
    - Correct suppression or display logic.
    """
    event_id: str
    event_type: str  # e.g., 'PVC', 'PAC', 'AF', 'VT', 'Sinus Bradycardia'
    event_category: EventCategory
    start_time: float  # Start time in seconds relative to recording start
    end_time: float    # End time in seconds relative to recording start
    beat_indices: List[int] = field(default_factory=list) # Indices of beats involved
    ml_evidence: Dict[str, Any] = field(default_factory=dict) # Raw outputs from ML models
    rule_evidence: Dict[str, Any] = field(default_factory=dict) # Clinical rule match details
    pattern_label: Optional[str] = None  # e.g., 'bigeminy', 'trigeminy', 'couplet', 'run'
    priority: int = 0
    suppressed_by: Optional[str] = None
    display_state: DisplayState = DisplayState.DISPLAYED
    annotation_source: Optional[str] = None # e.g., 'cardiologist' or 'ai'
    used_for_training: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API/JSON serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "event_category": self.event_category.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "beat_indices": self.beat_indices,
            "ml_evidence": self.ml_evidence,
            "rule_evidence": self.rule_evidence,
            "pattern_label": self.pattern_label,
            "priority": self.priority,
            "suppressed_by": self.suppressed_by,
            "display_state": self.display_state.value,
            "annotation_source": self.annotation_source,
            "used_for_training": self.used_for_training
        }

@dataclass
class SegmentDecision:
    """
    The single source of truth for all decisions made on a specific ECG segment.
    
    Aggregates background rhythm and all detected events after arbitration.
    """
    segment_index: int
    segment_state: SegmentState
    background_rhythm: str
    events: List[Event] = field(default_factory=list)
    final_display_events: List[Event] = field(default_factory=list)
    xai_notes: Dict[str, Any] = field(default_factory=dict) # Structured notes/explanations

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API/JSON serialization."""
        return {
            "segment_index": self.segment_index,
            "segment_state": self.segment_state.value,
            "background_rhythm": self.background_rhythm,
            "events": [e.to_dict() for e in self.events],
            "final_display_events": [e.to_dict() for e in self.final_display_events],
            "xai_notes": self.xai_notes
        }
