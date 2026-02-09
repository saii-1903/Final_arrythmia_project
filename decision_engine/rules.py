
import numpy as np
import uuid
from typing import List, Dict, Any, Optional
from decision_engine.models import Event, EventCategory, DisplayState

# =============================================================================
# 1. RULE-BASED EVENT DERIVATION
# =============================================================================

def derive_rule_events(features: Dict[str, Any]) -> List[Event]:
    """
    Analyzes clinical features to detect arrhythmias strictly via rules.
    Returns a list of 'Rule-Derived Events' (AF, SVT, VT, AV Blocks, Pauses).
    Sinus rhythms are handled as background only and NEVER produce events.
    """
    events = []
    
    # Extract features safely
    hr_val = features.get("mean_hr")
    hr = float(hr_val) if hr_val is not None else 0.0
    
    pr_val = features.get("pr_interval")
    pr = float(pr_val) if pr_val is not None else 0.0
    
    rr_intervals = features.get("rr_intervals_ms", [])
    rr_arr = np.array([])
    if isinstance(rr_intervals, list) and len(rr_intervals) > 2:
        rr_arr = np.array([x for x in rr_intervals if x is not None and isinstance(x, (int, float))])
        
    qrs_mean = 0.0
    try:
        raw_qrs = features.get("qrs_durations_ms")
        if isinstance(raw_qrs, list):
            q_list = [x for x in raw_qrs if isinstance(x, (int, float))]
            if q_list: 
                qrs_mean = float(sum(q_list) / len(q_list))
    except: pass
    
    cv = 0.0
    if len(rr_arr) > 3:
        rr_std = np.std(rr_arr)
        rr_mean = np.mean(rr_arr)
        cv = rr_std / rr_mean if rr_mean > 0 else 0

    # ---------------------------------------------------------
    # 1. Atrial Fibrillation (Irregular + No P-wave)
    # ---------------------------------------------------------
    is_af = False
    if len(rr_arr) > 3:
        p_waves_absent = (pr < 10)
        if cv > 0.15 and p_waves_absent:
            is_af = True
            events.append(Event(
                event_id=str(uuid.uuid4()),
                event_type="AF",
                event_category=EventCategory.RHYTHM,
                start_time=0.0, end_time=10.0,
                rule_evidence={"rule": "AF_Strict", "cv": cv, "pr": pr},
                priority=90,
                used_for_training=True
            ))

    # ---------------------------------------------------------
    # 2. SVT (Regular + Fast + Narrow + Not AF/VT)
    # ---------------------------------------------------------
    # Criteria: Regular (CV < 0.08) + Narrow QRS (< 120ms) + HR > 130 + Not AF
    if hr > 130 and qrs_mean < 120 and cv < 0.08 and not is_af:
        events.append(Event(
            event_id=str(uuid.uuid4()),
            event_type="SVT",
            event_category=EventCategory.RHYTHM,
            start_time=0.0, end_time=10.0,
            rule_evidence={"rule": "SVT_Strict", "hr": hr, "qrs": qrs_mean, "cv": cv},
            priority=80,
            used_for_training=True
        ))

    # ---------------------------------------------------------
    # 3. VT (Wide QRS + HR > 100)
    # ---------------------------------------------------------
    if hr > 100 and qrs_mean >= 120:
        events.append(Event(
            event_id=str(uuid.uuid4()),
            event_type="VT",
            event_category=EventCategory.RHYTHM,
            start_time=0.0, end_time=10.0,
            rule_evidence={"rule": "VT_Rule", "hr": hr, "qrs": qrs_mean},
            priority=100,
            used_for_training=True
        ))

    # ---------------------------------------------------------
    # 4. AV Blocks
    # ---------------------------------------------------------
    if pr > 200:
        events.append(Event(
            event_id=str(uuid.uuid4()),
            event_type="1st Degree AV Block",
            event_category=EventCategory.RHYTHM,
            start_time=0.0, end_time=10.0,
            rule_evidence={"rule": "1stDegBlock", "pr": pr},
            priority=50,
            used_for_training=False # Never train on AV block
        ))

    # ---------------------------------------------------------
    # 5. Pause
    # ---------------------------------------------------------
    if any(rr > 2000 for rr in rr_arr):
        events.append(Event(
            event_id=str(uuid.uuid4()),
            event_type="Pause",
            event_category=EventCategory.RHYTHM,
            start_time=0.0, end_time=10.0,
            rule_evidence={"rule": "Pause_Detected"},
            priority=85,
            used_for_training=False # Never train on Pause
        ))

    return events


# =============================================================================
# 2. ECTOPY PATTERN RECOGNITION
# =============================================================================

def apply_ectopy_patterns(events: List[Event]) -> None:
    """
    Scans ECTOPY events and applies pattern labels (Bigeminy, Trigeminy, Couplet, Triplet).
    Mutates the 'pattern_label' only. Never changes 'event_type'.
    
    NSVT LOGIC:
    - If 3+ consecutive PVCs occur AND instantaneous rate > 100 BPM, create an NSVT Event.
    - Else, label as "Triplet".
    """
    ectopy = sorted(
        [e for e in events if e.event_category == EventCategory.ECTOPY],
        key=lambda e: e.start_time
    )
    
    if len(ectopy) < 2:
        return

    # PVC Pattern Recognition with Time-Gap Constraint
    pvc_clusters = [] 
    current_cluster = []
    MAX_GAP = 1.2 # seconds - medical limit for "consecutive" beats
    
    for i, e in enumerate(ectopy):
        if "PVC" in e.event_type:
            if not current_cluster:
                current_cluster.append(e)
            else:
                # Check time gap with previous PVC
                gap = e.start_time - current_cluster[-1].start_time
                if gap <= MAX_GAP:
                    current_cluster.append(e)
                else:
                    # Gap too large, finalize previous cluster if valid
                    if len(current_cluster) >= 2:
                        pvc_clusters.append(current_cluster)
                    # Start new cluster with current PVC
                    current_cluster = [e]
        else:
            # Non-PVC event breaks the chain
            if len(current_cluster) >= 2:
                pvc_clusters.append(current_cluster)
            current_cluster = []
            
    # Final cleanup for last pending cluster
    if len(current_cluster) >= 2:
        pvc_clusters.append(current_cluster)

    for cluster in pvc_clusters:
        count = len(cluster)
        
        # Calculate Rate
        duration = cluster[-1].start_time - cluster[0].start_time
        # rate = (beats - 1) * 60 / duration
        rate = (count - 1) * (60.0 / duration) if duration > 0 else 0
        
        if count >= 3 and rate > 100:
            # Create NSVT Event
            nsvt = Event(
                event_id=str(uuid.uuid4()),
                event_type="VT", 
                event_category=EventCategory.RHYTHM,
                start_time=cluster[0].start_time,
                end_time=cluster[-1].end_time,
                pattern_label="NSVT",
                rule_evidence={"rule": "NSVT_Detected", "count": count, "rate": round(rate, 1)},
                priority=100,
                used_for_training=True
            )
            events.append(nsvt)
        elif count == 3:
            for e in cluster: e.pattern_label = "Triplet"
        elif count == 2:
            for e in cluster: e.pattern_label = "Couplet"


# =============================================================================
# 3. DISPLAY ARBITRATION RULES
# =============================================================================

def apply_display_rules(background_rhythm: str, events: List[Event]) -> List[Event]:
    """
    Decides which events are displayed vs hidden based on clinical hierarchy.
    Returns the list of events to display (final_display_events).
    Mutates events to set display_state and suppressed_by.
    """
    
    # Hierarchy Flags
    has_life_threatening = any(e.priority >= 95 for e in events) # VF, VT, 3rd Deg
    has_svt_or_block = any(e.priority >= 70 and e.priority < 95 for e in events if "AF" not in e.event_type)
    has_af = any(e.event_type in ["Atrial Fibrillation", "Atrial Flutter"] for e in events)
    has_ectopy = any(e.event_category == EventCategory.ECTOPY for e in events)
    
    # Pass 1: Initial decision making (deterministic)
    for event in events:
        should_display = True
        suppression_reason = None
        
        # Rule A: Life-Threatening (VT, 3rd Deg) always shows.
        if event.priority >= 95:
            should_display = True
            
        # Rule B: AF + Ectopy -> Show Ectopy only (Correction 2)
        elif has_af and has_ectopy and not has_life_threatening and not has_svt_or_block:
            if event.event_type == "AF" or "Flutter" in event.event_type:
                should_display = False
                suppression_reason = "Ectopy Dominance"
            else:
                should_display = True
        
        # Rule B.1: AF Dominance (Show AF, hide other minor rhythms like brady)
        elif has_af:
            if event.event_category == EventCategory.RHYTHM and event.event_type != "AF" and "Flutter" not in event.event_type:
                should_display = False
                suppression_reason = "AF Dominance"
            else:
                should_display = True

        # Rule C: Redundant / Sinus (Sinus is background only)
        elif "Sinus" in event.event_type:
             should_display = False
             suppression_reason = "Background Rhythm"
             
        # Initial assignment
        event.display_state = DisplayState.DISPLAYED if should_display else DisplayState.HIDDEN
        event.suppressed_by = suppression_reason

    # Pass 2: Artifact Suppression (Correction 4)
    # Artifact displayed ONLY if no other displayed events exist.
    displayed_count = sum(1 for e in events if e.display_state == DisplayState.DISPLAYED and e.event_type != "Artifact")
    
    for event in events:
        if event.event_type == "Artifact":
            if displayed_count > 0:
                event.display_state = DisplayState.HIDDEN
                event.suppressed_by = "Arrhythmia Presence"
            else:
                event.display_state = DisplayState.DISPLAYED
                event.suppressed_by = None
    
    # Pass 3: Final refinement & Sorting
    # We sort by priority (descending) so the dashboard shows the most critical event first.
    final_list = [e for e in events if e.display_state == DisplayState.DISPLAYED]
    final_list.sort(key=lambda x: x.priority, reverse=True)
    
    return final_list


# =============================================================================
# 4. TRAINING FLAG LOGIC
# =============================================================================

def apply_training_flags(events: List[Event]) -> None:
    """
    Sets used_for_training flag based on event type.
    Train ONLY on: PAC, PVC, AF, VT, SVT.
    Never train on: Sinus, Artifact, Pause, AV block.
    """
    training_set = {
        "PAC", "PVC", 
        "AF", "Atrial Fibrillation", 
        "VT", "Ventricular Tachycardia", 
        "SVT", "Supraventricular Tachycardia"
    }
    for event in events:
        if event.event_type in training_set:
            event.used_for_training = True
        else:
            event.used_for_training = False
