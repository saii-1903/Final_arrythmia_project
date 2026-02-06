# IMPLEMENTATION PLAN: PHASE 2 - CLINICAL RULES REFACTOR

The goal of this phase is to move **Domain Logic** into `decision_engine/rules.py` and ensure `xai.py` acts purely as an explainer. The `RhythmOrchestrator` will coordinate the application of these rules.

## 1. MIGRATION FROM XAI.PY
The following logic currently in `xai.py`'s `_apply_clinical_rules` will be extracted and refactored into modular functions in `rules.py`.

- **Basic Rhythm Rules**:
    - High Irregularity -> AFib
    - Regular + HR > 150 -> SVT
    - Low HR -> Bradycardia
    - AV Blocks (3rd Degree, 2nd Type II, Wenckebach)
- **Ectopy Patterns**:
    - PVC Bigeminy, Trigeminy
    - PVC Couplets, Triplets
    - PAC Trigeminy
    - Runs (Ventricular/Atrial)

### Action Items:
- [ ] Create `decision_engine/rules.py`
- [ ] Implement `apply_clinical_rules(features)` in `rules.py` (migrated logic).
- [ ] Empty `_apply_clinical_rules` in `xai.py` or modify it to act as a wrapper/stub for legacy calls (or remove if Orchestrator handles it entirely).

## 2. NEW FILE: `decision_engine/rules.py` Structure

This file will contain three distinct layers of logic:

### A. Core Pattern Detection (`apply_ectopy_patterns`)
This function iterates over `decision.events` (specifically ectopy events) and looks for temporal patterns. It assigns `pattern_label` metadata to the individual events involved.

**Logic Maps:**
- **Bigeminy**: Alternating rhythm (Normal, PVC, Normal, PVC).
- **Trigeminy**: Every 3rd beat (Normal, Normal, PVC).
- **Couplet**: Two consecutive ectopics.
- **Run/NSVT**: >= 3 consecutive ectopics.

*Note: This requires access to beat indices and types. We will approximate using the event timestamps/indices if beat-level data is sparse, or explicit beat lists if available.*

### B. Display Arbitration (`apply_display_rules`)
This function receives the `background_rhythm` and the list of `events`. It filters the `events` into `final_display_events` based on clinical priority.

**Rules:**
1.  **Life-Threatening Overrides** (VT, VF, 3rd Deg Block):
    - ALWAYS ensure these are displayed.
    - If present, they might suppress minor ectopy if it reduces clutter, but usually, we show everything relevant.
2.  **AF / Flutter Dominance**:
    - If `Detect(AF)` is positive:
        - The BACKGROUND is AF.
        - "Rhythm Events" (like "Sinus Rhythm" mislabeled) are HIDDEN.
        - Ectopy (PVCs) are SHOWN.
3.  **Clean Sinus**:
    - If Background is Sinus and No Ectopy -> Show "Sinus Rhythm" (or just implies background).
4.  **Pure Ectopy**:
    - Background Sinus + PVCs -> Show PVCs (highlighted).

### C. Training Flag Logic
A helper to enforce:
- **Sinus** -> `used_for_training = False` (retraining focuses on pathology).
- **Artifact** -> `used_for_training = False`.
- **Suppressed Events** -> `used_for_training = True` (we still want to learn them even if UI hides them).

## 3. UPDATE `rhythm_orchestrator.py`
The `decide()` method will be updated to:
1.  Initialize `SegmentDecision`.
2.  Run strict `SQI` check.
3.  Detect `Background Rhythm`.
4.  **NEW**: Call `rules.apply_clinical_rules(features)` to get High-Confidence Rule Events (e.g. AV Block detected via PR interval).
    - Add these as `Event` objects.
5.  Mix in ML-detected `Events`.
6.  **NEW**: Call `rules.apply_ectopy_patterns(events)`.
7.  **NEW**: Call `rules.apply_display_rules(background, events)` to populate `final_display_events`.
8.  Return `SegmentDecision`.

## 4. XAI REFACTOR
- `xai.py` will no longer calculate rules.
- It will imply read `Event.rule_evidence` from the passed segment data (if we update the API) or we keep a reduced "Explainer" version that just translates the decision into text.

## 5. EXAMPLE SCENARIOS
- **Input**: ML says "Sinus", Rule says "AFib" (High CV).
    - **Orchestrator**: Creates AFib Event (Priority 90) via Rule. ML Sinus is ignored/suppressed.
    - **Display**: Background = AFib. Display = AFib Event.
- **Input**: ML says "PVC" x3.
    - **Orchestrator**: Creates 3 PVC Events.
    - **Pattern Rule**: Detects sequential timestamps -> Labels them "PVC Triplet" / "VT Warning".
    - **Display**: Highlights the run.

## 6. NEXT STEPS
1.  **User Approval**: Confirm this plan matches Phase 2 expectations.
2.  **Execution**: Write `rules.py` and update `rhythm_orchestrator.py`.
