from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from qtpy.QtWidgets import QPushButton

from napari_deeplabcut.tracking.core.merge import TrackingMergeConflictEntry, TrackingMergePolicy, TrackingMergePreview
from napari_deeplabcut.tracking.ui import merger as merger_ui

# -----------------------------------------------------------------------------#
# Small helpers
# -----------------------------------------------------------------------------#


def _make_preview(
    *,
    policy=TrackingMergePolicy.FILL_MISSING,
    is_valid=True,
    invalid_reason=None,
    n_appendable=0,
    n_identical=0,
    n_conflicts=0,
    n_overwriteable=0,
    n_invalid_source=0,
    conflicts=(),
    overwrites=(),
    truncated_conflicts=0,
    truncated_overwrites=0,
    source_layer_name="source",
    target_layer_name="target",
):
    # Fingerprints are not relevant for UI-only tests
    dummy_fp = SimpleNamespace()

    return TrackingMergePreview(
        source_layer_name=source_layer_name,
        target_layer_name=target_layer_name,
        policy=policy,
        source_fingerprint=dummy_fp,
        target_fingerprint=dummy_fp,
        n_source_rows=0,
        n_appendable=n_appendable,
        n_identical=n_identical,
        n_conflicts=n_conflicts,
        n_overwriteable=n_overwriteable,
        n_invalid_source=n_invalid_source,
        has_source_duplicates=False,
        has_target_duplicates=False,
        is_valid=is_valid,
        invalid_reason=invalid_reason,
        append_source_indices=(),
        identical_source_indices=(),
        conflict_source_indices=(),
        overwrite_source_indices=(),
        invalid_source_indices=(),
        conflicts=tuple(conflicts),
        overwrites=tuple(overwrites),
        truncated_conflicts=truncated_conflicts,
        truncated_overwrites=truncated_overwrites,
    )


def _make_conflict_entry():
    return TrackingMergeConflictEntry(
        frame_label="5",
        keypoint_label="nose (id: animal-a)",
        source_coords_text="(x=20.000, y=10.000)",
        target_coords_text="(x=21.000, y=11.000)",
    )


# -----------------------------------------------------------------------------#
# Pure helper tests
# -----------------------------------------------------------------------------#


def test_layer_display_name_handles_none_and_named_layer(fake_points_layer_factory):
    assert merger_ui._layer_display_name(None) == "—"

    layer = fake_points_layer_factory(
        name="Tracked result",
        data=[[0, 10, 20]],
        labels=["nose"],
        ids=[""],
    )
    assert merger_ui._layer_display_name(layer) == "Tracked result"


def test_preview_summary_text_for_none_and_invalid_preview():
    assert "Choose a tracking-result source layer" in merger_ui._preview_summary_text(None)

    preview = _make_preview(is_valid=False, invalid_reason="bad merge")
    text = merger_ui._preview_summary_text(preview)
    assert "Merge unavailable" in text
    assert "bad merge" in text


def test_preview_summary_text_fill_missing_mentions_skipped_conflicts():
    preview = _make_preview(
        policy=TrackingMergePolicy.FILL_MISSING,
        n_appendable=2,
        n_conflicts=3,
        n_identical=1,
        n_invalid_source=1,
    )

    text = merger_ui._preview_summary_text(preview)

    assert '2 tracked points will be added to "target".' in text
    assert "3 conflicting points will be skipped." in text
    assert "1 point already match the target exactly." in text or "1 point already matches the target exactly." in text
    assert "1 invalid source row will be ignored." in text


def test_preview_summary_text_overwrite_policy_mentions_overwrites():
    preview = _make_preview(
        policy=TrackingMergePolicy.OVERWRITE_EXISTING,
        n_appendable=1,
        n_overwriteable=2,
    )

    text = merger_ui._preview_summary_text(preview)

    assert '1 tracked point will be added to "target".' in text
    assert "2 existing target points will be overwritten by tracked coordinates." in text


def test_preview_details_text_uses_conflicts_or_overwrites():
    entry = _make_conflict_entry()

    fill_preview = _make_preview(
        policy=TrackingMergePolicy.FILL_MISSING,
        conflicts=(entry,),
    )
    overwrite_preview = _make_preview(
        policy=TrackingMergePolicy.OVERWRITE_EXISTING,
        overwrites=(entry,),
    )

    fill_text = merger_ui._preview_details_text(fill_preview)
    overwrite_text = merger_ui._preview_details_text(overwrite_preview)

    assert "5 → nose (id: animal-a)" in fill_text
    assert "source: (x=20.000, y=10.000)" in fill_text
    assert "target: (x=21.000, y=11.000)" in fill_text

    assert "5 → nose (id: animal-a)" in overwrite_text
    assert "source: (x=20.000, y=10.000)" in overwrite_text
    assert "target: (x=21.000, y=11.000)" in overwrite_text


# -----------------------------------------------------------------------------#
# Review dialog tests
# -----------------------------------------------------------------------------#


@pytest.mark.usefixtures("qtbot")
def test_review_dialog_titles_and_buttons_change_with_mode(qtbot):
    preview = _make_preview(
        policy=TrackingMergePolicy.FILL_MISSING,
        n_conflicts=1,
        conflicts=(_make_conflict_entry(),),
    )

    review_dlg = merger_ui.TrackingMergeReviewDialog(
        parent=None,
        preview=preview,
        review_only=True,
    )
    qtbot.addWidget(review_dlg)

    assert review_dlg.windowTitle() == "Review merge changes"
    close_buttons = [b for b in review_dlg.findChildren(QPushButton) if b.text() == "Close"]
    assert len(close_buttons) == 1

    confirm_dlg = merger_ui.TrackingMergeReviewDialog(
        parent=None,
        preview=preview,
        review_only=False,
    )
    qtbot.addWidget(confirm_dlg)

    assert confirm_dlg.windowTitle() == "Confirm merge"
    merge_buttons = [b for b in confirm_dlg.findChildren(QPushButton) if b.text() == "Merge non-conflicting points"]
    assert len(merge_buttons) == 1


@pytest.mark.usefixtures("qtbot")
def test_review_dialog_overwrite_mode_uses_overwrite_button_text(qtbot):
    preview = _make_preview(
        policy=TrackingMergePolicy.OVERWRITE_EXISTING,
        n_overwriteable=1,
        overwrites=(_make_conflict_entry(),),
    )

    dlg = merger_ui.TrackingMergeReviewDialog(
        parent=None,
        preview=preview,
        review_only=False,
    )
    qtbot.addWidget(dlg)

    buttons = [b for b in dlg.findChildren(QPushButton) if b.text() == "Overwrite and merge"]
    assert len(buttons) == 1


# -----------------------------------------------------------------------------#
# Main dialog tests
# -----------------------------------------------------------------------------#


@pytest.mark.usefixtures("qtbot")
def test_tracking_merge_dialog_defaults_to_fill_missing_and_initial_preview(
    qtbot,
    monkeypatch,
    fake_points_layer_factory,
):
    source = fake_points_layer_factory(
        name="source",
        data=[[0, 10, 20]],
        labels=["nose"],
        ids=[""],
    )
    target = fake_points_layer_factory(
        name="target",
        data=[[0, 10, 20]],
        labels=["nose"],
        ids=[""],
    )

    preview = _make_preview(
        policy=TrackingMergePolicy.FILL_MISSING,
        n_appendable=1,
    )

    monkeypatch.setattr(merger_ui, "preview_tracking_merge", lambda *args, **kwargs: preview)

    dlg = merger_ui.TrackingMergeDialog(
        parent=None,
        layer_manager=SimpleNamespace(),
        source_candidates=(source,),
        target_candidates=(target,),
        initial_source=source,
        initial_target=target,
        fixed_source_layer=None,
    )
    qtbot.addWidget(dlg)

    assert dlg.selected_policy() is TrackingMergePolicy.FILL_MISSING
    assert dlg._merge_btn.isEnabled() is True
    assert dlg._review_btn.isEnabled() is False
    assert "will be added" in dlg._summary_label.text()


@pytest.mark.usefixtures("qtbot")
def test_tracking_merge_dialog_enables_review_for_conflicts_in_fill_missing_mode(
    qtbot,
    monkeypatch,
    fake_points_layer_factory,
):
    source = fake_points_layer_factory(name="source", data=[[0, 10, 20]], labels=["nose"], ids=[""])
    target = fake_points_layer_factory(name="target", data=[[0, 10, 20]], labels=["nose"], ids=[""])

    preview = _make_preview(
        policy=TrackingMergePolicy.FILL_MISSING,
        n_appendable=1,
        n_conflicts=2,
        conflicts=(_make_conflict_entry(),),
    )

    monkeypatch.setattr(merger_ui, "preview_tracking_merge", lambda *args, **kwargs: preview)

    dlg = merger_ui.TrackingMergeDialog(
        parent=None,
        layer_manager=SimpleNamespace(),
        source_candidates=(source,),
        target_candidates=(target,),
        initial_source=source,
        initial_target=target,
    )
    qtbot.addWidget(dlg)

    assert dlg._review_btn.isEnabled() is True
    assert dlg._merge_btn.isEnabled() is True
    assert "skipped" in dlg._summary_label.text()


@pytest.mark.usefixtures("qtbot")
def test_tracking_merge_dialog_overwrite_policy_changes_summary_and_review_enablement(
    qtbot,
    monkeypatch,
    fake_points_layer_factory,
):
    source = fake_points_layer_factory(name="source", data=[[0, 10, 20]], labels=["nose"], ids=[""])
    target = fake_points_layer_factory(name="target", data=[[0, 10, 20]], labels=["nose"], ids=[""])

    def _fake_preview(_source, _target, *, policy, **kwargs):
        if policy is TrackingMergePolicy.OVERWRITE_EXISTING:
            return _make_preview(
                policy=policy,
                n_appendable=0,
                n_overwriteable=2,
                overwrites=(_make_conflict_entry(),),
            )
        return _make_preview(
            policy=policy,
            n_appendable=1,
            n_conflicts=0,
        )

    monkeypatch.setattr(merger_ui, "preview_tracking_merge", _fake_preview)

    dlg = merger_ui.TrackingMergeDialog(
        parent=None,
        layer_manager=SimpleNamespace(),
        source_candidates=(source,),
        target_candidates=(target,),
        initial_source=source,
        initial_target=target,
    )
    qtbot.addWidget(dlg)

    # switch to overwrite mode
    dlg._policy_combo.setCurrentIndex(1)

    assert dlg.selected_policy() is TrackingMergePolicy.OVERWRITE_EXISTING
    assert dlg._review_btn.isEnabled() is True
    assert dlg._merge_btn.isEnabled() is True
    assert "overwritten by tracked coordinates" in dlg._summary_label.text()


@pytest.mark.usefixtures("qtbot")
def test_tracking_merge_dialog_disables_merge_when_preview_invalid(
    qtbot,
    monkeypatch,
    fake_points_layer_factory,
):
    source = fake_points_layer_factory(name="source", data=[[0, 10, 20]], labels=["nose"], ids=[""])
    target = fake_points_layer_factory(name="target", data=[[0, 10, 20]], labels=["nose"], ids=[""])

    preview = _make_preview(
        is_valid=False,
        invalid_reason="broken preview",
    )

    monkeypatch.setattr(merger_ui, "preview_tracking_merge", lambda *args, **kwargs: preview)

    dlg = merger_ui.TrackingMergeDialog(
        parent=None,
        layer_manager=SimpleNamespace(),
        source_candidates=(source,),
        target_candidates=(target,),
        initial_source=source,
        initial_target=target,
    )
    qtbot.addWidget(dlg)

    assert dlg._merge_btn.isEnabled() is False
    assert dlg._review_btn.isEnabled() is False
    assert "Merge unavailable" in dlg._summary_label.text()


# -----------------------------------------------------------------------------#
# Workflow tests
# -----------------------------------------------------------------------------#


def test_workflow_returns_false_and_warns_when_no_source_candidates(monkeypatch):
    warnings = []

    monkeypatch.setattr(
        merger_ui.QMessageBox,
        "warning",
        lambda *args, **kwargs: warnings.append((args, kwargs)),
    )

    workflow = merger_ui.TrackingMergeWorkflow(
        parent=None,
        viewer=SimpleNamespace(layers=SimpleNamespace(selection=SimpleNamespace(active=None)), status=""),
        layer_manager=SimpleNamespace(
            iter_tracking_result_layers=lambda: (),
        ),
        logger_=None,
    )

    ok = workflow.run()

    assert ok is False
    assert len(warnings) == 1
    assert "No tracking result layer" in warnings[0][0]


def test_workflow_returns_false_and_warns_when_no_target_candidates(monkeypatch, fake_points_layer_factory):
    source = fake_points_layer_factory(name="source", data=[[0, 10, 20]], labels=["nose"], ids=[""])

    warnings = []

    monkeypatch.setattr(
        merger_ui.QMessageBox,
        "warning",
        lambda *args, **kwargs: warnings.append((args, kwargs)),
    )

    layer_manager = SimpleNamespace(
        iter_tracking_result_layers=lambda: (source,),
        is_tracking_result_layer=lambda layer: layer is source,
        iter_mergeable_dlc_points_layers=lambda prefer_managed=True: (),
    )

    viewer = SimpleNamespace(
        layers=SimpleNamespace(selection=SimpleNamespace(active=source)),
        status="",
    )

    workflow = merger_ui.TrackingMergeWorkflow(
        parent=None,
        viewer=viewer,
        layer_manager=layer_manager,
        logger_=None,
    )

    ok = workflow.run(source_layer=source)

    assert ok is False
    assert len(warnings) == 1
    assert "No DLC target layer" in warnings[0][0] or "No Points layer is currently available" in warnings[0][0][2]


def test_workflow_success_path_applies_merge_and_updates_target(monkeypatch, fake_points_layer_factory):
    source = fake_points_layer_factory(name="source", data=[[0, 10, 20]], labels=["nose"], ids=[""])
    target = fake_points_layer_factory(name="target", data=[[1, 30, 40]], labels=["tail"], ids=[""])

    preview = _make_preview(
        policy=TrackingMergePolicy.FILL_MISSING,
        n_appendable=1,
        source_layer_name="source",
        target_layer_name="target",
    )

    class _FakeDialog:
        def __init__(self, *args, **kwargs):
            self.latest_preview = preview
            self._source = source
            self._target = target

        def exec_(self):
            return merger_ui.QDialog.Accepted

        def selected_source_layer(self):
            return self._source

        def selected_target_layer(self):
            return self._target

    applied = {}

    monkeypatch.setattr(merger_ui, "TrackingMergeDialog", _FakeDialog)
    monkeypatch.setattr(
        merger_ui,
        "apply_tracking_merge",
        lambda source_layer, target_layer, preview: (
            np.array([[0.0, 10.0, 20.0], [1.0, 30.0, 40.0]], dtype=float),
            pd.DataFrame({"label": ["nose", "tail"], "id": ["", ""]}),
        ),
    )
    monkeypatch.setattr(merger_ui, "mark_layer_presentation_changed", lambda layer: applied.setdefault("marked", layer))

    layer_manager = SimpleNamespace(
        iter_tracking_result_layers=lambda: (source,),
        is_tracking_result_layer=lambda layer: layer is source,
        iter_mergeable_dlc_points_layers=lambda prefer_managed=True: (target,),
        suggest_merge_target=lambda initial_source: target,
    )

    viewer = SimpleNamespace(
        layers=SimpleNamespace(selection=SimpleNamespace(active=source)),
        status="",
    )

    workflow = merger_ui.TrackingMergeWorkflow(
        parent=None,
        viewer=viewer,
        layer_manager=layer_manager,
        logger_=None,
    )

    ok = workflow.run(source_layer=source)

    assert ok is True
    assert viewer.layers.selection.active is target
    assert applied["marked"] is target
    assert 'Merged 1 tracked point into "target"' in viewer.status or "Merged 1 tracked point" in viewer.status
    assert np.asarray(target.data).shape == (2, 3)
    assert list(pd.DataFrame(target.features).columns) == ["label", "id"]


def test_workflow_requires_confirmation_for_overwrite_policy(monkeypatch, fake_points_layer_factory):
    source = fake_points_layer_factory(name="source", data=[[0, 10, 20]], labels=["nose"], ids=[""])
    target = fake_points_layer_factory(name="target", data=[[1, 30, 40]], labels=["tail"], ids=[""])

    preview = _make_preview(
        policy=TrackingMergePolicy.OVERWRITE_EXISTING,
        n_appendable=0,
        n_overwriteable=2,
        source_layer_name="source",
        target_layer_name="target",
    )

    class _FakeDialog:
        def __init__(self, *args, **kwargs):
            self.latest_preview = preview
            self._source = source
            self._target = target

        def exec_(self):
            return merger_ui.QDialog.Accepted

        def selected_source_layer(self):
            return self._source

        def selected_target_layer(self):
            return self._target

    monkeypatch.setattr(merger_ui, "TrackingMergeDialog", _FakeDialog)
    monkeypatch.setattr(merger_ui.TrackingMergeReviewDialog, "confirm", lambda *args, **kwargs: False)

    layer_manager = SimpleNamespace(
        iter_tracking_result_layers=lambda: (source,),
        is_tracking_result_layer=lambda layer: layer is source,
        iter_mergeable_dlc_points_layers=lambda prefer_managed=True: (target,),
        suggest_merge_target=lambda initial_source: target,
    )

    viewer = SimpleNamespace(
        layers=SimpleNamespace(selection=SimpleNamespace(active=source)),
        status="",
    )

    workflow = merger_ui.TrackingMergeWorkflow(
        parent=None,
        viewer=viewer,
        layer_manager=layer_manager,
        logger_=None,
    )

    ok = workflow.run(source_layer=source)

    assert ok is False
