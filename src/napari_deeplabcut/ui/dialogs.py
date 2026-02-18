# src/napari_deeplabcut/ui/dialogs.py
import pandas as pd
from qtpy.QtWidgets import QDialog, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QSizePolicy, QVBoxLayout

from napari_deeplabcut import misc


def _conflict_stats(key_conflict: pd.DataFrame) -> tuple[int, int]:
    """
    Returns:
      n_pairs  = number of (image, keypoint) pairs that will be overwritten
      n_images = number of distinct images affected
    """
    n_pairs = int(key_conflict.to_numpy().sum())
    n_images = int(key_conflict.any(axis=1).to_numpy().sum())
    return n_pairs, n_images


def _build_overwrite_warning_text(key_conflict: pd.DataFrame, max_items: int = 15) -> tuple[str, str]:
    """
    Returns (summary, details).
    Summary: short sentence for dialog top.
    Details: scrollable text body listing image → keypoint examples.
    """
    n_pairs, n_images = _conflict_stats(key_conflict)
    summary = f"{n_pairs} existing keypoint(s) will be overwritten across {n_images} image(s)."
    details = misc.summarize_keypoint_conflicts(key_conflict, max_items=max_items)
    return summary, details


def _maybe_confirm_overwrite(metadata: dict, key_conflict: pd.DataFrame) -> bool:
    """
    Returns True if save should proceed, False if user cancels.
    If no GUI controls are present, returns True (non-interactive).
    """
    if not key_conflict.to_numpy().any():
        return True

    controls = metadata.get("controls")
    if controls is None:
        return True  # headless/scripted save: no dialog

    summary, details = _build_overwrite_warning_text(key_conflict)
    return OverwriteConflictsDialog.confirm(controls, summary=summary, details=details)


class OverwriteConflictsDialog(QDialog):
    """
    Scrollable warning dialog listing keypoints that will be overwritten.
    Returns True if user chooses to proceed, False otherwise.
    """

    def __init__(self, parent, *, title: str, summary: str, details: str):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumSize(700, 450)

        layout = QVBoxLayout(self)

        summary_label = QLabel(summary)
        summary_label.setWordWrap(True)
        layout.addWidget(summary_label)

        text = QPlainTextEdit()
        text.setReadOnly(True)
        text.setPlainText(details)
        text.setLineWrapMode(QPlainTextEdit.NoWrap)
        text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(text)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)

        self.overwrite_btn = QPushButton("Overwrite")
        self.overwrite_btn.setDefault(True)
        self.overwrite_btn.setAutoDefault(True)
        self.overwrite_btn.clicked.connect(self.accept)

        btn_row.addWidget(self.cancel_btn)
        btn_row.addWidget(self.overwrite_btn)
        layout.addLayout(btn_row)

    @staticmethod
    def confirm(parent, *, summary: str, details: str, title="Overwrite warning") -> bool:
        dlg = OverwriteConflictsDialog(parent, title=title, summary=summary, details=details)
        return dlg.exec_() == QDialog.Accepted
