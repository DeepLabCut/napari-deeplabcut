# src/napari_deeplabcut/core/conflicts.py
from __future__ import annotations

from pathlib import Path

import pandas as pd

from napari_deeplabcut.config.models import AnnotationKind, OverwriteConflictReport, PointsMetadata
from napari_deeplabcut.core import schemas as dlc_schemas
from napari_deeplabcut.core.dataframes import set_df_scorer
from napari_deeplabcut.core.errors import AmbiguousSaveError, MissingProvenanceError
from napari_deeplabcut.core.metadata import parse_points_metadata
from napari_deeplabcut.core.project_paths import infer_dlc_project_from_points_meta
from napari_deeplabcut.core.provenance import (
    resolve_output_path_from_metadata,
)


def compute_overwrite_report_for_points_save(
    data,
    attributes: dict,
) -> OverwriteConflictReport | None:
    """
    Compute an overwrite-conflict report for a prospective points-layer save.

    This is a non-interactive preflight helper intended for UI/controller code
    to call *before* invoking the napari writer. It mirrors the writer's save
    routing logic closely enough to predict whether saving this points layer
    would merge into an existing GT file and overwrite existing keypoints.

    Parameters
    ----------
    data:
        Napari Points layer data, expected to be array-like of shape (N, 3)
        in [frame, y, x] order.
    attributes:
        Napari layer attributes dict for the points layer. This is the same
        payload shape passed to the npe2 writer.

    Returns
    -------
    OverwriteConflictReport | None
        - OverwriteConflictReport if the save target is an existing GT file and
          at least one keypoint overwrite conflict would occur.
        - None if:
            * there is no existing GT file to merge into,
            * the destination is not GT,
            * or no overwrite conflicts are detected.

    Raises
    ------
    ValueError
        If the layer attributes / points payload are invalid for save.
    MissingProvenanceError
        If saving a MACHINE source without a resolvable promotion target.
    AmbiguousSaveError
        If GT fallback resolution finds multiple CollectedData_*.h5 files.
    """
    # Local imports keep core.conflicts free of import cycles:
    # - core.dataframes imports ConflictEntry / OverwriteConflictReport
    # - core.io imports dataframe helpers and metadata parsing
    from napari_deeplabcut.core.dataframes import (
        build_overwrite_conflict_report,
        form_df_from_validated,
        keypoint_conflicts,
    )

    attrs = dlc_schemas.PointsLayerAttributesModel.model_validate(attributes or {})
    pts_meta: PointsMetadata = parse_points_metadata(attrs.metadata, drop_header=False)

    if not pts_meta.header:
        raise ValueError("Layer metadata must include a valid DLC header to write keypoints.")

    points = dlc_schemas.PointsDataModel.model_validate({"data": data})
    props = dlc_schemas.KeypointPropertiesModel.model_validate(attrs.properties)

    # Bundle + validate cross-field invariants exactly like the writer
    ctx = dlc_schemas.PointsWriteInputModel.model_validate(
        {
            "points": points,
            "meta": pts_meta,
            "props": props,
        }
    )

    # Build the outgoing dataframe exactly like the writer
    df_new = form_df_from_validated(ctx)

    # Resolve output path using the same provenance-first routing as write_hdf(...)
    out_path, target_scorer, source_kind = resolve_output_path_from_metadata(attributes)

    # Promotion to GT may rewrite scorer level
    if target_scorer:
        df_new = set_df_scorer(df_new, target_scorer)

    # Never write back to machine sources without an explicit promotion target
    if not out_path and source_kind == AnnotationKind.MACHINE:
        raise MissingProvenanceError("Cannot resolve provenance output path for MACHINE source.")

    # Same GT fallback logic as write_hdf(...)
    if not out_path:
        project_ctx = infer_dlc_project_from_points_meta(pts_meta, prefer_project_root=False)
        dataset_dir = project_ctx.dataset_folder

        if dataset_dir is not None:
            dataset_dir.mkdir(parents=True, exist_ok=True)
            root_path = dataset_dir
        else:
            root = pts_meta.root
            if not root:
                raise MissingProvenanceError("GT fallback requires root (and dataset folder could not be inferred).")
            root_path = Path(root)

        candidates = sorted(root_path.glob("CollectedData_*.h5"))
        if len(candidates) > 1:
            raise AmbiguousSaveError(
                f"Multiple CollectedData_*.h5 files found in {root_path}."
                " Cannot determine where to save."
                " Please specify a save_target with explicit path and scorer.",
                candidates=[str(c) for c in candidates],
            )
        elif len(candidates) == 1:
            out = candidates[0]
        else:
            scorer = target_scorer or pts_meta.header.scorer
            out = root_path / f"CollectedData_{scorer}.h5"
    else:
        out = Path(out_path)

    # Only GT merge-on-save can produce overwrite conflicts
    has_save_target = pts_meta.save_target is not None
    destination_kind = (
        AnnotationKind.GT
        if has_save_target
        else ((pts_meta.io.kind if pts_meta.io is not None else None) or AnnotationKind.GT)
    )

    if destination_kind != AnnotationKind.GT:
        return None

    # No existing file -> no merge -> no overwrite conflict
    if not out.exists():
        return None

    try:
        df_old = pd.read_hdf(out, key="keypoints")
    except (KeyError, ValueError):
        df_old = pd.read_hdf(out)

    key_conflict = keypoint_conflicts(df_old, df_new)

    report = build_overwrite_conflict_report(
        key_conflict,
        layer_name=attributes.get("name"),
        destination_path=str(out),
    )

    return report if report.has_conflicts else None
