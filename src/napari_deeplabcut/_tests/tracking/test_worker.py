from napari_deeplabcut.tracking._data import TrackingWorkerOutput
from napari_deeplabcut.tracking._worker import TrackingWorker


def test_tracking_worker(qtbot, track_worker_inputs):
    worker = TrackingWorker()

    with qtbot.waitSignal(worker.trackingFinished, timeout=1000) as await_finished:
        worker.track(track_worker_inputs)

    output: TrackingWorkerOutput = await_finished.args[0]
    assert isinstance(output, TrackingWorkerOutput)
    # Expect (T * K, 3) with columns [frame_idx, x, y]
    T = track_worker_inputs.video.shape[0]
    K = track_worker_inputs.keypoints.shape[0]
    assert output.keypoints.shape == (T * K, 3)
    # Check frame indices
    frames = output.keypoints[:, 0]
    assert frames.min() == 0
    assert frames.max() == T - 1


def test_progress_emitted(track_worker_inputs):
    worker = TrackingWorker()

    progress_events = []
    worker.progress.connect(lambda tpl: progress_events.append(tuple(tpl)))
    worker.track(track_worker_inputs)

    T = track_worker_inputs.video.shape[0]
    assert len(progress_events) >= T  # at least one per frame
    assert progress_events[-1] == (T - 1, T)  # final progress emit is done elsewhere


def test_stop_tracking_emits_stopped(qtbot, track_worker_inputs):
    worker = TrackingWorker()
    await_stopped = qtbot.waitSignal(worker.trackingStopped, timeout=1000)

    def stop_on_first_progress(tpl):
        worker.stop_tracking()

    worker.progress.connect(stop_on_first_progress)
    worker.track(track_worker_inputs)
    assert await_stopped.signal_triggered


def test_unknown_tracker_emits_stopped(qtbot, track_worker_inputs):
    worker = TrackingWorker()
    inval_cfg = track_worker_inputs
    inval_cfg.tracker_name = "DoesNotExist"
    await_stopped = qtbot.waitSignal(worker.trackingStopped, timeout=1000)
    worker.track(inval_cfg)
    assert await_stopped.signal_triggered
