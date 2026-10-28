from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from eyehand.app import App


def test_video_panels_match_tripod_on_users_left(monkeypatch):
    from eyehand import app as module
    from eyehand.config import load_camera_configs

    widgets = Mock()
    monkeypatch.setattr(module, "ttk", widgets)
    monkeypatch.setattr(module, "tk", Mock())
    app = SimpleNamespace(configs=load_camera_configs(), _show_buttons=Mock())
    for method in ("start_cursor", "start_direct_cursor", "stop_cursor", "start_recalibration",
                   "start_emergency_recalibration", "mark_moved", "undo_view", "finish_capture",
                   "cancel", "back_to_place", "_set_tracking", "use_failed_calibration"):
        setattr(app, method, Mock())
    App._build_ui(app)
    assert list(app.video) == ["F", "L"]
    assert list(app.caption) == ["F", "L"]
    labels = [call.kwargs["text"] for call in widgets.LabelFrame.call_args_list]
    assert labels[0].startswith("Camera F:") and "user's left" in labels[0]
    assert labels[1].startswith("Camera L:")
    assert [call.kwargs["column"] for call in widgets.LabelFrame.return_value.grid.call_args_list] == [0, 1]


def test_monitor_warning_caps_sounds_but_keeps_refreshing(monkeypatch):
    play = Mock()
    monkeypatch.setattr("eyehand.app.sound.play", play)
    app = SimpleNamespace(monitor_beeps=0, refresh_status=Mock())
    for _ in range(10):
        App._monitor_warning(app)
    assert play.call_count == 3
    assert app.refresh_status.call_count == 10
    app.refresh_status.assert_called_with(reload_calibration=False)


def test_warning_refresh_preserves_monitor_state(monkeypatch):
    monkeypatch.setattr("eyehand.app.status.calibration_status", lambda _: [])
    monkeypatch.setattr("eyehand.app.status.next_action", lambda _: None)
    health = SimpleNamespace(configure=Mock(), alarm_logged=True)
    app = SimpleNamespace(configs={}, status_frame=Mock(), next_label=Mock(), health=health)
    app.status_frame.winfo_children.return_value = []
    App.refresh_status(app, reload_calibration=False)
    health.configure.assert_not_called()
    assert health.alarm_logged



def test_cursor_uses_existing_preview_workers(monkeypatch):
    from eyehand import browser_cursor

    worker, server = Mock(), Mock()
    make_server = Mock(return_value=server)
    thread = Mock()
    monkeypatch.setattr("eyehand.app.ThreadingHTTPServer", make_server)
    monkeypatch.setattr("eyehand.app.threading.Thread", Mock(return_value=thread))
    monkeypatch.setattr("eyehand.app.browser_cursor.load_geometry", Mock(return_value=(1, 2, 3, 4)))
    open_browser = Mock()
    monkeypatch.setattr("eyehand.app.webbrowser.open", open_browser)
    app = SimpleNamespace(mode="place", configs={}, workers={"L": worker}, cursor_server=None,
                          cursor_samples=browser_cursor.Samples(), cursor_thread=None, header=Mock(), track_var=Mock(),
                          _show_buttons=Mock(), _set_tracking=Mock())
    App.start_cursor(app)
    assert app.mode == "place"
    worker.stop.assert_not_called()
    worker.join.assert_not_called()
    worker.start.assert_not_called()
    assert worker.want_profile == "measurement"
    thread.start.assert_called_once()
    open_browser.assert_called_once_with("http://127.0.0.1:8765")
    App.stop_cursor(app)
    server.shutdown.assert_called_once()
    server.server_close.assert_called_once()
    assert app.cursor_server is None
    assert worker.want_profile == "placement"
    worker.stop.assert_not_called()


def test_gui_cursor_publishes_both_hands_and_rejects_stale_frames(monkeypatch):
    import time
    from eyehand import browser_cursor

    samples = browser_cursor.Samples()
    samples.set_settings({"eye": "L", "primary_hand": "L", "thumb_weight": 0.25})
    geometry = ({"L": "intr L", "F": "intr F"}, "pose", "screen", "size")
    worker = SimpleNamespace(profile="measurement", error=None)
    app = SimpleNamespace(cursor_server=Mock(), mode="place", c5_usable=True, track_var=Mock(),
                          workers={"L": worker, "F": worker}, cursor_samples=samples,
                          cursor_geometry=geometry)
    t = time.monotonic() * 1000
    a, b = SimpleNamespace(t_ms=t-10), SimpleNamespace(t_ms=t)
    snaps = {"L": SimpleNamespace(landmarks=a), "F": SimpleNamespace(landmarks=b)}
    solve = Mock(return_value={"L": {"valid": True}, "R": {"valid": False}})
    monkeypatch.setattr("eyehand.app.browser_cursor._pointer_samples", solve)
    App._publish_cursor(app, snaps)
    solve.assert_called_once_with(a, b, "L", *geometry, thumb_weight=0.25,
                                  stabilizer=samples.fingertips)
    assert samples.after(0)[1]["primary_hand"] == "L"
    a.t_ms -= 1000
    b.t_ms -= 1000
    App._publish_cursor(app, snaps)
    assert samples.after(1)[1]["reason"] == "Waiting for fresh camera frames"
    assert solve.call_count == 1


def test_cursor_start_failure_does_not_stop_preview(monkeypatch):
    monkeypatch.setattr("eyehand.app.browser_cursor.load_geometry", Mock(side_effect=RuntimeError("bad calibration")))
    worker = Mock()
    app = SimpleNamespace(mode="place", configs={}, cursor_server=None, header=Mock(), workers={"L": worker})
    App.start_cursor(app)
    worker.stop.assert_not_called()
    assert app.cursor_server is None
    assert "bad calibration" in app.header.config.call_args.kwargs["text"]


def test_emergency_starts_three_pose_capture():
    app = SimpleNamespace(mode="place", can_recal=True, start_capture=Mock(), stop_cursor=Mock())
    App.start_emergency_recalibration(app)
    assert app.target_views == 3
    app.start_capture.assert_called_once()
    app.mode = "capture"
    App.start_emergency_recalibration(app)
    assert app.start_capture.call_count == 1


def test_normal_capture_keeps_coverage_and_depth_requirements():
    cells = np.zeros((2, 3, 2), int)
    app = SimpleNamespace(target_views=20, session=SimpleNamespace(views=[None] * 20), cells=cells)
    cells[:, :, 0] = 1
    assert not App.capture_complete(app)
    cells[0, 0, 1] = 1
    assert App.capture_complete(app)
    app.session.views.pop()
    assert not App.capture_complete(app)
    app.session.views.append(None)
    cells[0, :, :] = 0
    assert not App.capture_complete(app)


def test_failed_emergency_solve_does_not_install(monkeypatch):
    install = Mock()
    monkeypatch.setattr("eyehand.app.stereo.install", install)
    monkeypatch.setattr("eyehand.app.stereo.format_report", Mock(return_value="failed"))
    monkeypatch.setattr("eyehand.app.sound.play", Mock())
    app = SimpleNamespace(solve_result={"passed": False, "gates": []},
                          header=Mock(), _show_buttons=Mock())
    App.show_result(app)
    install.assert_not_called()
    assert app.mode == "result"
    assert app.failed_result == {"passed": False, "gates": []}


def test_failed_result_can_be_explicitly_applied(monkeypatch):
    install = Mock()
    monkeypatch.setattr("eyehand.app.stereo.install", install)
    monkeypatch.setattr("eyehand.app.status.log_event", Mock())
    result = {"passed": False, "gates": [{"name": "accuracy", "pass": False}], "raw_dir": "test"}
    app = SimpleNamespace(mode="result", failed_result=result, _save_room_refs=Mock(),
                          back_to_place=Mock(), header=Mock())
    App.use_failed_calibration(app)
    install.assert_called_once_with({**result, "accepted_failed_gates": True})
    assert not result["passed"] and "accepted_failed_gates" not in result
    assert app.failed_result is None
    app.back_to_place.assert_called_once()
    App.use_failed_calibration(app)
    assert install.call_count == 1


def test_solve_error_offers_no_failed_result(monkeypatch):
    install = Mock()
    monkeypatch.setattr("eyehand.app.stereo.install", install)
    app = SimpleNamespace(solve_result={"error": "no solution"}, failed_result={"old": True},
                          header=Mock(), _show_buttons=Mock())
    App.show_result(app)
    assert app.failed_result is None
    install.assert_not_called()


def test_emergency_capture_finishes_at_three_without_coverage_wait():
    app = SimpleNamespace(target_views=3, session=SimpleNamespace(views=[None] * 2))
    assert not App.capture_complete(app)
    app.session.views.append(None)
    assert App.capture_complete(app)


def test_emergency_never_queues_a_fourth_capture():
    import threading
    worker = SimpleNamespace(profile="measurement")
    selector = SimpleNamespace(accepted=[None] * 3, offer=Mock())
    app = SimpleNamespace(mode="capture", workers={"L": worker, "F": worker},
                          target_views=3, selector=selector, lock=threading.Lock())
    App._on_detection(app, "L", None, None, None)
    selector.offer.assert_not_called()


def test_emergency_result_applies_without_face_baseline_wait(monkeypatch):
    install = Mock()
    monkeypatch.setattr("eyehand.app.stereo.install", install)
    monkeypatch.setattr("eyehand.app.status.log_event", Mock())
    result = {"passed": True, "emergency": True, "raw_dir": "test"}
    app = SimpleNamespace(solve_result=result, _save_room_refs=Mock(), back_to_place=Mock(),
                          header=Mock())
    App.show_result(app)
    install.assert_called_once_with(result)
    app.back_to_place.assert_called_once()


def test_cursor_only_releases_preview_and_reuses_controls(monkeypatch):
    import threading
    from eyehand import browser_cursor

    worker = Mock()
    worker.is_alive.return_value = False
    runtime = Mock()
    monkeypatch.setattr("eyehand.app.browser_cursor.run", runtime)
    samples = browser_cursor.Samples()
    samples.set_settings({"eye": "L", "primary_hand": "L"})
    app = SimpleNamespace(mode="place", workers={"L": worker}, cursor_stop=threading.Event(),
                          cursor_samples=samples, cursor_error=None, stop_cursor=Mock(),
                          _show_buttons=Mock(), header=Mock())
    App.start_direct_cursor(app)
    app.cursor_thread.join(timeout=1)
    app.stop_cursor.assert_called_once()
    worker.stop.assert_called_once()
    worker.join.assert_called_once()
    assert app.mode == "cursor"
    runtime.assert_called_once_with(stop_event=app.cursor_stop, open_browser=True, samples=samples)
    assert samples.settings() == {"eye": "L", "primary_hand": "L", "thumb_weight": 0.75, "release_sensitivity": 0.5}


def test_cursor_only_tick_skips_preview_and_health_checks():
    thread = Mock()
    thread.is_alive.return_value = True
    App._tick(SimpleNamespace(mode="cursor", cursor_thread=thread))


def test_stop_cursor_only_waits_before_restoring_preview(monkeypatch):
    import threading

    thread = Mock()
    thread.is_alive.return_value = True
    worker = Mock()
    make_worker = Mock(return_value=worker)
    monkeypatch.setattr("eyehand.app.CameraWorker", make_worker)
    app = SimpleNamespace(cursor_thread=thread, cursor_stop=threading.Event(), header=Mock(),
                          configs={"L": "left", "F": "front"}, _on_detection=Mock(), back_to_place=Mock())
    App.stop_cursor(app)
    assert app.cursor_stop.is_set()
    make_worker.assert_not_called()
    thread.is_alive.return_value = False
    App.stop_cursor(app)
    assert app.cursor_thread is None
    assert worker.start.call_count == 2
    app.back_to_place.assert_called_once()


def test_cursor_only_failure_returns_to_preview():
    thread = Mock()
    thread.is_alive.return_value = False
    app = SimpleNamespace(mode="cursor", cursor_thread=thread, cursor_error="port busy", cursor_resume_videos=False,
                          stop_cursor=Mock(), header=Mock())
    App._tick(app)
    app.stop_cursor.assert_called_once()
    assert "port busy" in app.header.config.call_args.kwargs["text"]


def test_close_cursor_only_does_not_restart_preview():
    import threading

    thread, worker = Mock(), Mock()
    app = SimpleNamespace(cursor_stop=threading.Event(), cursor_thread=thread, cursor_server=None,
                          workers={"L": worker}, destroy=Mock())
    App.close(app)
    assert app.cursor_stop.is_set()
    thread.join.assert_called_once()
    worker.stop.assert_called_once()
    worker.start.assert_not_called()
    app.destroy.assert_called_once()


def test_switch_to_videos_waits_for_direct_camera_release():
    import threading

    thread = Mock()
    thread.is_alive.return_value = True
    app = SimpleNamespace(mode="cursor", cursor_thread=thread, cursor_stop=threading.Event(),
                          cursor_resume_videos=False, cursor_error=None, header=Mock(),
                          _show_buttons=Mock(), stop_cursor=Mock(), start_cursor=Mock())
    App.start_cursor(app)
    assert app.cursor_stop.is_set()
    assert app.cursor_resume_videos
    App._tick(app)
    app.stop_cursor.assert_not_called()
    app.start_cursor.assert_not_called()
    thread.is_alive.return_value = False
    App._tick(app)
    app.stop_cursor.assert_called_once()
    app.start_cursor.assert_called_once()


def test_both_mode_buttons_remain_visible_with_active_mode_disabled():
    buttons = {name: Mock() for name in ("btn_recal", "btn_moved", "btn_undo", "btn_finish",
               "btn_cancel", "btn_retry", "btn_done", "chk_track", "btn_cursor",
               "btn_direct_cursor", "btn_stop_cursor", "btn_emergency", "btn_use_failed")}
    app = SimpleNamespace(**buttons, mode="place", cursor_server=Mock(),
                          cursor_resume_videos=False, progress=Mock(), failed_result=None)
    App._show_buttons(app)
    app.btn_cursor.pack.assert_called_once()
    app.btn_direct_cursor.pack.assert_called_once()
    app.btn_cursor.state.assert_called_with(["disabled"])
    app.btn_direct_cursor.state.assert_called_with(["!disabled"])
    app.mode, app.cursor_server = "cursor", None
    App._show_buttons(app)
    app.btn_cursor.state.assert_called_with(["!disabled"])
    app.btn_direct_cursor.state.assert_called_with(["disabled"])
    app.mode = "result"
    App._show_buttons(app)
    app.btn_use_failed.pack.assert_not_called()
    app.failed_result = {"passed": False}
    App._show_buttons(app)
    app.btn_use_failed.pack.assert_called_once()
