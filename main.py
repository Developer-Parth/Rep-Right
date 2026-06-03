
from __future__ import annotations

import sys
import time

import cv2
import numpy as np

from config import (
    AppState,
    CAMERA_INDEX,
    CONSECUTIVE_FRAMES,
    EXERCISES,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    GOOD_REP_VOICE_COOLDOWN,
    INFERENCE_EVERY,
    VOICE_COOLDOWN_SEC,
)
from menu import (
    MenuState,
    _C,
    draw_button,
    hit_test,
    make_button,
    render_menu,
    render_resign_dialog,
    render_summary,
)
from pose_detector import PoseDetector
from rep_counter import RepCounter, SquatPhase
from session_logger import SessionLogger
from squat_analyzer import SquatAnalyzer
from utils import MovingAverage, clamp, draw_color_for_score
from voice_coach import VoiceCoach

_PHASE_COLOR = {
    SquatPhase.STANDING:    _C["green"],
    SquatPhase.DESCENDING:  _C["cyan"],
    SquatPhase.BOTTOM:      _C["magenta"],
    SquatPhase.ASCENDING:   _C["yellow"],
    SquatPhase.CALIBRATING: _C["gray"],
}

_QUALITY_LABELS = [
    ("ELITE", 95, _C["cyan"]),
    ("GOOD",  80, _C["green"]),
    ("AVERAGE", 60, _C["yellow"]),
    ("POOR",   0, _C["red"]),
]

_EXERCISE = "SQUAT"

_click_pos: list[tuple[int, int] | None] = [None]


def _mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        _click_pos[0] = (x, y)


def consume_click():
    pos = _click_pos[0]
    _click_pos[0] = None
    return pos


def _draw_panel(frame, x, y, pw, ph, alpha=0.55):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + pw, y + ph), _C["panel"], -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _put_text(frame, text, pos, color, scale=0.55, thickness=1):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                _C["black"], thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def _draw_glow_text(frame, text, pos, color, scale=0.55, thickness=1, glow_layers=3):
    for i in range(glow_layers):
        factor = 1.0 / (2 ** (glow_layers - i))
        gc = tuple(int(c * factor) for c in color)
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                     gc, thickness + (glow_layers - i) * 3, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                _C["black"], thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def _draw_corner_brackets(frame, x, y, w, h, color, length=12, thickness=2):
    cv2.line(frame, (x, y + length), (x, y), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x, y), (x + length, y), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x + w - length, y), (x + w, y), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x + w, y), (x + w, y + length), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x, y + h - length), (x, y + h), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x, y + h), (x + length, y + h), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x + w - length, y + h), (x + w, y + h), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x + w, y + h - length), (x + w, y + h), color, thickness, cv2.LINE_AA)


def _draw_ring_meter(frame, cx, cy, radius, score):
    cv2.ellipse(frame, (cx, cy), (radius, radius), 0, 135, 405,
                _C["gray"], 8, cv2.LINE_AA)
    end_angle = 135 + 270 * clamp(score, 0, 100) / 100
    cv2.ellipse(frame, (cx, cy), (radius, radius), 0, 135, end_angle,
                draw_color_for_score(score), 8, cv2.LINE_AA)
    text = f"{score:.0f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    _put_text(frame, text, (cx - tw // 2, cy + th // 2), _C["white"], scale=0.7, thickness=2)


def _draw_body_visibility(frame, pct, x, y):
    _put_text(frame, "BODY", (x, y), _C["gray"], scale=0.4, thickness=1)
    bar_x = x + 38
    bar_w, bar_h = 80, 8
    fill_w = int(bar_w * clamp(pct, 0, 100) / 100)
    cv2.rectangle(frame, (bar_x, y - 6), (bar_x + bar_w, y - 6 + bar_h), _C["gray"], 1)
    bar_color = _C["cyan"] if pct > 60 else _C["orange"] if pct > 30 else _C["red"]
    cv2.rectangle(frame, (bar_x, y - 6), (bar_x + fill_w, y - 6 + bar_h), bar_color, -1)
    _put_text(frame, f"{pct:.0f}%", (bar_x + bar_w + 6, y), _C["white"], scale=0.4, thickness=1)


def _draw_quality_badge(frame, score, x, y):
    for label, threshold, color in _QUALITY_LABELS:
        if score >= threshold:
            _put_text(frame, label, (x, y), color, scale=0.5, thickness=2)
            return


def _lerp_color(c1, c2, t):
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def draw_workout_overlay(
    frame,
    phase, reps, score, fps, issues, issue_counters,
    feedback_text, feedback_color, voice_on, debug, debug_metrics,
    recent_scores,
    body_visibility=0.0,
    pulse_color=None,
):
    h, w = frame.shape[:2]

    bar_h = 44
    _draw_panel(frame, 0, 0, w, bar_h)
    _put_text(frame, "REP RIGHT", (14, 18), _C["cyan"], scale=0.6, thickness=2)

    ring_cx = w - 100
    ring_cy = 58
    ring_r = 52
    _draw_ring_meter(frame, ring_cx, ring_cy, ring_r, score)
    _draw_quality_badge(frame, score, ring_cx - 24, ring_cy + ring_r + 10)

    phase_color = _PHASE_COLOR.get(phase, _C["white"])
    cv2.circle(frame, (16, bar_h + 18), 4, phase_color, -1, cv2.LINE_AA)
    _put_text(frame, f"{phase.value}", (28, bar_h + 22), phase_color, scale=0.5, thickness=1)
    _put_text(frame, f"EXERCISE  {_EXERCISE}", (14, bar_h + 46), _C["gray"], scale=0.4, thickness=1)
    _put_text(frame, f"REPS  {reps}", (200, bar_h + 22), _C["white"], scale=0.45, thickness=1)
    _put_text(frame, f"ESC  end", (w - 240, bar_h + 22), _C["gray"], scale=0.4, thickness=1)

    if body_visibility > 0:
        _draw_body_visibility(frame, body_visibility, 14, bar_h + 72)

    row_h = 26
    n_rows = max(1, len(issues))
    coach_h = 48
    issue_h = n_rows * row_h
    spark_h = 36 if recent_scores else 0
    panel_h = coach_h + issue_h + spark_h + 20
    panel_y = h - panel_h
    _draw_panel(frame, 0, panel_y, w, panel_h, alpha=0.6)
    _draw_corner_brackets(frame, 8, panel_y + 4, w - 16, panel_h - 8,
                          _C["cyan"], length=10, thickness=1)

    use_color = pulse_color if pulse_color else feedback_color
    _draw_glow_text(frame, feedback_text, (w // 2 - 140, panel_y + 32),
                    use_color, scale=0.9, thickness=2, glow_layers=3)

    bar_max_w = 160
    for i, key in enumerate(issues):
        row_y = panel_y + coach_h + 4 + i * row_h
        count = issue_counters.get(key, 0)
        pct = min(count / max(CONSECUTIVE_FRAMES, 1), 1.0)
        fill_w = int(bar_max_w * pct)
        bar_color = _C["red"] if pct >= 1.0 else _C["magenta"]
        cv2.rectangle(frame, (14, row_y + 4), (14 + fill_w, row_y + 18), bar_color, -1)
        cv2.rectangle(frame, (14, row_y + 4), (14 + bar_max_w, row_y + 18), _C["gray"], 1)
        _put_text(frame, key, (bar_max_w + 22, row_y + 18), bar_color, scale=0.45, thickness=1)

    if recent_scores:
        spark_w, spark_h2 = 200, 32
        sx = w - spark_w - 20
        sy = panel_y + panel_h - spark_h2 - 10
        pts = []
        max_n = 30
        vals = recent_scores[-max_n:]
        for i, v in enumerate(vals):
            px = sx + int(i * spark_w / max(len(vals) - 1, 1))
            py = sy + spark_h2 - int(clamp(v, 0, 100) / 100 * spark_h2)
            pts.append((px, py))
        if len(pts) > 1:
            cv2.polylines(frame, [np.array(pts, np.int32)], False,
                          _C["cyan"], 2, cv2.LINE_AA)
        _put_text(frame, "Recent", (sx, sy - 6), _C["gray"], scale=0.35, thickness=1)

    if debug and debug_metrics:
        lines = [
            f"Back  {debug_metrics.get('back_angle', 0):.1f}",
            f"KneeL {debug_metrics.get('knee_l', 0):.1f}",
            f"KneeR {debug_metrics.get('knee_r', 0):.1f}",
            f"HipY  {debug_metrics.get('hip_y', 0):.0f}",
        ]
        for j, line in enumerate(lines):
            _put_text(frame, line, (w - 200, bar_h + 16 + j * 18), _C["green"], scale=0.4, thickness=1)

    if not voice_on:
        _put_text(frame, "VOICE OFF", (14, h - 8), _C["gray"], scale=0.35, thickness=1)


def draw_banner(frame, text, color):
    h, w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
    bx = (w - tw) // 2
    by = h // 2 + th // 2
    pad = 18
    overlay = frame.copy()
    cv2.rectangle(overlay, (bx - pad, by - th - pad), (bx + tw + pad, by + pad),
                  _C["panel"], -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    _draw_corner_brackets(frame, bx - pad, by - th - pad, tw + pad * 2, th + pad * 2,
                          _C["cyan"], length=12, thickness=2)
    _draw_glow_text(frame, text, (bx, by), color, scale=1.2, thickness=3, glow_layers=3)


def main():
    window = "Rep Right"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, FRAME_WIDTH, FRAME_HEIGHT)
    cv2.setMouseCallback(window, _mouse_cb)

    state = AppState.MENU
    menu_state = MenuState()
    menu_buttons = [
        make_button("start", (FRAME_WIDTH - 240) // 2, 375, 240, 46),
    ]

    detector = PoseDetector(smoothing_window=5)
    counter  = RepCounter(smoothing_window=7)
    analyzer = SquatAnalyzer()
    coach    = VoiceCoach(enabled=True)
    logger   = SessionLogger()

    cap = None
    voice_on = coach.available
    debug = False
    paused = False
    issue_counters: dict[str, int] = {}
    last_kp = None
    last_analysis = None
    frame_count = 0
    fps_smoother = MovingAverage(window=12)
    prev_t = time.monotonic()
    prev_active_issues: set = set()
    pulse_active = False
    pulse_start = 0.0
    pulse_color_val = None
    last_frame = None

    resign_buttons = [
        make_button("resume", FRAME_WIDTH // 2 - 180, 410, 160, 40),
        make_button("end", FRAME_WIDTH // 2 + 20, 410, 160, 40),
    ]

    summary_buttons = [
        make_button("continue", (FRAME_WIDTH - 240) // 2, 520, 240, 46),
    ]
    summary_data = {}

    blank_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

    def _rep_complete(record):
        score, issues, depth_ok = analyzer.finalize_rep()
        logger.record_rep(record.rep_number, score, issues, depth_ok)
        for iss in issues:
            logger.record_correction(iss)

    counter.on_rep_complete(_rep_complete)

    def _open_cam():
        nonlocal cap
        import platform
        if platform.system() == "Darwin":
            cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)
        else:
            cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap or not cap.isOpened():
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return True

    def _close_cam():
        nonlocal cap
        if cap:
            cap.release()
            cap = None

    def _reset_workout_state():
        nonlocal frame_count, paused, issue_counters
        nonlocal last_kp, last_analysis, prev_active_issues
        nonlocal pulse_active, prev_t, last_frame
        counter.reset()
        analyzer.reset()
        issue_counters.clear()
        last_kp = last_analysis = None
        prev_active_issues.clear()
        pulse_active = False
        frame_count = 0
        paused = False
        fps_smoother.reset()
        prev_t = time.monotonic()
        last_frame = None

    print("=" * 60)
    print("  REP RIGHT")
    print("=" * 60)
    print("  Controls:")
    print("    q / ESC -> quit            p -> pause / resume")
    print("    r       -> reset session   v -> toggle voice")
    print("    d       -> toggle debug overlay")
    print()
    print("  TIP: Stand side-on ~1.5-2 m away,")
    print("       full body in frame. Calibrates for ~1 s.")
    print("=" * 60)

    try:
        while True:
            now = time.monotonic()

            if state == AppState.MENU:
                click = consume_click()
                if click:
                    clicked = hit_test(menu_buttons, click[0], click[1])
                    if clicked and clicked.action == "start":
                        _reset_workout_state()
                        logger = SessionLogger()
                        if not _open_cam():
                            print("ERROR: Cannot open camera.")
                            _close_cam()
                        else:
                            state = AppState.WORKOUT
                            print(f"Starting: {EXERCISES[menu_state.selected_exercise]}")

                render_menu(blank_frame, menu_state.selected_exercise, menu_buttons)
                cv2.imshow(window, blank_frame)

                k = cv2.waitKey(30) & 0xFF
                if k == ord('q') or k == 27:
                    break
                if k == 13 or k == 10:
                    _reset_workout_state()
                    logger = SessionLogger()
                    if not _open_cam():
                        print("ERROR: Cannot open camera.")
                        _close_cam()
                    else:
                        state = AppState.WORKOUT
                        print(f"Starting: {EXERCISES[menu_state.selected_exercise]}")

            elif state == AppState.WORKOUT:
                if cap is None or not cap.isOpened():
                    state = AppState.MENU
                    _close_cam()
                    continue

                ret, frame = cap.read()
                if not ret or frame is None:
                    if cv2.waitKey(10) & 0xFF in (ord('q'), 27):
                        break
                    continue

                frame = cv2.flip(frame, 1)

                dt = max(now - prev_t, 1e-4)
                fps = fps_smoother.update(1.0 / dt)
                prev_t = now

                if paused:
                    draw_banner(frame, "PAUSED - press P to resume", _C["yellow"])
                    cv2.imshow(window, frame)
                    k = cv2.waitKey(30) & 0xFF
                    if k == ord('q') or k == 27:
                        state = AppState.RESIGN_DIALOG
                    if k == ord('p'):
                        paused = False
                    continue

                frame_count += 1
                run_inference = (frame_count % INFERENCE_EVERY == 0)

                if run_inference:
                    kp = detector.process(frame)
                    last_kp = kp
                else:
                    detector.draw_cached(frame)
                    kp = last_kp

                phase = SquatPhase.CALIBRATING
                score = 100.0
                current_issues: list[str] = []
                feedback_text = "Get into view - stand side-on, full body"
                feedback_color = _C["white"]
                debug_metrics: dict = {}
                body_visibility = 0.0

                if kp is not None and kp.valid:
                    l_hip = kp.get("left_hip")
                    r_hip = kp.get("right_hip")
                    hip_y = float(np.mean([l_hip[1], r_hip[1]])) \
                        if (l_hip is not None and r_hip is not None) else 0.0

                    phase = counter.update(hip_y, FRAME_HEIGHT)

                    if run_inference:
                        analysis = analyzer.analyse(kp, phase)
                        last_analysis = analysis
                    else:
                        analysis = last_analysis

                    if analysis is not None and analysis.analysed:
                        score = analysis.score
                        current_issues = list(analysis.issues)
                        debug_metrics = {
                            "back_angle": analysis.back_angle,
                            "knee_l": analysis.knee_angle_left,
                            "knee_r": analysis.knee_angle_right,
                            "hip_y": hip_y,
                        }
                        counter.record_rep_score(score, current_issues, analysis.depth_ok)

                    vis_values = [v for v in kp.visibility.values() if v > 0]
                    body_visibility = float(np.mean(vis_values)) * 100 if vis_values else 0.0

                    active_set = set(current_issues)
                    for key in list(issue_counters):
                        if key not in active_set:
                            issue_counters[key] = 0
                    for key in current_issues:
                        issue_counters[key] = issue_counters.get(key, 0) + 1

                    if active_set and active_set != prev_active_issues:
                        pulse_active = True
                        pulse_start = now
                    prev_active_issues = active_set

                    if voice_on and current_issues:
                        sustained = [
                            k for k in current_issues
                            if issue_counters.get(k, 0) >= CONSECUTIVE_FRAMES
                        ]
                        if sustained:
                            priority = ["Go deeper", "Push knees out",
                                        "Knees too far forward", "Keep chest up"]
                            sustained.sort(key=lambda k: priority.index(k)
                                           if k in priority else 99)
                            coach.alert(sustained[0], cooldown=VOICE_COOLDOWN_SEC)

                    if current_issues:
                        feedback_text = current_issues[0]
                        feedback_color = _C["red"]
                    elif phase == SquatPhase.CALIBRATING:
                        feedback_text = "Calibrating - stand still"
                        feedback_color = _C["yellow"]
                    elif phase == SquatPhase.STANDING:
                        feedback_text = "Ready - squat when you like"
                        feedback_color = _C["green"]
                    else:
                        feedback_text = f"{phase.value} - form looks good"
                        feedback_color = _C["green"]

                else:
                    draw_banner(frame, "Step into frame", _C["white"])

                if pulse_active:
                    elapsed = (now - pulse_start) * 1000
                    if elapsed < 400:
                        t = elapsed / 400
                        if t < 0.5:
                            pulse_color_val = _lerp_color(_C["cyan"], _C["magenta"], t * 2)
                        else:
                            pulse_color_val = _lerp_color(_C["magenta"], _C["red"], (t - 0.5) * 2)
                    else:
                        pulse_active = False
                        pulse_color_val = None
                else:
                    pulse_color_val = None

                recent_scores = [r.score for r in counter.rep_history if r.score > 0]
                draw_workout_overlay(
                    frame=frame,
                    phase=phase,
                    reps=counter.rep_count,
                    score=score,
                    fps=fps,
                    issues=current_issues,
                    issue_counters=issue_counters,
                    feedback_text=feedback_text,
                    feedback_color=feedback_color,
                    voice_on=voice_on,
                    debug=debug,
                    debug_metrics=debug_metrics,
                    recent_scores=recent_scores,
                    body_visibility=body_visibility,
                    pulse_color=pulse_color_val,
                )

                cv2.imshow(window, frame)
                last_frame = frame.copy()

                k = cv2.waitKey(1) & 0xFF
                if k == 27:
                    state = AppState.RESIGN_DIALOG
                elif k == ord('q'):
                    break
                elif k == ord('p'):
                    paused = True
                elif k == ord('r'):
                    _reset_workout_state()
                elif k == ord('v'):
                    voice_on = not voice_on and coach.available
                    coach.set_enabled(voice_on)
                elif k == ord('d'):
                    debug = not debug

            elif state == AppState.RESIGN_DIALOG:
                bg = last_frame if last_frame is not None else blank_frame.copy()
                render_resign_dialog(bg, resign_buttons)
                cv2.imshow(window, bg)

                click = consume_click()
                if click:
                    clicked = hit_test(resign_buttons, click[0], click[1])
                    if clicked:
                        if clicked.action == "resume":
                            state = AppState.WORKOUT
                        elif clicked.action == "end":
                            _close_cam()
                            coach.stop()
                            logger.save()
                            summary_data = {
                                "exercise": EXERCISES[menu_state.selected_exercise],
                                "total_reps": logger.total_reps,
                                "avg_score": logger.avg_score,
                                "duration_str": f"{logger.duration / 60:.0f}:{logger.duration % 60:02.0f}",
                                "top_issue": logger.most_frequent_issue or "None",
                            }
                            state = AppState.SUMMARY

                k = cv2.waitKey(30) & 0xFF
                if k == 27:
                    state = AppState.WORKOUT
                elif k == ord('q'):
                    break

            elif state == AppState.SUMMARY:
                render_summary(blank_frame, summary_data, summary_buttons)
                cv2.imshow(window, blank_frame)

                click = consume_click()
                if click:
                    clicked = hit_test(summary_buttons, click[0], click[1])
                    if clicked and clicked.action == "continue":
                        state = AppState.MENU
                        coach = VoiceCoach(enabled=True)
                        voice_on = coach.available

                k = cv2.waitKey(30) & 0xFF
                if k == ord('q') or k == 27:
                    break
                if k == 13 or k == 10:
                    state = AppState.MENU
                    coach = VoiceCoach(enabled=True)
                    voice_on = coach.available

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        _close_cam()
        cv2.destroyAllWindows()
        detector.release()
        coach.stop()
        if logger.total_reps > 0:
            logger.save()


if __name__ == "__main__":
    main()
