
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from config import EXERCISES

_C = {
    "bg":       (5, 5, 5),
    "black":    (0, 0, 0),
    "cyan":     (255, 255, 50),
    "magenta":  (255, 50, 255),
    "green":    (50, 255, 50),
    "orange":   (50, 180, 255),
    "red":      (50, 50, 255),
    "yellow":   (50, 255, 255),
    "white":    (230, 230, 230),
    "gray":     (100, 100, 100),
    "panel":    (15, 10, 20),
}


def _glow_text(frame, text, pos, color, scale=0.55, thickness=1, glow_layers=3):
    for i in range(glow_layers):
        factor = 1.0 / (2 ** (glow_layers - i))
        gc = tuple(int(c * factor) for c in color)
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                     gc, thickness + (glow_layers - i) * 3, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                _C["black"], thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def _put_text(frame, text, pos, color, scale=0.55, thickness=1):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                _C["black"], thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def _corner_brackets(frame, x, y, w, h, color, length=12, thickness=2):
    cv2.line(frame, (x, y + length), (x, y), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x, y), (x + length, y), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x + w - length, y), (x + w, y), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x + w, y), (x + w, y + length), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x, y + h - length), (x, y + h), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x, y + h), (x + length, y + h), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x + w - length, y + h), (x + w, y + h), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x + w, y + h - length), (x + w, y + h), color, thickness, cv2.LINE_AA)


def _draw_rounded_box(frame, x, y, w, h, color, thickness=1):
    r = 6
    cv2.rectangle(frame, (x + r, y), (x + w - r, y + h), color, thickness, cv2.LINE_AA)
    cv2.rectangle(frame, (x, y + r), (x + w, y + h - r), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x + r, y), (x + 1, y), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x + w - r, y), (x + w - 1, y), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x + r, y + h), (x + 1, y + h), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x + w - r, y + h), (x + w - 1, y + h), color, thickness, cv2.LINE_AA)


def _panel(frame, x, y, w, h, alpha=0.55):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), _C["panel"], -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _fill_rect(frame, x, y, w, h, color, alpha=0.85):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


# ─── Button helpers ────────────────────────────────────────

class Button:
    def __init__(self, label, x, y, w, h, action=""):
        self.label = label
        self.rect = (x, y, w, h)
        self.action = action
        self.hovered = False


def hit_test(buttons, mx, my):
    for b in buttons:
        x, y, w, h = b.rect
        if x <= mx <= x + w and y <= my <= y + h:
            return b
    return None


def draw_button(frame, btn, color, glow=False):
    x, y, w, h = btn.rect
    if btn.hovered:
        _fill_rect(frame, x - 1, y - 1, w + 2, h + 2, color, alpha=0.2)
    if glow:
        _glow_text(frame, btn.label, (x + 16, y + h - 8), color, scale=0.65, thickness=2, glow_layers=3)
    else:
        _put_text(frame, btn.label, (x + 16, y + h - 8), color, scale=0.65, thickness=2)
    _draw_rounded_box(frame, x, y, w, h, color, thickness=1)
    _corner_brackets(frame, x - 3, y - 3, w + 6, h + 6, color, length=8, thickness=1)


def draw_exercise_card(frame, label, x, y, w, h, selected=False):
    color = _C["cyan"] if selected else _C["gray"]
    _fill_rect(frame, x, y, w, h, (10, 8, 15), alpha=0.7)
    _draw_rounded_box(frame, x, y, w, h, color, thickness=1)
    prefix = "▸ " if selected else "  "
    _put_text(frame, f"{prefix}{label}", (x + 20, y + h - 10), color, scale=0.55, thickness=2)


# ─── Screen renderers ─────────────────────────────────────

def render_menu(frame, selected_idx: int, buttons: list[Button]):
    h, w = frame.shape[:2]
    frame[:] = _C["bg"]

    # Title
    _glow_text(frame, "REP", (w // 2 - 150, 140), _C["cyan"], scale=2.2, thickness=4, glow_layers=4)
    _glow_text(frame, "RIGHT", (w // 2 + 10, 180), _C["magenta"], scale=1.6, thickness=3, glow_layers=4)

    # Tagline
    _put_text(frame, "AI-Powered Exercise Form Analysis",
              (w // 2 - 170, 230), _C["gray"], scale=0.5, thickness=1)

    # Exercise section
    _put_text(frame, "SELECT EXERCISE", (w // 2 - 95, 290), _C["white"], scale=0.5, thickness=1)

    card_w, card_h = 260, 44
    card_x = (w - card_w) // 2
    card_y = 305
    draw_exercise_card(frame, EXERCISES[selected_idx], card_x, card_y, card_w, card_h, selected=True)

    # Hint
    _put_text(frame, "↑ ↓ navigate  |  Enter start  |  Click to select",
              (w // 2 - 200, h - 40), _C["gray"], scale=0.4, thickness=1)

    # Draw action buttons
    for btn in buttons:
        if btn.action == "start":
            draw_button(frame, btn, _C["cyan"], glow=True)

    return buttons


def render_resign_dialog(frame, buttons: list[Button]):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    dw, dh = 380, 200
    dx, dy = (w - dw) // 2, (h - dh) // 2
    _panel(frame, dx, dy, dw, dh, alpha=0.8)
    _corner_brackets(frame, dx, dy, dw, dh, _C["cyan"], length=14, thickness=2)

    _put_text(frame, "END EXERCISE?", (dx + 100, dy + 40), _C["white"], scale=0.7, thickness=2)

    for btn in buttons:
        if btn.action == "resume":
            draw_button(frame, btn, _C["green"])
        elif btn.action == "end":
            draw_button(frame, btn, _C["red"])

    return buttons


def render_summary(frame, data: dict, buttons: list[Button]):
    h, w = frame.shape[:2]
    frame[:] = _C["bg"]

    _glow_text(frame, "SESSION COMPLETE", (w // 2 - 200, 120),
               _C["cyan"], scale=1.1, thickness=3, glow_layers=3)

    lines = [
        ("Exercise",  data.get("exercise", "SQUAT")),
        ("Reps",      str(data.get("total_reps", 0))),
        ("Avg Score", f"{data.get('avg_score', 0):.1f}"),
        ("Duration",  data.get("duration_str", "0:00")),
    ]
    top = 200
    for i, (label, val) in enumerate(lines):
        _put_text(frame, label, (w // 2 - 180, top + i * 36),
                  _C["gray"], scale=0.5, thickness=1)
        _put_text(frame, val, (w // 2 + 20, top + i * 36),
                  _C["white"], scale=0.55, thickness=2)

    top_issue = data.get("top_issue")
    if top_issue:
        _put_text(frame, "Top Issue", (w // 2 - 180, top + 4 * 36),
                  _C["gray"], scale=0.5, thickness=1)
        _put_text(frame, top_issue, (w // 2 + 20, top + 4 * 36),
                  _C["orange"], scale=0.55, thickness=2)

    for btn in buttons:
        if btn.action == "continue":
            draw_button(frame, btn, _C["cyan"], glow=True)

    return buttons


def make_button(action, x, y, w=200, h=40):
    return Button(action.capitalize(), x, y, w, h, action)


# ─── Menu state (persistent) ──────────────────────────────

class MenuState:
    def __init__(self):
        self.selected_exercise = 0
        self.click_pos: Optional[tuple[int, int]] = None

    def set_click(self, x, y):
        self.click_pos = (x, y)

    def consume_click(self) -> Optional[tuple[int, int]]:
        pos = self.click_pos
        self.click_pos = None
        return pos
