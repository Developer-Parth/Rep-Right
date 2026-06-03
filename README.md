# AI Fitness Coach

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Real-time squat form analysis** powered by Python, OpenCV, and MediaPipe.
> It watches you through your webcam, counts reps, scores your form, and speaks corrections out loud — all running locally, no internet required.

---

## Features

| Capability | How it Works |
|---|---|
| **Pose Tracking** | MediaPipe PoseLandmarker runs every 2nd frame; cached skeleton replays on skipped frames for smooth full-fps visuals |
| **Rep Counting** | 5-state FSM — `Calibrating → Standing → Descending → Bottom → Ascending → Standing (1 rep)` |
| **Form Checks** | Depth (hip below knee), knee-over-toe, knee valgus (inward collapse), back angle (forward lean) |
| **Scoring** | Each rep starts at 100; deductions per fault; session average tracked |
| **Voice Coaching** | Cross-platform TTS (macOS `say`, Linux `espeak`, Windows PowerShell SAPI) with cooldowns and message cycling |
| **Live Overlay** | Rep counter, phase badge, form score bar, coaching text, per-issue progress bars, score sparkline |
| **Session Logging** | Appended to `session_log.json` on exit — reps, scores, corrections by type, duration |
| **Debug Mode** | Toggle raw back/knee angles overlay |

---

## Quick Start

### Prerequisites

- Python 3.10+
- A webcam

### Install

```bash
git clone https://github.com/Developer-Parth/Rep-Right.git
cd ai-fitness-coach
pip install -r requirements.txt
```

### Run

```bash
python main.py
```

On Windows, you can also double-click `run.bat`.

### Setup

1. Stand **1.5–2 m** from the camera
2. Face **side-on** (profile view) — not facing the camera
3. Keep **full body in frame** (head to feet)
4. Stand still for ~1 second while the coach calibrates
5. Start squatting!

---

## Controls

| Key | Action |
|---|---|
| `q` / `ESC` | Quit and save session to `session_log.json` |
| `p` | Pause / resume |
| `r` | Reset reps, scores, and calibration |
| `v` | Toggle voice feedback |
| `d` | Toggle debug overlay (raw back/knee angles) |

---

## Project Structure

```
ai-fitness-coach/
├── main.py               Entry point — OpenCV capture + render loop
├── config.py             Thresholds, cooldowns, coaching messages
├── pose_detector.py      MediaPipe PoseLandmarker wrapper + smoothing + cached draw
├── rep_counter.py        Squat FSM + calibration
├── squat_analyzer.py     Depth / knee-toe / valgus / back-angle checks + scoring
├── voice_coach.py        Queue-based threaded TTS worker
├── session_logger.py     Per-session JSON logger
├── utils.py              Math helpers (angle, moving average, colours)
├── pose_landmarker_lite.task  MediaPipe pose model
├── requirements.txt
├── run.bat               Windows launcher
├── session_log.json      Session history (appended on each run)
├── README.md
└── CONTRIBUTING.md
```

---

## How Scoring Works

- Each rep starts at **100 points**
- **−25** if hips don't go deep enough
- **−15** if knees travel too far forward over toes
- **−20** if knees cave inward (valgus collapse)
- **−15** if chest drops / excessive forward lean
- The rep's final score = **worst frame** during that rep
- Session score = **average** of all completed reps

---

## Configuration

All thresholds live in `config.py`. Key ones to tune:

| Setting | Default | What it Controls |
|---|---|---|
| `DEPTH_HIP_BELOW_KNEE_PX` | `-5` | How strict depth-below-parallel is |
| `KNEE_TOE_SLACK_FRACTION` | `0.08` | Forward knee travel allowance |
| `VALGUS_SLACK_FRACTION` | `0.04` | Inward knee collapse allowance |
| `BACK_LEAN_MAX_FROM_VERTICAL` | `50.0` | Max torso lean angle (°) |
| `INFERENCE_EVERY` | `2` | Run pose model every N frames |
| `CONSECUTIVE_FRAMES` | `6` | Bad frames before voice alert |
| `VOICE_COOLDOWN_SEC` | `4.0` | Min seconds before re-speaking a cue |
| `ISSUE_MESSAGES` | — | The actual coaching lines |

---

## Platform Support

| Platform | Camera Backend | Voice Backend | Status |
|---|---|---|---|
| **Windows** | DirectShow (default) | PowerShell System.Speech | ✅ |
| **macOS** | AVFoundation | `say` | ✅ |
| **Linux** | V4L2 (default) | `espeak` / `espeak-ng` / `spd-say` | ✅ |

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| "Cannot open camera" | Close Zoom, Teams, OBS, etc. Grant camera permission in OS settings |
| No skeleton / "Step into frame" | Step back — full body must be visible. Improve lighting. Avoid busy backgrounds |
| Laggy / low FPS | Set `INFERENCE_EVERY = 3` in `config.py` or reduce `FRAME_WIDTH` / `FRAME_HEIGHT` |
| No voice | Press `v` to toggle on. On Linux: `sudo apt install espeak` |
| Reps miscounting | Stand still longer during calibration. Tune `DESCENT_THRESHOLD` / `COMPLETE_THRESHOLD` |

---

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

This project is open source under the [MIT License](LICENSE).

---

Built by [Parth Thukral](https://parththukral.xyz)
