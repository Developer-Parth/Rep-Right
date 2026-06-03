# Contributing to AI Fitness Coach

Thanks for your interest in contributing! This project is built with Python, OpenCV, and MediaPipe, and welcomes contributions of all kinds — bug fixes, new features, documentation improvements, or new exercise modes.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Development Setup](#development-setup)
3. [Project Architecture](#project-architecture)
4. [Coding Guidelines](#coding-guidelines)
5. [Pull Request Process](#pull-request-process)
6. [Feature Ideas](#feature-ideas)

---

## Code of Conduct

Be respectful, constructive, and inclusive. Harassment, trolling, and personal attacks will not be tolerated.

---

## Development Setup

### Prerequisites

- Python 3.10+
- A webcam
- [Git](https://git-scm.com/)

### Setup

```bash
git clone https://github.com/Developer-Parth/Rep-Right.git
cd ai-fitness-coach
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
python main.py
```

The MediaPipe pose model (`pose_landmarker_lite.task`) is downloaded automatically on first run. If needed, you can download it manually from the [MediaPipe Model Zoo](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task) and place it in the project root.

---

## Project Architecture

```
main.py               Entry point — camera capture, main loop, rendering, keyboard input
pose_detector.py      MediaPipe wrapper — runs inference, smooths landmarks, draws skeleton
rep_counter.py        Finite state machine — tracks squat phases and counts reps
squat_analyzer.py     Form analysis — checks depth, knee-toe, valgus, back angle per frame
voice_coach.py        Threaded TTS — queues spoken cues with cooldowns and message cycling
session_logger.py     JSON logger — writes session summaries to session_log.json
config.py             Central config — all thresholds, deduction amounts, coaching messages
utils.py              Shared utilities — angle calc, moving averages, point smoothing, colors
```

### Data Flow

```
Camera → frame (mirrored)
  └── every N frames → PoseDetector.process() → smoothed landmarks + skeleton draw
  └── skipped frames → PoseDetector.draw_cached() → replay last skeleton
  └── hip Y → RepCounter.update() → FSM state transition
  └── landmarks → SquatAnalyzer.analyse() → form score + issues list
       └── issues sustained for N frames → VoiceCoach.alert() → TTS
       └── on rep complete → logger.record_rep()
  └── draw_overlay() → OpenCV window
  └── on exit → logger.save() → session_log.json
```

### Key Design Decisions

- **Performance**: `model_complexity=0` + inference every 2nd frame + cached skeleton replay
- **Smoothing**: Per-landmark 3-axis moving average (5-frame window) + hip Y moving average (7-frame)
- **Side selection**: Auto-chooses the side with better visibility (user stands side-on)
- **Rep scoring**: Uses the worst (minimum) frame score across the entire rep
- **Voice gating**: Issue must persist for 6 consecutive frames before TTS fires

---

## Coding Guidelines

### Style

- Follow **[PEP 8](https://peps.python.org/pep-0008/)**
- Use type hints for all function signatures
- Keep functions small and single-purpose
- Use descriptive variable names

### Naming

| Convention | Example |
|---|---|
| Classes | `PascalCase` — `RepCounter`, `SquatAnalyzer` |
| Functions / methods | `snake_case` — `draw_overlay`, `finalize_rep` |
| Constants | `UPPER_SNAKE` — `MIN_VISIBILITY`, `INFERENCE_EVERY` |
| Private members | Prefix with `_` — `self._calibrated`, `self._smoothers` |

### Testing

- Include tests for new functionality when possible
- Verify nothing is broken before submitting:
  ```bash
  python -c "import py_compile, os; [py_compile.compile(f, doraise=True) for f in os.listdir() if f.endswith('.py')]"
  ```

### Import Order

1. Standard library (`import sys`, `from enum import Enum`)
2. Third-party (`import cv2`, `import numpy as np`)
3. Local (`from config import ...`, `from utils import ...`)

---

## Pull Request Process

1. **Fork** the repository and create a feature branch from `main`
2. **Make your changes** following the coding guidelines above
3. **Test** your changes — ensure the app runs without errors
4. **Commit** with a clear, descriptive message
   - Good: `Add hip hinge angle check to squat analyzer`
   - Avoid: `fix stuff`, `update`
5. **Open a PR** against the `main` branch
   - Describe what the change does and why
   - Mention any related issues
   - Attach screenshots if the change affects the visual output

### What to Expect

- I'll review your PR within a few days
- I may request changes or ask questions
- Once approved, I'll merge it

---

## Feature Ideas

If you're looking for ideas to contribute, here are some that would be great:

- **New exercises**: Add push-up, deadlift, or overhead press analysis modes
- **Web GUI**: Build a simple web interface (Flask/FastAPI) to control the coach from a phone
- **Rep history charts**: Better visual analytics of progress over time
- **Video recording**: Save sessions as video with overlay
- **Multi-person support**: Track multiple people in frame
- **Mobile support**: Adapt for smartphone cameras (via IP webcam)

---

## Getting Help

- Open an [issue](https://github.com/Developer-Parth/Rep-Right/issues) for bugs or questions
- Tag with `bug`, `enhancement`, or `question` labels

---

Built by [Parth Thukral](https://parththukral.xyz)
