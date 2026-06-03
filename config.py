CAMERA_INDEX = 0
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720

INFERENCE_EVERY = 2

DEPTH_HIP_BELOW_KNEE_PX = -5

KNEE_TOE_SLACK_FRACTION = 0.08

VALGUS_SLACK_FRACTION = 0.04

BACK_LEAN_MAX_FROM_VERTICAL = 50.0

DESCENT_THRESHOLD  = 0.04
BOTTOM_THRESHOLD   = 0.15
ASCENT_THRESHOLD   = 0.05
COMPLETE_THRESHOLD = 0.06

DEDUCT_DEPTH    = 25
DEDUCT_KNEE_TOE = 15
DEDUCT_VALGUS   = 20
DEDUCT_BACK     = 15

VOICE_COOLDOWN_SEC     = 4.0
GOOD_REP_VOICE_COOLDOWN = 8.0
CONSECUTIVE_FRAMES     = 6

ISSUE_MESSAGES: dict[str, list[str]] = {
    "Go deeper": [
        "Go deeper. Hip must pass below your knee.",
        "Not deep enough. Drop those hips.",
        "Sit lower. Break parallel.",
    ],
    "Knees too far forward": [
        "Knees too far forward. Push your hips back.",
        "Shift weight into your heels.",
        "Sit back more. Load the hips, not the knees.",
    ],
    "Push knees out": [
        "Push your knees out. Don't let them cave in.",
        "Drive your knees outward. Follow your toes.",
        "Knees collapsing. Spread the floor with your feet.",
    ],
    "Keep chest up": [
        "Keep your chest up. Don't lean forward.",
        "Proud chest. Upright torso.",
        "Less forward lean. Stay tall.",
    ],
    "Good rep": [
        "Great rep!",
        "Perfect form. Keep going.",
        "Beautiful squat.",
        "Solid rep. Stay tight.",
    ],
    "Rep complete": [
        "Rep complete.",
        "Nice one.",
        "Good work.",
    ],
}
