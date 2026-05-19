# ============================================================
#  pi_fall_detection.py  —  Run this on RASPBERRY PI 4
#
#  RL Fix Summary:
#    - State now includes BOTH angle + angular velocity (2D state)
#    - Q-table updated with ACTUAL next-frame state (not same frame)
#    - prev_state and prev_action stored across frames correctly
#    - Fall confirmed only after N consecutive fall frames (debounce)
#
#  Buzzer wiring:
#    Buzzer (+) → GPIO 17 (Pin 11)
#    Buzzer (−) → GND    (Pin 9)
#
#  Install on Pi:
#    pip install ultralytics opencv-python twilio RPi.GPIO
# ============================================================

import cv2
import numpy as np
import random
import time
import RPi.GPIO as GPIO
from ultralytics import YOLO
from twilio.rest import Client

# ─────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────

LAPTOP_STREAM_URL  = "http://10.16.120.208:5000/video"   # Change to your laptop IP

BUZZER_PIN         = 17

TWILIO_ACCOUNT_SID = "ACf0c60be8aab6bfe092081bd13f395a27"
TWILIO_AUTH_TOKEN  = "668e329cbb395ec4e05f0fefe431c727"
TWILIO_FROM_NUMBER = "+1 762 237 2700"
ALERT_TO_NUMBER    = "+918148002483"

ALERT_COOLDOWN_SEC  = 30
BUZZER_BEEP_SEC     = 2.0
FALL_CONFIRM_FRAMES = 3     # Consecutive fall decisions before alerting


# ─────────────────────────────────────────
#  GPIO SETUP
# ─────────────────────────────────────────

GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.output(BUZZER_PIN, GPIO.LOW)


def buzz(duration=BUZZER_BEEP_SEC):
    GPIO.output(BUZZER_PIN, GPIO.HIGH)
    time.sleep(duration)
    GPIO.output(BUZZER_PIN, GPIO.LOW)


# ─────────────────────────────────────────
#  TWILIO SETUP
# ─────────────────────────────────────────

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)


def send_sms_alert():
    try:
        msg = twilio_client.messages.create(
            body="FALL DETECTED! Please check the monitored person immediately.",
            from_=TWILIO_FROM_NUMBER,
            to=ALERT_TO_NUMBER
        )
        print(f"[SMS] Sent. SID: {msg.sid}")
    except Exception as e:
        print(f"[SMS ERROR] {e}")


# ─────────────────────────────────────────
#  YOLO POSE MODEL
# ─────────────────────────────────────────

print("[INFO] Loading YOLOv8 pose model...")
pose_model = YOLO("yolov8n-pose.pt")


# ─────────────────────────────────────────
#  RL Q-LEARNING (CORRECTED)
#
#  State space is 2D:
#    Dim 1 — angle bucket    (0-9):  torso angle 0-180 in 18-degree bins
#    Dim 2 — velocity bucket (0-4):  rate of angle change per frame
#
#  Why velocity matters:
#    Angle alone cannot distinguish a person lying still from actively falling.
#    Velocity catches the rapid drop that is the signature of a fall event.
#
#  Q-table: (10 x 5, 2) = (50 states, 2 actions)
# ─────────────────────────────────────────

N_ANGLE_BINS = 10
N_VEL_BINS   = 5
N_STATES     = N_ANGLE_BINS * N_VEL_BINS   # 50
actions      = [0, 1]                       # 0=normal, 1=fall

q_table = np.zeros((N_STATES, 2))

alpha   = 0.1
gamma   = 0.9
epsilon = 0.1


def angle_to_bin(angle):
    return min(int(angle // 18), N_ANGLE_BINS - 1)


def velocity_to_bin(dangle):
    """
    Bin 0: fast drop  (< -15 deg/frame)  — likely falling
    Bin 1: slow drop  (-15 to -3)
    Bin 2: stationary (-3 to +3)
    Bin 3: slow rise  (+3 to +15)
    Bin 4: fast rise  (> +15)            — rising from ground
    """
    if   dangle < -15: return 0
    elif dangle <  -3: return 1
    elif dangle <   3: return 2
    elif dangle <  15: return 3
    else:              return 4


def encode_state(angle_bin, vel_bin):
    return angle_bin * N_VEL_BINS + vel_bin


def get_state(keypoints, prev_angle):
    """
    Compute torso angle (shoulder -> hip vector) and angular velocity.
    COCO keypoints: 5 = left shoulder, 11 = left hip
    """
    shoulder = keypoints[5]
    hip      = keypoints[11]
    dx = shoulder[0] - hip[0]
    dy = shoulder[1] - hip[1]
    angle  = abs(np.degrees(np.arctan2(dy, dx)))
    dangle = angle - prev_angle
    state  = encode_state(angle_to_bin(angle), velocity_to_bin(dangle))
    return state, angle, dangle


def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)
    return int(np.argmax(q_table[state]))


def get_reward(angle, dangle, action):
    """
    Rewards use both angle AND velocity.
    is_horizontal + is_falling together = strongest fall signal.
    """
    is_horizontal = angle < 35
    is_falling    = dangle < -10

    if is_horizontal and is_falling and action == 1:
        return 15    # caught a live fall event — best reward
    if is_horizontal and action == 1:
        return 8     # person is down (could be fallen or resting)
    if not is_horizontal and action == 0:
        return 2     # correctly identified normal upright posture
    return -5        # wrong decision


def update_q_table(state, action, reward, next_state):
    """
    Bellman Q-learning update.
    KEY FIX: next_state is the actual state from the NEXT frame,
    not the same frame (which was the bug in the original code).
    """
    current_q = q_table[state, action]
    best_next  = np.max(q_table[next_state])
    q_table[state, action] = current_q + alpha * (
        reward + gamma * best_next - current_q
    )


# ─────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────

print(f"[INFO] Connecting to: {LAPTOP_STREAM_URL}")
cap = cv2.VideoCapture(LAPTOP_STREAM_URL)

if not cap.isOpened():
    print("[ERROR] Cannot connect to stream. Is laptop_stream.py running?")
    GPIO.cleanup()
    exit(1)

print("[INFO] Connected. Starting fall detection...\n")

prev_angle   = 90.0
prev_state   = encode_state(angle_to_bin(90), velocity_to_bin(0))
prev_action  = 0

last_alert_time  = 0
fall_frame_count = 0

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("[WARN] Stream lost. Retrying...")
            time.sleep(1)
            cap = cv2.VideoCapture(LAPTOP_STREAM_URL)
            continue

        results   = pose_model(frame, verbose=False)
        annotated = results[0].plot()

        if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            keypoints = results[0].keypoints.xy[0].cpu().numpy()

            if keypoints[5].sum() == 0 or keypoints[11].sum() == 0:
                cv2.imshow("RL Fall Detection [Pi]", annotated)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            # Current frame state
            curr_state, curr_angle, dangle = get_state(keypoints, prev_angle)

            # Update Q-table: previous action in previous state led to current state
            reward = get_reward(curr_angle, dangle, prev_action)
            update_q_table(prev_state, prev_action, reward, curr_state)

            # Choose action for current state
            action = choose_action(curr_state)

            # Debounce: require N consecutive fall frames before alerting
            if action == 1:
                fall_frame_count += 1
            else:
                fall_frame_count = 0

            confirmed_fall = fall_frame_count >= FALL_CONFIRM_FRAMES

            # Overlay
            a_bin = angle_to_bin(curr_angle)
            v_bin = velocity_to_bin(dangle)
            cv2.putText(annotated,
                        f"Angle:{curr_angle:.1f}  dA:{dangle:+.1f}  S:({a_bin},{v_bin})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(annotated,
                        f"Action:{'FALL' if action==1 else 'NORMAL'}  Streak:{fall_frame_count}",
                        (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            if confirmed_fall:
                cv2.putText(annotated, "FALL DETECTED!", (40, 105),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
                print(f"[FALL] angle={curr_angle:.1f} dangle={dangle:+.1f}")

                now = time.time()
                if now - last_alert_time >= ALERT_COOLDOWN_SEC:
                    last_alert_time = now
                    buzz(BUZZER_BEEP_SEC)
                    send_sms_alert()

            # Carry state forward to next frame
            prev_state  = curr_state
            prev_action = action
            prev_angle  = curr_angle

        #cv2.imshow("RL Fall Detection [Pi]", annotated)
        #if cv2.waitKey(1) & 0xFF == 27:
            #break

except KeyboardInterrupt:
    print("\n[INFO] Stopped by user.")

finally:
    cap.release()
    #cv2.destroyAllWindows()
    GPIO.output(BUZZER_PIN, GPIO.LOW)
    GPIO.cleanup()
    print("[INFO] Cleanup done.")
