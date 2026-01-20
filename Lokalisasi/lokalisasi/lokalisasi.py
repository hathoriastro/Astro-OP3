#!/usr/bin/env python3

# menampilkan data imu
# flowchart perhitungan joint states
# coba implementasi di hardware

import math
import time
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from op3_walking_module_msgs.msg import WalkingParam

import matplotlib
matplotlib.use("TkAgg")  # ganti sesuai env-mu kalau perlu
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

class SimpleLocalizationFused(Node):
    # =======================
    # KONFIGURASI & KONSTANTA
    # =======================
    FIELD_LENGTH = 9.0
    FIELD_WIDTH  = 6.0
    GOAL_WIDTH   = 2.6
    GOAL_DEPTH   = 0.6
    CENTER_DIAM  = 1.5
    PENALTY_DIST = 2.1

    # Gain estimasi dari JointState (heuristik)
    JS_VX_GAIN          = 0.38                    # knee/ankle -> m/s
    JS_VY_GAIN          = 0.25                    # hip roll diff -> m/s
    JS_W_GAIN           = math.radians(45.0)      # hip pitch diff -> rad/s

    # Threshold deteksi “sedang berjalan”
    KNEE_DTH       = 0.015
    ANKLE_DTH      = 0.015
    HIPROLL_DTH    = 0.01
    PHASE_ALT_DTH  = 0.6          # rasio alternating (kanan vs kiri)

    # Fusion/Timing
    WP_STALE_SEC   = 0.35          # set_params dianggap basi setelah ini

    def __init__(self):
        super().__init__('simple_localization_fused')

        # Pose
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0  # rad → absolut dari IMU

        # Last timestamps
        self.last_intg = time.time()
        self.last_wp_ts = 0.0

        # Buffer JointState
        self.prev_js = None
        self.last_js = None

        # Target perpindahan dari set_params (TOTAL untuk 2 langkah)
        self.target_dx_two = 0.0
        self.target_dy_two = 0.0
        self.remaining_dx = 0.0
        self.remaining_dy = 0.0

        # ---- ROS subs ----
        self.create_subscription(Imu, "/robotis_op3/imu", self.imu_cb, 20)
        self.create_subscription(WalkingParam, "/robotis/walking/set_params", self.wp_cb, 10)
        self.create_subscription(JointState, "/robotis_op3/joint_states", self.js_cb, 50)

        # ---- Plot setup ----
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.robot_marker, = self.ax.plot([], [], 'ro', markersize=8, label='Robot')
        self.orientation_line, = self.ax.plot([], [], 'r-', linewidth=2)
        self._setup_field()

        # Timer integrasi 50 Hz
        self.timer = self.create_timer(0.02, self.integrate)

        self.get_logger().info("✅ simple_localization_fused started")

    # ==============
    #  SUBSCRIBERS
    # ==============
    def imu_cb(self, msg: Imu):
        # quaternion → yaw
        x, y, z, w = msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)  # absolut, tidak diintegrasi

    def wp_cb(self, msg: WalkingParam):
        # Anggap x_move_amplitude dan y_move_amplitude = TOTAL perpindahan untuk 2 langkah (meter)
        self.last_wp_ts = time.time()
        self.target_dx_two = float(msg.x_move_amplitude)
        self.target_dy_two = float(msg.y_move_amplitude)
        # Reset sisa target; akan diisi saat integrasi berikutnya
        self.remaining_dx = self.target_dx_two
        self.remaining_dy = self.target_dy_two

    def js_cb(self, msg: JointState):
        self.prev_js, self.last_js = self.last_js, msg

    # =======================
    #  DETEKSI JALAN & KECEPATAN DARI JOINTSTATE
    # =======================
    def _joint_delta(self, name, cur, prev):
        try:
            i1 = cur.name.index(name); i0 = prev.name.index(name)
            return cur.position[i1] - prev.position[i0]
        except ValueError:
            return 0.0

    def _js_velocity_estimate(self):
        """Estimasi (vx, vy, w) dari perubahan joint antar dua sampel."""
        if self.prev_js is None or self.last_js is None:
            return 0.0, 0.0, 0.0, False

        # waktu antar sampel
        try:
            t1 = self.last_js.header.stamp.sec + self.last_js.header.stamp.nanosec*1e-9
            t0 = self.prev_js.header.stamp.sec + self.prev_js.header.stamp.nanosec*1e-9
            dt = max(t1 - t0, 1e-3)
        except Exception:
            dt = 0.02

        # ambil delta beberapa sendi kunci
        d_r_knee   = abs(self._joint_delta('r_knee', self.last_js, self.prev_js))
        d_l_knee   = abs(self._joint_delta('l_knee', self.last_js, self.prev_js))
        d_r_ank    = abs(self._joint_delta('r_ank_pitch', self.last_js, self.prev_js))
        d_l_ank    = abs(self._joint_delta('l_ank_pitch', self.last_js, self.prev_js))
        d_r_hroll  = abs(self._joint_delta('r_hip_roll', self.last_js, self.prev_js))
        d_l_hroll  = abs(self._joint_delta('l_hip_roll', self.last_js, self.prev_js))
        # untuk rotasi pakai hip_pitch beda sisi
        d_r_hpitch = abs(self._joint_delta('r_hip_pitch', self.last_js, self.prev_js))
        d_l_hpitch = abs(self._joint_delta('l_hip_pitch', self.last_js, self.prev_js))

        # total gerak kedua kaki
        right_total = d_r_knee + d_r_ank
        left_total  = d_l_knee + d_l_ank
        total = right_total + left_total

        # deteksi walking: ada gerak signifikan + alternating (rasio)
        moving = (
            (d_r_knee > self.KNEE_DTH or d_r_ank > self.ANKLE_DTH or
             d_l_knee > self.KNEE_DTH or d_l_ank > self.ANKLE_DTH)
        )
        ratio = (min(right_total, left_total) / max(right_total, left_total)) if max(right_total, left_total) > 1e-6 else 0.0
        alternating = ratio > self.PHASE_ALT_DTH

        is_walking = moving and alternating

        # kecepatan estimasi dari JS (heuristik)
        vx_js = self.JS_VX_GAIN * total / dt                     # maju
        vy_js = self.JS_VY_GAIN * (d_r_hroll - d_l_hroll) / dt   # lateral (sign ikut kanan - kiri)
        w_js  = self.JS_W_GAIN  * (d_r_hpitch - d_l_hpitch) / dt # rotasi

        # deadzone kecil agar gak “merambat”
        if abs(vx_js) < 0.01: vx_js = 0.0
        if abs(vy_js) < 0.01: vy_js = 0.0
        if abs(w_js)  < math.radians(2): w_js = 0.0

        return vx_js, vy_js, w_js, is_walking

    # ============
    #  INTEGRASI
    # ============
    def integrate(self):
        now = time.time()
        dt = now - self.last_intg
        if dt <= 0.0:
            return
        self.last_intg = now

        # deteksi fase berjalan
        vx_js, vy_js, w_js, walking = self._js_velocity_estimate()

        # Jika set_params baru diterima namun belum ada gerak kaki, jangan langsung loncat
        # remaining_dx/dy sudah di-set pada wp_cb(). Di sini hanya “habiskan” saat walking True.
        move_dx, move_dy = 0.0, 0.0
        if walking:
            # Fraksi kemajuan per tick dipandu oleh besar gerak kaki (lebih besar → lebih cepat)
            speed_norm = min(0.12, max(0.02, 0.6*(abs(vx_js) + abs(vy_js))))  # 2%..12% per tick
            frac = speed_norm

            # Ambil fraksi dari sisa target (agar tidak overshoot)
            take_dx = np.sign(self.remaining_dx) * min(abs(self.remaining_dx), abs(self.remaining_dx)*frac)
            take_dy = np.sign(self.remaining_dy) * min(abs(self.remaining_dy), abs(self.remaining_dy)*frac)

            self.remaining_dx -= take_dx
            self.remaining_dy -= take_dy

            move_dx, move_dy = take_dx, take_dy

        # transform ke koordinat global pakai IMU yaw
        dx_global = move_dx * math.cos(self.yaw) - move_dy * math.sin(self.yaw)
        dy_global = move_dx * math.sin(self.yaw) + move_dy * math.cos(self.yaw)

        self.x += dx_global
        self.y += dy_global

        # clamp di dalam lapangan
        self.x = float(np.clip(self.x, -self.FIELD_LENGTH/2, self.FIELD_LENGTH/2))
        self.y = float(np.clip(self.y, -self.FIELD_WIDTH/2,  self.FIELD_WIDTH/2))

        self._update_plot()

        self.ax.set_title(f"KRSBI Humanoid Field (9×6 m) — Yaw={math.degrees(self.yaw):6.2f}°\n")

        print(f"Yaw={math.degrees(self.yaw):6.2f}° | Pos=({self.x:6.3f},{self.y:6.3f}) m "
              f"| Remain=({self.remaining_dx:.3f},{self.remaining_dy:.3f}) m | walking={walking}")

    # =================
    #  PLOTTING FIELD
    # =================
    def _setup_field(self):
        ax = self.ax
        ax.clear()
        ax.set_title("KRSBI Humanoid Field (9×6 m) — Localization")
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        ax.set_xlim(-self.FIELD_LENGTH/2 - 0.5, self.FIELD_LENGTH/2 + 0.5)
        ax.set_ylim(-self.FIELD_WIDTH/2 - 0.5, self.FIELD_WIDTH/2 + 0.5)
        ax.set_aspect('equal', adjustable='box')

        # Garis luar
        field_rect = Rectangle(
            (-self.FIELD_LENGTH/2, -self.FIELD_WIDTH/2),
            self.FIELD_LENGTH, self.FIELD_WIDTH,
            fill=False, color='black', linewidth=2)
        ax.add_patch(field_rect)

        # Garis tengah & lingkaran tengah
        ax.plot([0, 0], [-self.FIELD_WIDTH/2, self.FIELD_WIDTH/2], 'k--', linewidth=1)
        ax.add_patch(Circle((0, 0), self.CENTER_DIAM/2, fill=False, color='black', linestyle='--'))

        # Area gawang + titik penalti di dua sisi
        for side in (-1, 1):
            goal = Rectangle(
                (side * self.FIELD_LENGTH/2 - side * self.GOAL_DEPTH, -self.GOAL_WIDTH/2),
                self.GOAL_DEPTH * side, self.GOAL_WIDTH,
                fill=False, color='blue')
            ax.add_patch(goal)
            px = side * (self.FIELD_LENGTH/2 - self.PENALTY_DIST)
            ax.plot(px, 0.0, 'bx', ms=8)

        # Marker robot & heading
        self.robot_marker, = ax.plot(self.x, self.y, 'ro', markersize=8, label='Robot')
        head_x = self.x + 0.5 * math.cos(self.yaw)
        head_y = self.y + 0.5 * math.sin(self.yaw)
        self.orientation_line, = ax.plot([self.x, head_x], [self.y, head_y], 'r-', linewidth=2)
        ax.legend(loc='upper right')

        plt.draw(); plt.pause(0.001)

    def _update_plot(self):
        arrow_len = 0.5
        hx = self.x + arrow_len * math.cos(self.yaw)
        hy = self.y + arrow_len * math.sin(self.yaw)
        self.robot_marker.set_data([self.x], [self.y])
        self.orientation_line.set_data([self.x, hx], [self.y, hy])
        plt.draw(); plt.pause(0.01)


def main(args=None):
    rclpy.init(args=args)
    node = SimpleLocalizationFused()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()