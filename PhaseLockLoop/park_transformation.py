# -*- coding: utf-8 -*-
"""
Park Transformation Visualization with Manim (No-LaTeX Version)

Tested with: Manim Community v0.19.0

Render examples:
    manim -pqh park_transformation.py ParkTransformation
    # 1080p
    # manim -p -r 1920,1080 park_transformation.py ParkTransformation

This scene explains the abc → αβ0 (Clarke) → dq0 (Park) transforms
without using MathTex/LaTeX. All formula panels are drawn using Text,
so there is NO LaTeX dependency.

"""
from manim import *
import numpy as np

# ---------- Configurable Parameters ----------
F_ELECTRICAL = 1.0          # electrical frequency [Hz]
OMEGA = TAU * F_ELECTRICAL  # electrical speed [rad/s]
V_AMPLITUDE = 1.0           # amplitude of phase voltages
PHASE_OFFSET = 0.0          # initial electrical angle [rad]
SHOW_ZERO_AXIS = True       # toggle to display 0-sequence
DURATION = 8                # animation seconds

# ---------- Math Helpers ----------

def abc_waveforms(t, A=V_AMPLITUDE, omega=OMEGA, phi=PHASE_OFFSET):
    Va = A * np.cos(omega * t + phi)
    Vb = A * np.cos(omega * t + phi - 2*np.pi/3)
    Vc = A * np.cos(omega * t + phi + 2*np.pi/3)
    return np.array([Va, Vb, Vc])


def clarke_matrix():
    """abc -> αβ0 matrix (power-invariant variant with 2/3 scaling)."""
    return np.array([
        [2/3, -1/3, -1/3],
        [0,   np.sqrt(3)/3, -np.sqrt(3)/3],
        [1/3, 1/3, 1/3],
    ])


def park_matrix(theta):
    """αβ0 -> dq0 rotation (d along cosθ, q along sinθ)."""
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([
        [ c,  s, 0],
        [-s,  c, 0],
        [ 0,  0, 1],
    ])
    return R


def abc_to_dq0(abc, theta):
    T_clarke = clarke_matrix()
    alpha_beta_0 = T_clarke @ abc
    T_park = park_matrix(theta)
    dq0 = T_park @ alpha_beta_0
    return dq0


# ---------- Utility: projection helper ----------
def project_point_to_axis(point, axis_dir):
    """Project a point P (in scene coords) onto a directed axis through origin."""
    p = np.array([point[0], point[1]])
    u = np.array([axis_dir[0], axis_dir[1]])
    u = u / (np.linalg.norm(u) + 1e-9)
    proj_len = np.dot(p, u)
    proj = proj_len * u
    return np.array([proj[0], proj[1], 0.0])


# ---------- Manim Scene ----------
class ParkTransformation(Scene):
    def construct(self):
        self.camera.background_color = "#0d1321"
        title = Text("Park Transformation (abc → dq0)", weight=BOLD).scale(0.8)
        subtitle = Text("Clarke → Park with rotating reference frame", font_size=28, color=GRAY_B)
        header = VGroup(title, subtitle).arrange(DOWN, buff=0.2).to_edge(UP)
        self.play(FadeIn(header, shift=UP))

        # Left: time-domain phase waveforms; Right: space vector & rotating axes
        left_panel = Axes(
            x_range=[0, DURATION, 1], y_range=[-1.3, 1.3, 1],
            x_length=6.8, y_length=3.2,
            tips=False, axis_config={"include_numbers": False, "color": GRAY_B}
        )
        left_panel.to_edge(LEFT, buff=0.7).shift(DOWN*0.5)
        left_label = Text("Phase Voltages: a, b, c", font_size=26, color=GRAY_B).next_to(left_panel, UP, buff=0.2)

        right_plane = NumberPlane(
            x_range=[-1.6, 1.6, 1], y_range=[-1.6, 1.6, 1],
            x_length=5.0, y_length=5.0,
            faded_line_ratio=2,
            background_line_style={"stroke_color": GRAY_D, "stroke_opacity": 0.6}
        ).to_edge(RIGHT, buff=0.9).shift(DOWN*0.1)
        right_label = MathTex(r"\alpha\beta \text{ plane and dq frame}", font_size=26, color=GRAY_B).next_to(right_plane, UP, buff=0.2)

        # fade out the header
        self.play(FadeOut(header))
        self.play(FadeIn(left_panel), FadeIn(right_plane), FadeIn(left_label), FadeIn(right_label))

        # Time tracker (drives everything)
        t_tracker = ValueTracker(0.0)

        # --- Phase signals (left) ---
        colors = {"a": RED, "b": GREEN, "c": BLUE}
        def make_phase_curve(phase_shift, col):
            return left_panel.plot(
                lambda tau: V_AMPLITUDE * np.cos(OMEGA * tau + PHASE_OFFSET + phase_shift),
                x_range=[0, DURATION], color=col
            )
        curve_a = always_redraw(lambda: make_phase_curve(0, colors["a"]))
        curve_b = always_redraw(lambda: make_phase_curve(-2*np.pi/3, colors["b"]))
        curve_c = always_redraw(lambda: make_phase_curve(+2*np.pi/3, colors["c"]))
        self.play(Create(curve_a), Create(curve_b), Create(curve_c))

        # Moving dots on each waveform at current t
        dot_a = always_redraw(lambda: Dot(color=colors["a"]).move_to(left_panel.c2p(t_tracker.get_value(), V_AMPLITUDE*np.cos(OMEGA*t_tracker.get_value()+PHASE_OFFSET))))
        dot_b = always_redraw(lambda: Dot(color=colors["b"]).move_to(left_panel.c2p(t_tracker.get_value(), V_AMPLITUDE*np.cos(OMEGA*t_tracker.get_value()+PHASE_OFFSET-2*np.pi/3))))
        dot_c = always_redraw(lambda: Dot(color=colors["c"]).move_to(left_panel.c2p(t_tracker.get_value(), V_AMPLITUDE*np.cos(OMEGA*t_tracker.get_value()+PHASE_OFFSET+2*np.pi/3))))
        self.play(FadeIn(dot_a, scale=0.8), FadeIn(dot_b, scale=0.8), FadeIn(dot_c, scale=0.8))

        # --- αβ space vector (right) ---
        def alpha_beta_point():
            abc = abc_waveforms(t_tracker.get_value())
            alpha, beta, zero = (clarke_matrix() @ abc)
            return np.array([alpha, beta, 0])

        space_vec = always_redraw(lambda: Arrow(
            start=right_plane.c2p(0, 0), end=right_plane.c2p(*alpha_beta_point()[:2]),
            buff=0, stroke_width=6, max_tip_length_to_length_ratio=0.06, color=YELLOW
        ))
        self.play(GrowArrow(space_vec))

        # dq rotating axes anchored at origin
        d_axis = always_redraw(lambda: Arrow(
            right_plane.c2p(0, 0), right_plane.c2p(1.3*np.cos(OMEGA*t_tracker.get_value()+PHASE_OFFSET), 1.3*np.sin(OMEGA*t_tracker.get_value()+PHASE_OFFSET)),
            color=TEAL_A, buff=0, stroke_width=5
        ))
        q_axis = always_redraw(lambda: Arrow(
            right_plane.c2p(0, 0), right_plane.c2p(-1.3*np.sin(OMEGA*t_tracker.get_value()+PHASE_OFFSET), 1.3*np.cos(OMEGA*t_tracker.get_value()+PHASE_OFFSET)),
            color=PURPLE_A, buff=0, stroke_width=5
        ))
        d_label = always_redraw(lambda: Text("d", font_size=28, color=TEAL_A).move_to(d_axis.get_end()+0.3*RIGHT))
        q_label = always_redraw(lambda: Text("q", font_size=28, color=PURPLE_A).move_to(q_axis.get_end()+0.3*UP))
        self.play(FadeIn(d_axis, q_axis, d_label, q_label))

        # Projection of space vector onto dq axes
        def dq_components():
            abc = abc_waveforms(t_tracker.get_value())
            dq0 = abc_to_dq0(abc, theta=OMEGA*t_tracker.get_value()+PHASE_OFFSET)
            return dq0  # [d, q, 0]

        # Projection lines should project onto fixed αβ axes, not rotating dq axes
        d_comp_line = always_redraw(lambda: DashedLine(
            start=space_vec.get_end(),
            end=right_plane.c2p(alpha_beta_point()[0], 0),  # Project onto α-axis (horizontal)
            color=TEAL_A
        ))
        q_comp_line = always_redraw(lambda: DashedLine(
            start=space_vec.get_end(),
            end=right_plane.c2p(0, alpha_beta_point()[1]),  # Project onto β-axis (vertical)
            color=PURPLE_A
        ))
        self.play(FadeIn(d_comp_line), FadeIn(q_comp_line))

        # Small bars showing numerical d, q values
        d_bar_axis = NumberLine(x_range=[-1.3, 1.3, 0.5], length=5.5, include_numbers=True).next_to(right_plane, DOWN, buff=0.5)
        d_bar_title = Text("d, q components", font_size=24, color=GRAY_B).next_to(d_bar_axis, UP, buff=0.1)
        d_bar_tracker = ValueTracker(0.0)
        q_bar_tracker = ValueTracker(0.0)

        d_indicator = always_redraw(lambda: Triangle(color=TEAL_A, fill_opacity=1).scale(0.12).next_to(d_bar_axis.n2p(d_bar_tracker.get_value()), UP, buff=0))
        q_indicator = always_redraw(lambda: Triangle(color=PURPLE_A, fill_opacity=1).scale(0.12).next_to(d_bar_axis.n2p(q_bar_tracker.get_value()), DOWN, buff=0).rotate(np.pi))

        self.play(FadeIn(d_bar_axis), FadeIn(d_bar_title))
        self.play(FadeIn(d_indicator), FadeIn(q_indicator))

        # --- Equations panel (using MathTex for proper mathematical notation) ---
        # Clarke
        eq1_left  = MathTex(r"[\alpha, \beta, 0]^T", font_size=28)
        eq1_eq    = MathTex("=", font_size=28)
        eq1_mat   = MathTex(r"\begin{bmatrix} \frac{2}{3} & -\frac{1}{3} & -\frac{1}{3} \\ 0 & \frac{\sqrt{3}}{3} & -\frac{\sqrt{3}}{3} \\ \frac{1}{3} & \frac{1}{3} & \frac{1}{3} \end{bmatrix}", font_size=20)
        eq1_right = MathTex(r"[a, b, c]^T", font_size=28)
        eq1 = VGroup(eq1_left, eq1_eq, eq1_mat, eq1_right).arrange(RIGHT, buff=0.2)

        # Park
        eq2_left  = MathTex(r"[d, q, 0]^T", font_size=28)
        eq2_eq    = MathTex("=", font_size=28)
        eq2_mat   = MathTex(r"\begin{bmatrix} \cos\theta & \sin\theta & 0 \\ -\sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{bmatrix}", font_size=20)
        eq2_right = MathTex(r"[\alpha, \beta, 0]^T", font_size=28)
        eq2 = VGroup(eq2_left, eq2_eq, eq2_mat, eq2_right).arrange(RIGHT, buff=0.2)

        eq_group = VGroup(eq1, eq2).arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        eq_panel = SurroundingRectangle(eq_group, color=GRAY_C, corner_radius=0.15, fill_opacity=0.05)
        eq_block = VGroup(eq_panel, eq_group).scale(0.8).to_corner(UR).shift(0.2*LEFT + 0.1*DOWN)
        self.play(FadeIn(eq_block))

        # Note for theta
        theta_note = VGroup(
            MathTex(r"\theta = \omega_e t + \theta_0", font_size=24, color=YELLOW),
            Text("(electrical angle of rotating dq frame)", font_size=22, color=GRAY_B)
        ).arrange(DOWN, buff=0.08).next_to(eq_block, DOWN, buff=0.3).align_to(eq_block, RIGHT)
        self.play(FadeIn(theta_note, shift=UP))

        # --- Update function tying everything together ---
        def updater(m, dt):
            t = t_tracker.get_value() + dt
            t_tracker.set_value(t)
            d, q, _ = dq_components()
            d_bar_tracker.set_value(np.clip(d, -1.3, 1.3))
            q_bar_tracker.set_value(np.clip(q, -1.3, 1.3))
        ticker = always_redraw(lambda: Dot().set_opacity(0))
        ticker.add_updater(updater)
        self.add(ticker)

        # Axis labels
        alpha_label = MathTex(r"\alpha", font_size=28, color=GRAY_B).next_to(right_plane.x_axis, RIGHT, buff=0.1)
        beta_label = MathTex(r"\beta", font_size=28, color=GRAY_B).next_to(right_plane.y_axis, UP, buff=0.1)
        self.play(FadeIn(alpha_label), FadeIn(beta_label))

        # Optional zero sequence monitor
        if SHOW_ZERO_AXIS:
            zero_axis = NumberLine(x_range=[-1.0, 1.0, 0.5], length=4.2, include_numbers=True)
            zero_axis.next_to(d_bar_axis, DOWN, buff=0.7)
            zero_title = Text("0-seq (should be ≈0 for balanced)", font_size=22, color=GRAY_B).next_to(zero_axis, UP, buff=0.1)
            zero_indicator = always_redraw(lambda: Triangle(color=YELLOW, fill_opacity=1).scale(0.1).next_to(zero_axis.n2p((clarke_matrix() @ abc_waveforms(t_tracker.get_value()))[2]), UP, buff=0))
            self.play(FadeIn(zero_title), FadeIn(zero_axis), FadeIn(zero_indicator))

        self.wait(DURATION)
        ticker.clear_updaters()

        # Hide the old equation panel, theta note, title, and axis labels to make room for the direct formula
        self.play(FadeOut(eq_block), FadeOut(theta_note), FadeOut(right_label), FadeOut(alpha_label), FadeOut(beta_label))

        # Direct abc→dq0 (shown as plain text)
        eq_direct = MathTex(
            r"[d, q, 0]^T = \frac{2}{3} \begin{bmatrix} \cos\theta & \cos(\theta-\frac{2\pi}{3}) & \cos(\theta+\frac{2\pi}{3}) \\ -\sin\theta & -\sin(\theta-\frac{2\pi}{3}) & -\sin(\theta+\frac{2\pi}{3}) \\ \frac{1}{2} & \frac{1}{2} & \frac{1}{2} \end{bmatrix} [a, b, c]^T",
            font_size=18
        )
        eq_direct.to_corner(UR).shift(0.2*LEFT + 0.1*DOWN)
        self.play(Write(eq_direct))

        self.wait(2)
        self.play(*map(FadeOut, self.mobjects))
        outro = Text("Park transform → DC on d-axis for balanced three-phase.", font_size=34)
        self.play(FadeIn(outro, shift=UP))
        self.wait(2)
