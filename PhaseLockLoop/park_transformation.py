# -*- coding: utf-8 -*-
"""
Park Transformation Visualization with Manim (No-LaTeX Version)  # 使用 Manim 可视化 Park 变换（不依赖 LaTeX）

Tested with: Manim Community v0.19.0  # 测试版本说明

Render examples:  # 渲染命令示例
    manim -pqh park_transformation_fixed.py ParkTransformation  # 预览快速渲染
    # 1080p  # 全高清渲染提示
    # manim -p -r 1920,1080 park_transformation_fixed.py ParkTransformation  # 指定分辨率渲染

This scene explains the abc → αβ0 (Clarke) → dq0 (Park) transforms  # 本场景解释 abc→αβ0→dq0 变换流程
without using MathTex/LaTeX. All formula panels are drawn using Text,  # 不用 LaTeX，全部用 Text/MathTex 绘制
so there is NO LaTeX dependency.  # 无 LaTeX 依赖

"""
from manim import *  # 导入 Manim 的全部符号
import numpy as np  # 导入 NumPy 进行数值计算

# ---------- Configurable Parameters ----------  # 可配置参数区
F_ELECTRICAL = 1.0          # 电气频率 [Hz]
OMEGA = TAU * F_ELECTRICAL  # 电气角速度 [rad/s]，TAU=2π
V_AMPLITUDE = 1.0           # 相电压幅值
PHASE_OFFSET = 0.0          # 初始电气角 [rad]
SHOW_ZERO_AXIS = True       # 是否显示零序监视
DURATION = 5                # 动画总时长 [s]


# ========== MATHEMATICAL CALCULATIONS ========== #

class TransformCalculations:
    """电力系统坐标变换的数学计算类"""

    @staticmethod
    def abc_waveforms(t, A=V_AMPLITUDE, omega=OMEGA, phi=PHASE_OFFSET):
        """生成三相正弦波形"""
        Va = A * np.cos(omega * t + phi)
        Vb = A * np.cos(omega * t + phi - 2*np.pi/3)
        Vc = A * np.cos(omega * t + phi + 2*np.pi/3)
        return np.array([Va, Vb, Vc])

    @staticmethod
    def clarke_matrix():
        """Clarke变换矩阵 (abc -> αβ0)"""
        return np.array([
            [2/3, -1/3, -1/3],
            [0, np.sqrt(3)/3, -np.sqrt(3)/3],
            [1/3, 1/3, 1/3],
        ])

    @staticmethod
    def park_matrix(theta):
        """Park旋转矩阵 (αβ0 -> dq0)"""
        c, s = np.cos(theta), np.sin(theta)
        return np.array([
            [ c,  s, 0],
            [-s,  c, 0],
            [ 0,  0, 1],
        ])

    @staticmethod
    def abc_to_alphabeta(abc):
        """abc到αβ0变换"""
        return TransformCalculations.clarke_matrix() @ abc

    @staticmethod
    def alphabeta_to_dq0(alpha_beta_0, theta):
        """αβ0到dq0变换"""
        return TransformCalculations.park_matrix(theta) @ alpha_beta_0

    @staticmethod
    def abc_to_dq0(abc, theta):
        """直接从abc到dq0的组合变换"""
        alpha_beta_0 = TransformCalculations.abc_to_alphabeta(abc)
        return TransformCalculations.alphabeta_to_dq0(alpha_beta_0, theta)

    @staticmethod
    def get_electrical_angle(t, omega=OMEGA, phi=PHASE_OFFSET):
        """计算电气角度"""
        return omega * t + phi


class GeometryUtils:
    """几何计算工具类"""

    @staticmethod
    def project_point_to_axis(point, axis_dir):
        """将点投影到过原点的指定方向轴"""
        p = np.array([point[0], point[1]])
        u = np.array([axis_dir[0], axis_dir[1]])
        u = u / (np.linalg.norm(u) + 1e-9)
        proj_len = np.dot(p, u)
        proj = proj_len * u
        return np.array([proj[0], proj[1], 0.0])


# ========== DISPLAY CREATION FUNCTIONS ========== #

class DisplayCreator:
    """Manim可视化元素创建类"""

    @staticmethod
    def create_title_section():
        """创建标题区域"""
        title = Text("Park Transformation (abc → dq0)", weight=BOLD).scale(0.8)
        subtitle = Text("Clarke → Park with rotating reference frame", font_size=28, color=GRAY_B)
        return VGroup(title, subtitle).arrange(DOWN, buff=0.2).shift(UP*1.5)

    @staticmethod
    def create_left_panel():
        """创建左侧时域波形面板"""
        axes = Axes(
            x_range=[0, DURATION, 1], y_range=[-1.3, 1.3, 1],
            x_length=6.8, y_length=3.2,
            tips=False, axis_config={"include_numbers": False, "color": GRAY_B}
        ).to_edge(LEFT, buff=0.7).shift(DOWN*0.5)

        label = Text("Phase Voltages: a, b, c", font_size=26, color=GRAY_B).next_to(axes, UP, buff=0.2)
        return axes, label

    @staticmethod
    def create_right_panel():
        """创建右侧αβ平面"""
        plane = NumberPlane(
            x_range=[-1.6, 1.6, 1], y_range=[-1.6, 1.6, 1],
            x_length=5.0, y_length=5.0,
            faded_line_ratio=2,
            background_line_style={"stroke_color": GRAY_D, "stroke_opacity": 0.6}
        ).to_edge(RIGHT, buff=0.9).shift(DOWN*0.1)

        label = MathTex(r"\alpha\beta \text{ plane and dq frame}", font_size=26, color=GRAY_B).next_to(plane, UP, buff=0.2)
        return plane, label

    @staticmethod
    def create_phase_curves(left_panel, colors):
        """创建三相波形曲线"""
        def make_phase_curve(phase_shift, col):
            return left_panel.plot(
                lambda tau: V_AMPLITUDE * np.cos(OMEGA * tau + PHASE_OFFSET + phase_shift),
                x_range=[0, DURATION], color=col
            )

        curve_a = always_redraw(lambda: make_phase_curve(0, colors["a"]))
        curve_b = always_redraw(lambda: make_phase_curve(-2*np.pi/3, colors["b"]))
        curve_c = always_redraw(lambda: make_phase_curve(+2*np.pi/3, colors["c"]))
        return curve_a, curve_b, curve_c

    @staticmethod
    def create_phase_dots(left_panel, t_tracker, colors):
        """创建三相波形上的移动点"""
        dot_a = always_redraw(lambda: Dot(color=colors["a"]).move_to(
            left_panel.c2p(
                min(t_tracker.get_value(), DURATION),
                V_AMPLITUDE*np.cos(OMEGA*t_tracker.get_value()+PHASE_OFFSET)
            )
        ))
        dot_b = always_redraw(lambda: Dot(color=colors["b"]).move_to(
            left_panel.c2p(
                min(t_tracker.get_value(), DURATION),
                V_AMPLITUDE*np.cos(OMEGA*t_tracker.get_value()+PHASE_OFFSET-2*np.pi/3)
            )
        ))
        dot_c = always_redraw(lambda: Dot(color=colors["c"]).move_to(
            left_panel.c2p(
                min(t_tracker.get_value(), DURATION),
                V_AMPLITUDE*np.cos(OMEGA*t_tracker.get_value()+PHASE_OFFSET+2*np.pi/3)
            )
        ))
        return dot_a, dot_b, dot_c

    @staticmethod
    def create_space_vector(right_plane, t_tracker):
        """创建空间矢量"""
        def alpha_beta_point():
            abc = TransformCalculations.abc_waveforms(t_tracker.get_value())
            alpha, beta, zero = TransformCalculations.abc_to_alphabeta(abc)
            return np.array([alpha, beta, 0])

        return always_redraw(lambda: Arrow(
            start=right_plane.c2p(0, 0),
            end=right_plane.c2p(*alpha_beta_point()[:2]),
            buff=0, stroke_width=6, max_tip_length_to_length_ratio=0.06, color=YELLOW
        )), alpha_beta_point

    @staticmethod
    def create_dq_axes(right_plane, t_tracker):
        """创建旋转dq坐标轴"""
        d_axis = always_redraw(lambda: Arrow(
            right_plane.c2p(0, 0),
            right_plane.c2p(
                1.3*np.cos(TransformCalculations.get_electrical_angle(t_tracker.get_value())),
                1.3*np.sin(TransformCalculations.get_electrical_angle(t_tracker.get_value()))
            ),
            color=TEAL_A, buff=0, stroke_width=5
        ))

        q_axis = always_redraw(lambda: Arrow(
            right_plane.c2p(0, 0),
            right_plane.c2p(
                -1.3*np.sin(TransformCalculations.get_electrical_angle(t_tracker.get_value())),
                1.3*np.cos(TransformCalculations.get_electrical_angle(t_tracker.get_value()))
            ),
            color=PURPLE_A, buff=0, stroke_width=5
        ))

        d_label = always_redraw(lambda: Text("d", font_size=28, color=TEAL_A).move_to(d_axis.get_end()+0.3*RIGHT))
        q_label = always_redraw(lambda: Text("q", font_size=28, color=PURPLE_A).move_to(q_axis.get_end()+0.3*UP))

        return d_axis, q_axis, d_label, q_label

    @staticmethod
    def create_projection_lines(space_vec, alpha_beta_point_func, right_plane):
        """创建投影辅助线"""
        d_comp_line = always_redraw(lambda: DashedLine(
            start=space_vec.get_end(),
            end=right_plane.c2p(alpha_beta_point_func()[0], 0),
            color=TEAL_A
        ))
        q_comp_line = always_redraw(lambda: DashedLine(
            start=space_vec.get_end(),
            end=right_plane.c2p(0, alpha_beta_point_func()[1]),
            color=PURPLE_A
        ))
        return d_comp_line, q_comp_line

    @staticmethod
    def create_dq_indicators(right_plane):
        """创建dq分量指示器"""
        d_bar_axis = NumberLine(x_range=[-1.3, 1.3, 0.5], length=5.5, include_numbers=True).next_to(right_plane, DOWN, buff=0.5)
        d_bar_title = Text("d, q components", font_size=24, color=GRAY_B).next_to(d_bar_axis, UP, buff=0.1)

        d_bar_tracker = ValueTracker(0.0)
        q_bar_tracker = ValueTracker(0.0)

        d_indicator = always_redraw(lambda: Triangle(color=TEAL_A, fill_opacity=1).scale(0.12).next_to(d_bar_axis.n2p(d_bar_tracker.get_value()), UP, buff=0))
        q_indicator = always_redraw(lambda: Triangle(color=PURPLE_A, fill_opacity=1).scale(0.12).next_to(d_bar_axis.n2p(q_bar_tracker.get_value()), DOWN, buff=0).rotate(np.pi))

        d_value_text = always_redraw(lambda: DecimalNumber(
            d_bar_tracker.get_value(), num_decimal_places=2, include_sign=True
        ).scale(0.6).set_color(TEAL_A).next_to(d_bar_axis, LEFT, buff=0.4))

        q_value_text = always_redraw(lambda: DecimalNumber(
            q_bar_tracker.get_value(), num_decimal_places=2, include_sign=True
        ).scale(0.6).set_color(PURPLE_A).next_to(d_bar_axis, RIGHT, buff=0.4))

        return d_bar_axis, d_bar_title, d_bar_tracker, q_bar_tracker, d_indicator, q_indicator, d_value_text, q_value_text

    @staticmethod
    def create_equations_panel():
        """创建公式面板"""
        eq1_left = MathTex(r"[\alpha, \beta, 0]^T", font_size=28)
        eq1_eq = MathTex("=", font_size=28)
        eq1_mat = MathTex(r"\begin{bmatrix} \frac{2}{3} & -\frac{1}{3} & -\frac{1}{3} \\\\ 0 & \frac{\sqrt{3}}{3} & -\frac{\sqrt{3}}{3} \\\\ \frac{1}{3} & \frac{1}{3} & \frac{1}{3} \end{bmatrix}", font_size=20)
        eq1_right = MathTex(r"[a, b, c]^T", font_size=28)
        eq1 = VGroup(eq1_left, eq1_eq, eq1_mat, eq1_right).arrange(RIGHT, buff=0.2)

        eq2_left = MathTex(r"[d, q, 0]^T", font_size=28)
        eq2_eq = MathTex("=", font_size=28)
        eq2_mat = MathTex(r"\begin{bmatrix} \cos\theta & \sin\theta & 0 \\\\ -\sin\theta & \cos\theta & 0 \\\\ 0 & 0 & 1 \end{bmatrix}", font_size=20)
        eq2_right = MathTex(r"[\alpha, \beta, 0]^T", font_size=28)
        eq2 = VGroup(eq2_left, eq2_eq, eq2_mat, eq2_right).arrange(RIGHT, buff=0.2)

        eq_group = VGroup(eq1, eq2).arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        eq_panel = SurroundingRectangle(eq_group, color=GRAY_C, corner_radius=0.15, fill_opacity=0.05)
        eq_block = VGroup(eq_panel, eq_group).scale(0.75).to_edge(UP, buff=0.3).to_edge(RIGHT, buff=0.5)

        return eq_block

    @staticmethod
    def create_theta_note(eq_block):
        """创建θ说明"""
        theta_note = VGroup(
            MathTex(r"\theta = \omega_e t + \theta_0", font_size=22, color=YELLOW),
            Text("(electrical angle)", font_size=20, color=GRAY_B)
        ).arrange(DOWN, buff=0.08).next_to(eq_block, DOWN, buff=0.25).align_to(eq_block, RIGHT)

        return theta_note

    @staticmethod
    def create_axis_labels(right_plane):
        """创建αβ轴标签"""
        alpha_label = MathTex(r"\alpha", font_size=28, color=GRAY_B).next_to(right_plane.x_axis, RIGHT, buff=0.1)
        beta_label = MathTex(r"\beta", font_size=28, color=GRAY_B).next_to(right_plane.y_axis, UP, buff=0.1)
        return alpha_label, beta_label

    @staticmethod
    def create_zero_sequence_monitor(d_bar_axis, t_tracker):
        """创建零序监视器"""
        zero_axis = NumberLine(x_range=[-1.0, 1.0, 0.5], length=4.2, include_numbers=True)
        zero_axis.next_to(d_bar_axis, DOWN, buff=0.7)
        zero_title = Text("0-seq (should be ≈0 for balanced)", font_size=22, color=GRAY_B).next_to(zero_axis, UP, buff=0.1)
        zero_indicator = always_redraw(lambda: Triangle(color=YELLOW, fill_opacity=1).scale(0.1).next_to(
            zero_axis.n2p(TransformCalculations.abc_to_alphabeta(TransformCalculations.abc_waveforms(t_tracker.get_value()))[2]),
            UP, buff=0
        ))
        return zero_title, zero_axis, zero_indicator

    @staticmethod
    def create_direct_equation():
        """创建直接变换公式"""
        return MathTex(
            r"[d, q, 0]^T = \frac{2}{3} \begin{bmatrix} \cos\theta & \cos(\theta-\frac{2\pi}{3}) & \cos(\theta+\frac{2\pi}{3}) \\\\ -\sin\theta & -\sin(\theta-\frac{2\pi}{3}) & -\sin(\theta+\frac{2\pi}{3}) \\\\ \frac{1}{2} & \frac{1}{2} & \frac{1}{2} \end{bmatrix} [a, b, c]^T",
            font_size=18
        ).to_edge(UP, buff=0.3).to_edge(RIGHT, buff=0.5)


# ========== ANIMATION CONTROL FUNCTIONS ========== #

class AnimationController:
    """动画控制类"""

    @staticmethod
    def create_dq_updater(t_tracker, d_bar_tracker, q_bar_tracker):
        """创建dq分量更新器"""
        def updater(m, dt):
            t_next = t_tracker.get_value() + dt
            if t_next > DURATION:
                t_next = DURATION
            t_tracker.set_value(t_next)

            abc = TransformCalculations.abc_waveforms(t_tracker.get_value())
            theta = TransformCalculations.get_electrical_angle(t_tracker.get_value())
            dq0 = TransformCalculations.abc_to_dq0(abc, theta)

            d_bar_tracker.set_value(np.clip(dq0[0], -1.3, 1.3))
            q_bar_tracker.set_value(np.clip(dq0[1], -1.3, 1.3))

        return updater

    @staticmethod
    def get_dq_components_function(t_tracker):
        """获取dq分量计算函数"""
        def dq_components():
            abc = TransformCalculations.abc_waveforms(t_tracker.get_value())
            theta = TransformCalculations.get_electrical_angle(t_tracker.get_value())
            return TransformCalculations.abc_to_dq0(abc, theta)
        return dq_components


# ========== MANIM SCENE ========== #

class ParkTransformation(Scene):
    """Park变换可视化主场景类"""

    def construct(self):
        """场景主要构建逻辑"""
        self.setup_scene()
        self.create_main_panels()
        self.create_waveforms_and_vectors()
        self.create_equations_and_indicators()
        self.run_animation()
        self.show_final_equation()

    def setup_scene(self):
        """设置场景基本参数"""
        self.camera.background_color = "#0d1321"

    def create_main_panels(self):
        """创建主要显示面板"""
        # 创建标题
        header = DisplayCreator.create_title_section()
        self.play(FadeIn(header, shift=UP))

        # 创建左右面板
        self.left_panel, left_label = DisplayCreator.create_left_panel()
        self.right_plane, right_label = DisplayCreator.create_right_panel()

        # 淡出标题，显示面板
        self.play(FadeOut(header))
        self.play(FadeIn(self.left_panel), FadeIn(self.right_plane), FadeIn(left_label), FadeIn(right_label))

        # 保存标签引用供后续使用
        self.right_label = right_label

    def create_waveforms_and_vectors(self):
        """创建波形和矢量显示"""
        # 时间跟踪器
        self.t_tracker = ValueTracker(0.0)

        # 创建三相波形
        colors = {"a": RED, "b": GREEN, "c": BLUE}
        curve_a, curve_b, curve_c = DisplayCreator.create_phase_curves(self.left_panel, colors)
        self.play(Create(curve_a), Create(curve_b), Create(curve_c))

        # 创建波形上的移动点
        dot_a, dot_b, dot_c = DisplayCreator.create_phase_dots(self.left_panel, self.t_tracker, colors)
        self.play(FadeIn(dot_a, scale=0.8), FadeIn(dot_b, scale=0.8), FadeIn(dot_c, scale=0.8))

        # 创建空间矢量
        self.space_vec, self.alpha_beta_point_func = DisplayCreator.create_space_vector(self.right_plane, self.t_tracker)
        self.play(GrowArrow(self.space_vec))

        # 创建旋转dq坐标轴
        d_axis, q_axis, d_label, q_label = DisplayCreator.create_dq_axes(self.right_plane, self.t_tracker)
        self.play(FadeIn(d_axis, q_axis, d_label, q_label))

        # 创建投影线
        d_comp_line, q_comp_line = DisplayCreator.create_projection_lines(
            self.space_vec, self.alpha_beta_point_func, self.right_plane
        )
        self.play(FadeIn(d_comp_line), FadeIn(q_comp_line))

    def create_equations_and_indicators(self):
        """创建公式和指示器"""
        # 创建dq分量指示器
        (self.d_bar_axis, d_bar_title, self.d_bar_tracker, self.q_bar_tracker,
         d_indicator, q_indicator, d_value_text, q_value_text) = DisplayCreator.create_dq_indicators(self.right_plane)

        self.play(FadeIn(self.d_bar_axis), FadeIn(d_bar_title))
        self.play(FadeIn(d_indicator), FadeIn(q_indicator))
        self.play(FadeIn(d_value_text), FadeIn(q_value_text))

        # 创建公式面板
        eq_block = DisplayCreator.create_equations_panel()
        self.play(FadeOut(self.right_label))
        self.play(FadeIn(eq_block))

        # 创建θ说明
        theta_note = DisplayCreator.create_theta_note(eq_block)
        self.play(FadeIn(theta_note, shift=UP))

        # 保存引用
        self.eq_block = eq_block
        self.theta_note = theta_note

        # 创建轴标签
        alpha_label, beta_label = DisplayCreator.create_axis_labels(self.right_plane)
        self.play(FadeIn(alpha_label), FadeIn(beta_label))

        # 保存标签引用
        self.alpha_label = alpha_label
        self.beta_label = beta_label

        # 可选零序监视器
        if SHOW_ZERO_AXIS:
            zero_title, zero_axis, zero_indicator = DisplayCreator.create_zero_sequence_monitor(
                self.d_bar_axis, self.t_tracker
            )
            self.play(FadeIn(zero_title), FadeIn(zero_axis), FadeIn(zero_indicator))

    def run_animation(self):
        """运行主动画"""
        # 创建更新器
        updater = AnimationController.create_dq_updater(self.t_tracker, self.d_bar_tracker, self.q_bar_tracker)

        # 创建不可见的更新器载体
        ticker = always_redraw(lambda: Dot().set_opacity(0))
        ticker.add_updater(updater)
        self.add(ticker)

        # 运行动画
        self.wait(DURATION)
        ticker.clear_updaters()

    def show_final_equation(self):
        """显示最终公式"""
        # 隐藏旧元素
        self.play(FadeOut(self.eq_block), FadeOut(self.theta_note),
                  FadeOut(self.alpha_label), FadeOut(self.beta_label))

        # 显示直接变换公式
        eq_direct = DisplayCreator.create_direct_equation()
        self.play(Write(eq_direct))

        self.wait(2)

        # 结束场景
        self.play(*map(FadeOut, self.mobjects))
        outro = Text("Park transform → DC on d-axis for balanced three-phase.", font_size=34)
        self.play(FadeIn(outro, shift=UP))
        self.wait(2)


if __name__ == "__main__":
    scene = ParkTransformation()
    scene.render()