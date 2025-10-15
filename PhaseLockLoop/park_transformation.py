# -*- coding: utf-8 -*-  # 指定源码文件的编码为 UTF-8
"""
Park Transformation Visualization with Manim (No-LaTeX Version)  # 使用 Manim 可视化 Park 变换（不依赖 LaTeX）

Tested with: Manim Community v0.19.0  # 测试版本说明

Render examples:  # 渲染命令示例
    manim -pqh park_transformation.py ParkTransformation  # 预览快速渲染
    # 1080p  # 全高清渲染提示
    # manim -p -r 1920,1080 park_transformation.py ParkTransformation  # 指定分辨率渲染

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

# ---------- Math Helpers ----------  # 数学辅助函数

def abc_waveforms(t, A=V_AMPLITUDE, omega=OMEGA, phi=PHASE_OFFSET):  # 生成三相正弦波形
    Va = A * np.cos(omega * t + phi)  # a 相瞬时值
    Vb = A * np.cos(omega * t + phi - 2*np.pi/3)  # b 相，滞后 120°
    Vc = A * np.cos(omega * t + phi + 2*np.pi/3)  # c 相，超前 120°
    return np.array([Va, Vb, Vc])  # 返回列向量 [Va, Vb, Vc]


def clarke_matrix():  # 构造 Clarke 变换矩阵（功率不变型 2/3 缩放）
    """abc -> αβ0 matrix (power-invariant variant with 2/3 scaling)."""
    return np.array([
        [2/3, -1/3, -1/3],  # α 行：2/3*[1, -1/2, -1/2]
        [0,   np.sqrt(3)/3, -np.sqrt(3)/3],  # β 行：等效 2/3*[0, √3/2, -√3/2]
        [1/3, 1/3, 1/3],  # 0 行：零序分量平均值
    ])


def park_matrix(theta):  # 构造 Park 旋转矩阵（αβ0→dq0）
    """αβ0 -> dq0 rotation (d along cosθ, q along sinθ)."""
    c, s = np.cos(theta), np.sin(theta)  # 计算 cosθ 与 sinθ
    R = np.array([
        [ c,  s, 0],  # d 轴方向（沿 cosθ）
        [-s,  c, 0],  # q 轴方向（沿 sinθ）
        [ 0,  0, 1],  # 0 序不变
    ])
    return R  # 返回旋转矩阵


def abc_to_dq0(abc, theta):  # 组合 Clarke 与 Park，得到 dq0
    T_clarke = clarke_matrix()  # Clarke 矩阵
    alpha_beta_0 = T_clarke @ abc  # abc→αβ0
    T_park = park_matrix(theta)  # Park 矩阵
    dq0 = T_park @ alpha_beta_0  # αβ0→dq0
    return dq0  # 返回 [d, q, 0]


# ---------- Utility: projection helper ----------  # 投影辅助（当前未直接使用）
def project_point_to_axis(point, axis_dir):  # 将点投影到过原点的指定方向轴
    """Project a point P (in scene coords) onto a directed axis through origin."""
    p = np.array([point[0], point[1]])  # 取 xy 分量
    u = np.array([axis_dir[0], axis_dir[1]])  # 方向向量 xy 分量
    u = u / (np.linalg.norm(u) + 1e-9)  # 归一化，避免除零
    proj_len = np.dot(p, u)  # 投影长度
    proj = proj_len * u  # 投影点 xy
    return np.array([proj[0], proj[1], 0.0])  # 返回场景坐标系点


# ---------- Manim Scene ----------  # Manim 场景类
class ParkTransformation(Scene):  # 定义场景
    def construct(self):  # 场景主逻辑
        self.camera.background_color = "#0d1321"  # 设置背景颜色
        title = Text("Park Transformation (abc → dq0)", weight=BOLD).scale(0.8)  # 标题
        subtitle = Text("Clarke → Park with rotating reference frame", font_size=28, color=GRAY_B)  # 副标题
        header = VGroup(title, subtitle).arrange(DOWN, buff=0.2).to_edge(UP)  # 顶部排列
        self.play(FadeIn(header, shift=UP))  # 标题入场

        # Left: time-domain phase waveforms; Right: space vector & rotating axes  # 左侧时域曲线，右侧空间矢量与旋转坐标
        left_panel = Axes(  # 左侧坐标轴（时域）
            x_range=[0, DURATION, 1], y_range=[-1.3, 1.3, 1],  # x 时间范围，y 电压范围
            x_length=6.8, y_length=3.2,  # 尺寸
            tips=False, axis_config={"include_numbers": False, "color": GRAY_B}  # 不显示刻度数字
        )
        left_panel.to_edge(LEFT, buff=0.7).shift(DOWN*0.5)  # 放置到左侧
        left_label = Text("Phase Voltages: a, b, c", font_size=26, color=GRAY_B).next_to(left_panel, UP, buff=0.2)  # 左侧标题

        right_plane = NumberPlane(  # 右侧 αβ 平面
            x_range=[-1.6, 1.6, 1], y_range=[-1.6, 1.6, 1],  # 坐标范围
            x_length=5.0, y_length=5.0,  # 尺寸
            faded_line_ratio=2,  # 背景线淡化
            background_line_style={"stroke_color": GRAY_D, "stroke_opacity": 0.6}  # 网格样式
        ).to_edge(RIGHT, buff=0.9).shift(DOWN*0.1)  # 放置到右侧
        right_label = MathTex(r"\alpha\beta \text{ plane and dq frame}", font_size=26, color=GRAY_B).next_to(right_plane, UP, buff=0.2)  # 右侧说明

        # fade out the header  # 淡出页眉
        self.play(FadeOut(header))
        self.play(FadeIn(left_panel), FadeIn(right_plane), FadeIn(left_label), FadeIn(right_label))  # 面板入场

        # Time tracker (drives everything)  # 驱动时间的变量
        t_tracker = ValueTracker(0.0)

        # --- Phase signals (left) ---  # 左侧三相信号
        colors = {"a": RED, "b": GREEN, "c": BLUE}  # 颜色映射
        def make_phase_curve(phase_shift, col):  # 绘制某相的曲线
            return left_panel.plot(
                lambda tau: V_AMPLITUDE * np.cos(OMEGA * tau + PHASE_OFFSET + phase_shift),  # 余弦波
                x_range=[0, DURATION], color=col  # 范围与颜色
            )
        curve_a = always_redraw(lambda: make_phase_curve(0, colors["a"]))  # a 相曲线（动态重绘）
        curve_b = always_redraw(lambda: make_phase_curve(-2*np.pi/3, colors["b"]))  # b 相曲线
        curve_c = always_redraw(lambda: make_phase_curve(+2*np.pi/3, colors["c"]))  # c 相曲线
        self.play(Create(curve_a), Create(curve_b), Create(curve_c))  # 创建三条曲线

        # Moving dots on each waveform at current t  # 每条曲线上移动的点
        dot_a = always_redraw(lambda: Dot(color=colors["a"]).move_to(  # a 相点
            left_panel.c2p(
                min(t_tracker.get_value(), DURATION),  # 限制在坐标范围内
                V_AMPLITUDE*np.cos(OMEGA*t_tracker.get_value()+PHASE_OFFSET)  # 当前时刻值
            )
        ))
        dot_b = always_redraw(lambda: Dot(color=colors["b"]).move_to(  # b 相点
            left_panel.c2p(
                min(t_tracker.get_value(), DURATION),
                V_AMPLITUDE*np.cos(OMEGA*t_tracker.get_value()+PHASE_OFFSET-2*np.pi/3)
            )
        ))
        dot_c = always_redraw(lambda: Dot(color=colors["c"]).move_to(  # c 相点
            left_panel.c2p(
                min(t_tracker.get_value(), DURATION),
                V_AMPLITUDE*np.cos(OMEGA*t_tracker.get_value()+PHASE_OFFSET+2*np.pi/3)
            )
        ))
        self.play(FadeIn(dot_a, scale=0.8), FadeIn(dot_b, scale=0.8), FadeIn(dot_c, scale=0.8))  # 点入场

        # --- αβ space vector (right) ---  # 右侧空间矢量
        def alpha_beta_point():  # 计算 αβ 点
            abc = abc_waveforms(t_tracker.get_value())  # 取当前三相值
            alpha, beta, zero = (clarke_matrix() @ abc)  # Clarke 变换
            return np.array([alpha, beta, 0])  # 返回 αβ 坐标

        space_vec = always_redraw(lambda: Arrow(  # 空间矢量箭头
            start=right_plane.c2p(0, 0), end=right_plane.c2p(*alpha_beta_point()[:2]),  # 原点到 αβ
            buff=0, stroke_width=6, max_tip_length_to_length_ratio=0.06, color=YELLOW  # 样式
        ))
        self.play(GrowArrow(space_vec))  # 绘制箭头

        # dq rotating axes anchored at origin  # 旋转 dq 坐标轴
        d_axis = always_redraw(lambda: Arrow(  # d 轴箭头
            right_plane.c2p(0, 0), right_plane.c2p(1.3*np.cos(OMEGA*t_tracker.get_value()+PHASE_OFFSET), 1.3*np.sin(OMEGA*t_tracker.get_value()+PHASE_OFFSET)),
            color=TEAL_A, buff=0, stroke_width=5
        ))
        q_axis = always_redraw(lambda: Arrow(  # q 轴箭头（与 d 轴正交）
            right_plane.c2p(0, 0), right_plane.c2p(-1.3*np.sin(OMEGA*t_tracker.get_value()+PHASE_OFFSET), 1.3*np.cos(OMEGA*t_tracker.get_value()+PHASE_OFFSET)),
            color=PURPLE_A, buff=0, stroke_width=5
        ))
        d_label = always_redraw(lambda: Text("d", font_size=28, color=TEAL_A).move_to(d_axis.get_end()+0.3*RIGHT))  # d 轴标注
        q_label = always_redraw(lambda: Text("q", font_size=28, color=PURPLE_A).move_to(q_axis.get_end()+0.3*UP))  # q 轴标注
        self.play(FadeIn(d_axis, q_axis, d_label, q_label))  # 坐标轴入场

        # Projection of space vector onto dq axes  # 计算 dq 分量
        def dq_components():  # 返回 [d, q, 0]
            abc = abc_waveforms(t_tracker.get_value())  # 当前 abc
            dq0 = abc_to_dq0(abc, theta=OMEGA*t_tracker.get_value()+PHASE_OFFSET)  # 变换到 dq0
            return dq0  # [d, q, 0]

        # Projection lines should project onto fixed αβ axes, not rotating dq axes  # 在 αβ 上做投影辅助线
        d_comp_line = always_redraw(lambda: DashedLine(  # 到 α 轴的投影虚线
            start=space_vec.get_end(),
            end=right_plane.c2p(alpha_beta_point()[0], 0),  # 投到 α 轴（水平）
            color=TEAL_A
        ))
        q_comp_line = always_redraw(lambda: DashedLine(  # 到 β 轴的投影虚线
            start=space_vec.get_end(),
            end=right_plane.c2p(0, alpha_beta_point()[1]),  # 投到 β 轴（垂直）
            color=PURPLE_A
        ))
        self.play(FadeIn(d_comp_line), FadeIn(q_comp_line))  # 显示投影线

        # Small bars showing numerical d, q values  # 下方条形刻度显示 d/q 数值
        d_bar_axis = NumberLine(x_range=[-1.3, 1.3, 0.5], length=5.5, include_numbers=True).next_to(right_plane, DOWN, buff=0.5)  # 数轴
        d_bar_title = Text("d, q components", font_size=24, color=GRAY_B).next_to(d_bar_axis, UP, buff=0.1)  # 标题
        d_bar_tracker = ValueTracker(0.0)  # d 数值跟踪
        q_bar_tracker = ValueTracker(0.0)  # q 数值跟踪

        d_indicator = always_redraw(lambda: Triangle(color=TEAL_A, fill_opacity=1).scale(0.12).next_to(d_bar_axis.n2p(d_bar_tracker.get_value()), UP, buff=0))  # d 指示三角
        q_indicator = always_redraw(lambda: Triangle(color=PURPLE_A, fill_opacity=1).scale(0.12).next_to(d_bar_axis.n2p(q_bar_tracker.get_value()), DOWN, buff=0).rotate(np.pi))  # q 指示三角

        self.play(FadeIn(d_bar_axis), FadeIn(d_bar_title))  # 数轴与标题入场
        self.play(FadeIn(d_indicator), FadeIn(q_indicator))  # 指示器入场

        # Live numeric readouts for d and q  # 实时数值
        d_value_text = always_redraw(lambda: DecimalNumber(  # d 数字显示
            d_bar_tracker.get_value(), num_decimal_places=2, include_sign=True
        ).scale(0.6).set_color(TEAL_A).next_to(d_bar_axis, LEFT, buff=0.4))

        q_value_text = always_redraw(lambda: DecimalNumber(  # q 数字显示
            q_bar_tracker.get_value(), num_decimal_places=2, include_sign=True
        ).scale(0.6).set_color(PURPLE_A).next_to(d_bar_axis, RIGHT, buff=0.4))

        self.play(FadeIn(d_value_text), FadeIn(q_value_text))  # 数字显示入场

        # --- Equations panel (using MathTex for proper mathematical notation) ---  # 公式面板（MathTex）
        # Clarke  # Clarke 变换矩阵展示
        eq1_left  = MathTex(r"[\alpha, \beta, 0]^T", font_size=28)
        eq1_eq    = MathTex("=", font_size=28)
        eq1_mat   = MathTex(r"\begin{bmatrix} \frac{2}{3} & -\frac{1}{3} & -\frac{1}{3} \\ 0 & \frac{\sqrt{3}}{3} & -\frac{\sqrt{3}}{3} \\ \frac{1}{3} & \frac{1}{3} & \frac{1}{3} \end{bmatrix}", font_size=20)
        eq1_right = MathTex(r"[a, b, c]^T", font_size=28)
        eq1 = VGroup(eq1_left, eq1_eq, eq1_mat, eq1_right).arrange(RIGHT, buff=0.2)

        # Park  # Park 旋转矩阵展示
        eq2_left  = MathTex(r"[d, q, 0]^T", font_size=28)
        eq2_eq    = MathTex("=", font_size=28)
        eq2_mat   = MathTex(r"\begin{bmatrix} \cos\theta & \sin\theta & 0 \\ -\sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{bmatrix}", font_size=20)
        eq2_right = MathTex(r"[\alpha, \beta, 0]^T", font_size=28)
        eq2 = VGroup(eq2_left, eq2_eq, eq2_mat, eq2_right).arrange(RIGHT, buff=0.2)

        eq_group = VGroup(eq1, eq2).arrange(DOWN, aligned_edge=LEFT, buff=0.35)  # 堆叠两行公式
        eq_panel = SurroundingRectangle(eq_group, color=GRAY_C, corner_radius=0.15, fill_opacity=0.05)  # 外框
        eq_block = VGroup(eq_panel, eq_group).scale(0.8).to_corner(UR).shift(0.2*LEFT + 0.1*DOWN)  # 放置于右上角
        self.play(FadeIn(eq_block))  # 展示公式块

        # Note for theta  # θ 的说明
        theta_note = VGroup(
            MathTex(r"\theta = \omega_e t + \theta_0", font_size=24, color=YELLOW),  # 角度表达
            Text("(electrical angle of rotating dq frame)", font_size=22, color=GRAY_B)  # 文字注释
        ).arrange(DOWN, buff=0.08).next_to(eq_block, DOWN, buff=0.3).align_to(eq_block, RIGHT)  # 放置
        self.play(FadeIn(theta_note, shift=UP))  # 展示说明

        # --- Update function tying everything together ---  # 更新器：推进时间并更新数值
        def updater(m, dt):  # 每帧调用，dt 为帧间隔
            # Advance time but clamp within [0, DURATION] to avoid out-of-bounds on axes  # 推进时间并截断
            t_next = t_tracker.get_value() + dt  # 计算下一时刻
            if t_next > DURATION:  # 超出则截断
                t_next = DURATION
            t_tracker.set_value(t_next)  # 更新时间
            d, q, _ = dq_components()  # 计算 dq
            d_bar_tracker.set_value(np.clip(d, -1.3, 1.3))  # 限幅更新 d
            q_bar_tracker.set_value(np.clip(q, -1.3, 1.3))  # 限幅更新 q
        ticker = always_redraw(lambda: Dot().set_opacity(0))  # 不可见占位 mobject，用于挂载 updater
        ticker.add_updater(updater)  # 绑定更新器
        self.add(ticker)  # 加入场景

        # Axis labels  # αβ 轴标签
        alpha_label = MathTex(r"\alpha", font_size=28, color=GRAY_B).next_to(right_plane.x_axis, RIGHT, buff=0.1)  # α 标签
        beta_label = MathTex(r"\beta", font_size=28, color=GRAY_B).next_to(right_plane.y_axis, UP, buff=0.1)  # β 标签
        self.play(FadeIn(alpha_label), FadeIn(beta_label))  # 标签入场

        # Optional zero sequence monitor  # 可选零序监视条
        if SHOW_ZERO_AXIS:  # 控制是否显示
            zero_axis = NumberLine(x_range=[-1.0, 1.0, 0.5], length=4.2, include_numbers=True)  # 零序数轴
            zero_axis.next_to(d_bar_axis, DOWN, buff=0.7)  # 放置位置
            zero_title = Text("0-seq (should be ≈0 for balanced)", font_size=22, color=GRAY_B).next_to(zero_axis, UP, buff=0.1)  # 标题
            zero_indicator = always_redraw(lambda: Triangle(color=YELLOW, fill_opacity=1).scale(0.1).next_to(zero_axis.n2p((clarke_matrix() @ abc_waveforms(t_tracker.get_value()))[2]), UP, buff=0))  # 零序指示
            self.play(FadeIn(zero_title), FadeIn(zero_axis), FadeIn(zero_indicator))  # 入场

        self.wait(DURATION)  # 等待，让动画运行完时长
        ticker.clear_updaters()  # 清除更新器

        # Hide the old equation panel, theta note, title, and axis labels to make room for the direct formula  # 收起右上角内容
        self.play(FadeOut(eq_block), FadeOut(theta_note), FadeOut(right_label), FadeOut(alpha_label), FadeOut(beta_label))  # 淡出

        # Direct abc→dq0 (shown as plain text)  # 直接给出 abc→dq0 的矩阵式
        eq_direct = MathTex(
            r"[d, q, 0]^T = \frac{2}{3} \begin{bmatrix} \cos\theta & \cos(\theta-\frac{2\pi}{3}) & \cos(\theta+\frac{2\pi}{3}) \\ -\sin\theta & -\sin(\theta-\frac{2\pi}{3}) & -\sin(\theta+\frac{2\pi}{3}) \\ \frac{1}{2} & \frac{1}{2} & \frac{1}{2} \end{bmatrix} [a, b, c]^T",
            font_size=18  # 字体大小
        )
        eq_direct.to_corner(UR).shift(0.2*LEFT + 0.1*DOWN)  # 放到右上角
        self.play(Write(eq_direct))  # 写入公式

        self.wait(2)  # 暂停 2 秒
        self.play(*map(FadeOut, self.mobjects))  # 清空场景
        outro = Text("Park transform → DC on d-axis for balanced three-phase.", font_size=34)  # 结束语
        self.play(FadeIn(outro, shift=UP))  # 显示结束语
        self.wait(2)  # 结束前等待

if __name__ == "__main__":
    scene = ParkTransformation()
    scene.render()