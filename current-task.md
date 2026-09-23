# Current task

Goal: 以 TrackerLab 的原生 G1 23-DOF 语义为外部基线，改善 GMR 的 23-DOF 重定向质量，并保持现有 29-DOF 链路、输出 schema 和默认行为不变。
Do not: 不把 TrackerLab 或 UMR 引入 GMR 运行时；不修改 PHMR、BeyondMimic、训练或真机链路；不裁剪或补零动作；不先改通用求解器再验证 23-DOF 配置边界。

Current identity:
- Repository: `/Users/chengyuxuan/ArtiIntComVis/GMR`
- Branch: `main`
- HEAD: `c7c1d63 again bug fix`
- Reference: TrackerLab 仅作为 23-DOF 资产、T-pose、关节映射和目标语义的对照，不成为生产依赖。

Confirmed decisions:
- `unitree_g1` 继续表示 29-DOF；所有新行为由 `unitree_g1_23dof` 显式选择。
- GMR 的 Mink IK 继续作为生产求解器，先校正 23-DOF 目标语义，再决定是否需要默认关闭的求解器能力。
- 先处理用户指出的手臂偏转、前后错位和快速解交叉片段；不把动作风格优化扩大为全身算法重写。
- 每个生产修改切片执行前用 `engineering-plan` 明确 owner、保留行为和最近验证。

Progress:
- [x] 原生 23-DOF 链路存在性 — Done when: 独立资产、注册、IK 配置和导出入口可生成 `[T, 23]`
  - Evidence: `g1_mocap_23dof.xml`、`smplx_to_g1_23dof.json` 和 `unitree_g1_23dof` 已存在；此前短序列 smoke 得到有限的 23 维输出。
- [x] 当前差异定位 — Done when: 找到与手臂质量直接相关的当前 owner 和未决边界
  - Evidence: `smplx_to_g1_23dof.json` 仍以左右人体 wrist 驱动 `*_wrist_roll_rubber_hand` 位置；`motion_retarget.py` 已支持可选 posture cost 和输出逐帧限速，但没有证明目标语义正确。
- [x] 第一步：冻结 TrackerLab→GMR 23-DOF 语义差分 — Done when: 明确关节/body、T-pose/坐标、人体目标、尺度和问题片段的基准测量，得到一个可执行的配置变更清单
  - Evidence: TrackerLab 将 shoulder 映射到 `*_shoulder_pitch_link` 并以 T-pose 骨架旋转传递姿态；当前 GMR 23-DOF 的手臂有效任务只有左右 elbow/wrist 位置，所有肩部任务和手臂方向任务均为零。两套关节数组顺序不同，不能直接复制。
  - Evidence: `1:14–1:25` 中左右目标肩腕长度 p95 为 `0.411/0.401 m`，机器人实际最大约 `0.325 m`；左右 wrist 相对位置误差均值为 `0.235/0.215 m`，左右 shoulder pitch 全部帧处于 5% 限位边界内，手臂速度达到配置上限 `9.4248 rad/s`。
  - Evidence: 该窗口的人体源动作已有最大约 `3.85 cm` 的右腕相对前伸，机器人最大约 `2.77 cm`，这一项不是 GMR 放大；但全序列机器人左腕横向峰值速度 `2.243 m/s`，源动作峰值 `0.765 m/s`，说明快速动作被重定向放大。
- [x] 第二步：仅校正 23-DOF IK 配置 — Done when: 不改通用求解器，问题片段的手臂偏转、饱和和逐帧跳变相对当前输出改善，29-DOF 文件无差异
  - Evidence: 仅修改 `smplx_to_g1_23dof.json`：手臂尺度 `0.8→0.62`，恢复肩/肘方向任务并把 wrist 位置权重 `10→5`；JSON 与 diff 检查通过，29-DOF 文件未改。
  - Evidence: 服务器同一 `1:14–1:25` 330 帧 CPU smoke 输出有限 `[330, 23]`；左右 shoulder pitch 近限位比例从 `100%/100%` 降至 `59.4%/46.7%`。最大手臂速度仍为 `9.4248 rad/s`，上限命中仅从 `7→6`，证明配置修正有改善但不足以关闭质量问题。
  - Evidence: 服务器测试使用临时配置并已恢复为 `0.8 / [0,0] / [10,0]`；服务器原有 `main.urdf` 修改未触碰。
- [x] 第三步：按证据补充 23-DOF 可选求解能力 — Done when: 仅在第二步不足时增加默认关闭的软限位或轨迹正则，并证明 29-DOF 默认路径不变
  - Evidence: `motion_retarget.py` 新增配置驱动的软关节限位姿态任务；缺少 `soft_joint_limit` 时不创建任务，现有 IK task、硬限位和输出速度限制均未改。
  - Evidence: 仅 `smplx_to_g1_23dof.json` 启用该能力，对双臂肩/肘设置 10% 安全边界和 `1.0` 权重；配置中的 8 个关节均存在于 23-DOF XML 且带 range。
  - Evidence: Python 编译、JSON 解析和 `git diff --check` 通过；服务器现有 Mink 版本成功构造长度为 `model.nv` 的向量权重 `PostureTask`。完整序列质量验证归入第四步。
- [ ] 第四步：端到端离线验收 — Done when: 完整 SMPL-X→PKL→CSV 输出保持 23 关节契约，问题片段可视化和数值指标通过，29-DOF 最近回归通过

Blockers:
- 本地视频、SMPL-X 和 PKL 都只有 `225.067 s`，用户给出的 `4:20–4:21` 超出范围；在最终定点验收前需要对应的正确时间戳。全序列证据足以继续第二步配置修正。

Next:
- 使用 `engineering-plan` 规划第四步：同步到具备 SMPL-X checkpoint 的服务器后执行完整 23-DOF 离线验收，并做一次最近的 29-DOF 默认路径回归。
