# Current task

Goal: 在当前 `main` 分支为 GMR 增加可选的 Unitree G1 23-DOF 直接重定向链路，并保持现有 29-DOF 链路可用且行为不变。
Do not: 不新建分支；不替换或重命名 29-DOF 目标；不修改真机、训练、部署；不以裁剪或补零代替语义明确的 23-DOF 重定向。

Current identity:
- Repository: `/Users/chengyuxuan/ArtiIntComVis/GMR`
- Branch: `main`
- Baseline: `05334ed docs: add PHMR to BeyondMimic workflow documentation and shell scripts`

Confirmed decisions:
- 23-DOF 作为新增显式机器人变体，现有 `unitree_g1` 继续表示 29-DOF。
- 采用粗到细路线：系统边界 → 资产与接口契约 → GMR 注册/IK → CLI 与导出 → 29-DOF 回归和 23-DOF 离线验证。
- 每个实施切片先由 `engineering-plan` 定义 owner、保留行为和最近验证；实现尚未授权。

Progress:
- [x] 系统路线 — Done when: 明确输入、模型、IK、导出、下游边界和 29-DOF 兼容不变量
  - Evidence: 当前 `params.py` 只有 29-DOF G1 注册；UniLab 已有 23-DOF 关节顺序和 29→23 参考映射。
- [x] 资产与接口契约 — Done when: 冻结 23-DOF XML 来源、硬件版本、关节/body 顺序和输出 schema
  - Evidence: 暂定当前真机链路使用的 `g1_23dof_rev_1_0.xml`；23 个关节顺序与 UniLab `g1_23dof_joints` 一致；29→23 参考删除索引为 `[13, 14, 20, 21, 27, 28]`。
- [x] GMR 23-DOF 资产、IK 配置和注册 — Done when: 新目标具备独立 XML、SMPL-X IK 配置和注册项
  - Evidence: 新增 `assets/unitree_g1/g1_mocap_23dof.xml`、`general_motion_retargeting/ik_configs/smplx_to_g1_23dof.json`，并追加 `unitree_g1_23dof` 注册；29-DOF 文件未改。
- [x] GMR 直接重定向 smoke — Done when: 官方 SMPL-X 入口可从该序列生成 23 个关节，旧目标输出保持不变
  - Evidence: 修复 loader 后，官方入口用同一动作 3 帧分别跑 `unitree_g1_23dof` 和 `unitree_g1`，输出分别为 23/29 个 DoF，均为有限值；使用 AITViewer 的 SMPL-X `.pkl` body model 验证。
- [x] 入口与导出 — Done when: 新目标可显式选择，默认路径和 29-DOF 导出保持不变
  - Evidence: `smplx_to_robot.py` accepts `unitree_g1_23dof`; `convert_pkl_to_csv.py` accepts `-d 23` and uses an explicit 29→23 semantic index map; README documents both paths.
- [ ] 回归与离线验证 — Done when: 29-DOF 回归通过，23-DOF 短序列 FK/限位/格式检查通过

Blockers:
- BeyondMimic 当前 checkout 仍引用缺失的 `g1/main.urdf`，其 `g1.py`、`csv_to_npz.py` 和 tracking body 配置仍是 29-DOF；这只阻止下游训练联调。

Next:
- 进入 29DoF 回归与 23DoF 输出格式/限位检查，再处理 BeyondMimic 资产联调。
