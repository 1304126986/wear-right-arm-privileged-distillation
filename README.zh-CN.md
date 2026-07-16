# 三条右臂 IMU 路线：算法说明版

这个目录说明论文中的模型结构、三条路线差别和动态加权算法。

## 环境要求

- Python 3.10 或更高版本
- PyTorch 2.0 或更高版本

安装运行依赖：

```bash
python -m pip install -r requirements.txt
```

## 最小调用示例

```python
import torch

from three_routes import FourLocationTeacher, RightArmStudent

student = RightArmStudent().eval()
teacher = FourLocationTeacher().eval()

right_arm = torch.randn(2, 50, 7)
four_locations = torch.randn(2, 4, 50, 7)

with torch.no_grad():
    student_output = student(right_arm)
    teacher_output = teacher(four_locations)

print(student_output.logits.shape)  # torch.Size([2, 19])
print(teacher_output.logits.shape)  # torch.Size([2, 19])
```

## Student 结构

三条路线部署时使用完全相同的 right-arm student：

1. 将三轴加速度、三轴一阶差分和加速度向量模长组成 `50 × 7` 序列；
2. 使用两层带残差连接的膨胀时序卷积进行编码；
3. 拼接时间维 mean pooling 与 max pooling；
4. 经非线性分类前投影后，输出 19 类 logits。

推理时 student **只读取右臂 IMU**。

## Teacher 结构

Route 02 和 Route 03 使用四位置 teacher，其输入与前向路径为：

```text
4 × (50 × 7)
    → 共享 TCN 编码器
    → 4 × 192 维位置特征
    → 按固定顺序拼接
    → 768 → 192 可学习后期融合
    → 19 类 logits
```

四个位置的固定顺序为 `right_arm, right_leg, left_leg, left_arm`。共享编码器和
分类器也可以分别输出每个位置的 feature 与预测。融合 teacher 的监督目标为：

```text
L_teacher = CE(y, teacher_logits)
```

teacher 向 student 提供停止梯度的 logits 和融合后的 192 维 feature，不属于
部署模型。

## 三条路线

| 路线 | 学习目标 | 训练期特权信息 | 推理输入 |
|---|---|---|---|
| Route 01 | 仅监督交叉熵 | 无 | 右臂 IMU |
| Route 02 | 普通等权的 logit + feature 蒸馏 | 四位置 teacher | 右臂 IMU |
| Route 03 | 按样本、按知识源动态加权蒸馏 | 四位置 teacher | 右臂 IMU |

记 `CE` 为硬标签损失，`KL_T` 为带温度缩放的 logit 蒸馏损失，`MSE_z` 为
feature 蒸馏损失，`rho` 为蒸馏强度渐增系数，`alpha_l`、`alpha_f` 为两个
知识源的系数。

Route 01：

```text
L01 = CE
```

Route 02 对同一知识源内的所有样本使用相同权重：

```text
L02 = CE + rho * (alpha_l * KL_T + alpha_f * MSE_z)
```

这里的“等权”指不存在样本级权重，并不要求 logit 与 feature 的两个系数相等。

Route 03 为两个知识源分别计算样本级权重：

```text
L03_i = CE_i + rho * (
    alpha_l * w_logit_i   * KL_T_i
  + alpha_f * w_feature_i * MSE_z_i
)
```

## Route 03 的动态权重

动态权重用于判断某条 teacher 知识是否与 right-arm meta 目标一致。记当前
student 参数为 `theta`：

1. 在 fit batch 上计算 `CE + rho × (logit KD + feature KD)`；
2. 做一次可微的虚拟更新，得到 `theta_virtual`；
3. 在 `theta_virtual` 上计算独立 right-arm meta batch 的硬标签损失，并得到
   `g_meta = grad(theta_virtual, L_meta)`；
4. meta 梯度模长可靠时，构造 `theta + xi × normalize(g_meta)` 与
   `theta - xi × normalize(g_meta)` 两个参数探针；模长过小时，两项权重均置零；
5. 两个探针复用相同的 dropout 随机数，分别计算每个 fit 样本的 logit-KD 和
   feature-KD；
6. 对两个知识源分别计算方向影响：`(D_plus - D_minus) / (2 × xi)`；
7. 去除数值不确定或非正的证据，将剩余正证据在 batch 内归一化到 `[0, 1]`；
8. 将两组独立权重用于 student 的真实目标；若某知识源没有可靠正证据，则该
   batch 中对应权重全部置零。

`build_meta_parameter_probes` 实现第 3–4 步，`infer_dynamic_source_weights`
实现第 5–7 步中 logit 与 feature 两个知识源的独立权重映射。

## 参考实现

默认结构与上面的维度一致：student 参数量为 80,915，teacher 参数量为
191,507。

## 文件与边界

- `three_routes.py`：student/teacher 结构、teacher 监督目标、三条路线目标函数，
  以及 Route 03 的 meta 参数探针和动态权重映射。

English documentation: [README.md](README.md)
