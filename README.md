# Three right-arm IMU routes: algorithm reference

This folder presents the model architecture, learning objectives, and the
difference between the three routes.

## Requirements

- Python 3.10 or later
- PyTorch 2.0 or later

Install the runtime dependency with:

```bash
python -m pip install -r requirements.txt
```

## Minimal usage

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

## Student architecture

All three routes deploy the same right-arm student:

1. Build a `50 × 7` sequence from 3-axis acceleration, its 3-axis first
   difference, and acceleration-vector magnitude.
2. Encode the sequence with two residual dilated temporal-convolution blocks.
3. Concatenate temporal mean pooling and temporal max pooling.
4. Apply a nonlinear pre-classification layer and a 19-class linear head.

At inference time the student reads **right-arm IMU only**.

## Teacher architecture

Routes 02 and 03 use a four-location teacher while learning. The teacher input
and forward path are:

```text
4 × (50 × 7)
    → shared TCN encoder
    → 4 × 192-D location features
    → fixed-order concatenation
    → 768 → 192 learned late fusion
    → 19-class logits
```

The fixed location order is `right_arm, right_leg, left_leg, left_arm`. The
shared encoder and classifier can also produce one feature and one prediction
per location. The fused teacher is supervised by:

```text
L_teacher = CE(y, teacher_logits)
```

The teacher supplies detached logits and its fused 192-D feature to the student.
It is not part of the deployed model.

## Route definitions

| Route | Learning objective | Training-only privilege | Inference input |
|---|---|---|---|
| Route 01 | supervised cross-entropy | none | right-arm IMU |
| Route 02 | ordinary uniform logit and feature distillation | four-location teacher | right-arm IMU |
| Route 03 | per-sample, per-knowledge-source dynamic weighted distillation | four-location teacher | right-arm IMU |

For one sample, let `CE` be the hard-label loss, `KL_T` the temperature-scaled
teacher-to-student logit KL divergence, `MSE_z` the feature loss, `rho` a
distillation ramp, and `alpha_l`, `alpha_f` the two source coefficients.

Route 01 uses:

```text
L01 = CE
```

Route 02 uses the same distillation weight for every sample within each source:

```text
L02 = CE + rho * (alpha_l * KL_T + alpha_f * MSE_z)
```

“Uniform” does not mean that the logit and feature coefficients must be equal;
it means there is no sample-dependent weight.

Route 03 introduces two independent sample weights:

```text
L03_i = CE_i + rho * (
    alpha_l * w_logit_i   * KL_T_i
  + alpha_f * w_feature_i * MSE_z_i
)
```

## Dynamic weighting in Route 03

The dynamic weights measure whether a knowledge source agrees with the
right-arm meta objective. Let `theta` be the current student parameters:

1. Evaluate `CE + rho × (logit KD + feature KD)` on a fit batch.
2. Apply one differentiable virtual update to obtain `theta_virtual`.
3. Evaluate hard-label loss on a separate right-arm meta batch at
   `theta_virtual`, then compute `g_meta = grad(theta_virtual, L_meta)`.
4. If the meta-gradient norm is reliable, build parameter probes at
   `theta + xi × normalize(g_meta)` and `theta - xi × normalize(g_meta)`;
   otherwise set both knowledge-source weights to zero.
5. Evaluate both probes with the same dropout random numbers. For every fit
   sample, compute logit-KD and feature-KD losses at both probes.
6. For each source independently, estimate directional influence as
   `(D_plus - D_minus) / (2 * xi)`.
7. Remove numerically uncertain/non-positive evidence and normalize the
   remaining positive evidence to `[0, 1]` within the batch.
8. Use the resulting logit and feature weights in the real student objective. If
   a source has no reliable positive evidence, its batch weights are zero.

`build_meta_parameter_probes` implements steps 3–4, while
`infer_dynamic_source_weights` implements the independent logit/feature mapping
in steps 5–7.

## Reference implementation

The default architecture matches the dimensions above: an 80,915-parameter
student and a 191,507-parameter teacher.

## File

- `three_routes.py`: student/teacher architecture, teacher supervision, the
  three route objectives, and the Route 03 meta-probe weight mapping.

Chinese documentation: [README.zh-CN.md](README.zh-CN.md)
