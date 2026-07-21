# LPS inference (G7 §6)

**Post-hoc refinement — does not alter the pre-registered rule.** The G4 gate decision (results/gate2.md) was taken on the absolute LPS vs the pre-registered tau threshold; the permutation p-values, bootstrap CIs and lambda*_hat below are supporting inference computed afterwards under the identical LPS protocol (w=96, LightGBM, expanding CV, per-channel mean).

- Permutation: circular window shifts, B<=999 (exact enumeration when fewer shifts exist); aligned variant restricts shifts to whole weeks (daily/weekly phase kept under the null).
- CI: 90% moving-block bootstrap, B=499, block_len=ceil(n_windows^(1/3)).
- lambda*_hat: plug-in at h=96 with sigma_Delta^2=0 (lower bound; lam > lambda*_hat necessary, not sufficient, for CN — see src/theory/lps_inference.py docstring for proxy biases).

```
     dataset     lps  p_perm  p_perm_aligned   ci_lo  ci_hi  lambda_star_hat  n_windows  b_perm  b_perm_aligned  align_windows  block_len
   jeju_wind  0.7452  0.0055          0.0385  0.7641 0.8839          -0.1963        181     180              25              7          6
 gefcom_wind  0.7439  0.0084          0.0588  0.7001 0.8521          -0.5131        119     118              16              7          5
 gefcom_load  0.8938  0.0016          0.0110  0.8952 0.9505           0.6764        631     630              90              7          9
gefcom_solar  0.8754  0.0051          0.0345  0.8845 0.9484           0.0929        197     196              28              7          6
       etth1 -0.7165  0.3923          0.3846 -0.0195 0.4033           2.2561        181     180              25              7          6
       etth2 -0.2054  0.1878          0.1923 -0.2908 0.4241           2.8982        181     180              25              7          6
     weather  0.1098  0.3577          0.4074 -0.2223 0.6580           1.0209        548     547              26             21          9
 electricity  0.2830  0.0584          0.0750  0.3793 0.5727           1.3458        274     273              39              7          7
```

## Caveats (2026-07-21 최종)

1. **MBB CI 상향 편의**: 블록 재표집이 fold 간 분포 드리프트를 희석하므로, 음수/경계 LPS 시리즈
   (etth1·etth2·weather·electricity)의 CI는 점추정치보다 체계적으로 위에 놓인다. 표준 그룹의
   해석은 순열 p-값을 우선하라 (CI는 τ 근방·이상에서만 결정적 정보).
2. **electricity = 모든 진단의 경계 셀**: 절대 LPS 0.283(τ 직하), 순열 p 0.058(경계 유의),
   λ̂* 1.35(IN 우세), ΔLPS 0.031(무증분), 실측 gap −0.027 — 사전 등록에서 최저 신뢰로
   명시했던 셀이 사후 추론 전체에서 일관되게 경계로 진단됨.
3. 위 수치는 post-hoc 정밀화이며 사전 등록 τ 규칙(절대 LPS ≥ 0.3)을 대체하지 않는다.
