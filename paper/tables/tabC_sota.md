# Tab C 초안 — SOTA 확장 (사전 등록 외 robustness)

## TimeXer-MS (공식 exogenous 경로)

| dataset      |    raw |   revin |    san |    fan |   condnorm |
|:-------------|-------:|--------:|-------:|-------:|-----------:|
| gefcom_load  | 0.355  |  0.3249 | 0.3238 | 0.3434 |     0.104  |
| gefcom_solar | 0.1784 |  0.1778 | 0.2491 | 0.191  |     0.057  |
| gefcom_wind  | 1.027  |  1.0772 | 1.0502 | 1.0513 |     0.1668 |
| jeju_wind    | 0.6561 |  0.688  | 0.6529 | 0.6864 |     0.2989 |

평균 격차 RevIN−CondNorm = +0.4122 (55쌍) — CN 우세

## iTransformer-MS (공변량 variate 토큰)

| dataset      |    raw |   revin |    san |    fan |   condnorm |
|:-------------|-------:|--------:|-------:|-------:|-----------:|
| gefcom_load  | 0.3606 |  0.3288 | 0.3122 | 0.347  |     0.1053 |
| gefcom_solar | 0.1901 |  0.1784 | 0.229  | 0.1865 |     0.0567 |
| gefcom_wind  | 1.0267 |  1.1226 | 1.0449 | 1.0604 |     0.1658 |
| jeju_wind    | 0.6775 |  0.7036 | 0.664  | 0.6942 |     0.3012 |

평균 격차 RevIN−CondNorm = +0.4282 (55쌍) — CN 우세

## Endogenous 확장 (iTransformer / TimeXer-M): 정규화 arm 평균 MSE

|                                  |    raw |   revin |    san |    fan |   condnorm |
|:---------------------------------|-------:|--------:|-------:|-------:|-----------:|
| ('itransformer', 'electricity')  | 0.1341 |  0.1314 | 0.13   | 0.1339 |     0.2096 |
| ('itransformer', 'etth1')        | 0.4413 |  0.3826 | 0.3857 | 0.3976 |     1.4851 |
| ('itransformer', 'etth2')        | 0.6742 |  0.3037 | 0.3002 | 0.3329 |     3.4686 |
| ('itransformer', 'gefcom_load')  | 0.3524 |  0.3364 | 0.3251 | 0.3524 |     0.1074 |
| ('itransformer', 'gefcom_solar') | 0.1808 |  0.1774 | 0.1778 | 0.1837 |     0.0569 |
| ('itransformer', 'gefcom_wind')  | 1.0399 |  0.9734 | 0.9781 | 1.0844 |     0.1586 |
| ('itransformer', 'jeju_wind')    | 0.6963 |  0.7242 | 0.694  | 0.6984 |     0.2974 |
| ('itransformer', 'weather')      | 0.1907 |  0.1706 | 0.168  | 0.17   |     0.6992 |
| ('timexer', 'etth1')             | 0.3962 |  0.3805 | 0.3895 | 0.4309 |     1.4821 |
| ('timexer', 'etth2')             | 0.4838 |  0.2946 | 0.2897 | 0.3815 |     3.5344 |

주: GATE 2는 사전 등록된 원 4백본으로만 판정; 본 표는 등록 후 확장으로 별도 보고 (docs/design_audit.md 블록 C/D).