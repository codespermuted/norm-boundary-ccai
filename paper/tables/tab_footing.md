# Footing table (linmix, 6 cells)

| cell | RAW/global | RevIN/global | RAW/window | RevIN/window | RAW/scale | RevIN/scale | CN |
|---|---|---|---|---|---|---|---|
| GEFCom-Wind h=24 | 0.248 | 0.320 | 0.287 | 0.229 | 0.271 | 0.247 | 0.177 |
| GEFCom-Wind h=96 | 0.250 | 0.307 | 0.351 | 0.251 | 0.287 | 0.254 | 0.167 |
| GEFCom-Wind h=336 | 0.381 | 0.419 | 0.441 | 0.364 | 0.374 | 0.381 | 0.199 |
| GEFCom-Solar h=24 | 0.161 | 0.146 | 0.155 | 0.138 | 0.169 | 0.147 | 0.064 |
| GEFCom-Solar h=96 | 0.146 | 0.146 | 0.208 | 0.181 | 0.171 | 0.135 | 0.061 |
| GEFCom-Solar h=336 | 0.192 | 0.195 | 0.615 | 0.446 | 0.251 | 0.178 | 0.066 |
| **mean** | 0.230 | 0.255 | 0.343 | 0.268 | 0.254 | 0.224 | 0.122 |
| **RevIN-RAW** | +0.026 [+0.002,+0.052] |  | -0.075 [-0.118,-0.037] |  | -0.030 [-0.050,-0.012] | |

## endpoints (pre-registered in evidence/prereg_ramp_footing.md)

- PRIMARY (amended): min_f RevIN_f - RAW_global = -0.0142 [-0.0195,-0.0076], positive in 1/6 -> NOT CONFIRMED
- ORIGINAL endpoint: RevIN_scale - RAW_scale = -0.0303 [-0.0504,-0.0119], positive in 1/6
- SECONDARY 1: best endogenous (any footing) - best CN = +0.0930, CN better in 6/6 cells

## SECONDARY 2 -- destroying the covariate level (window_floor - global), per cell

- raw: GEFCom-Wind h=24 +0.039, GEFCom-Wind h=96 +0.101, GEFCom-Wind h=336 +0.060, GEFCom-Solar h=24 -0.006, GEFCom-Solar h=96 +0.062, GEFCom-Solar h=336 +0.422
- revin: GEFCom-Wind h=24 -0.091, GEFCom-Wind h=96 -0.056, GEFCom-Wind h=336 -0.054, GEFCom-Solar h=24 -0.009, GEFCom-Solar h=96 +0.036, GEFCom-Solar h=336 +0.251

## unfloored window footing (degenerate on solar: all-night windows have ~no covariate variance)

- raw: GEFCom-Wind h=24 0.287, GEFCom-Wind h=96 0.351, GEFCom-Wind h=336 0.441, GEFCom-Solar h=24 1.791, GEFCom-Solar h=96 23.377, GEFCom-Solar h=336 787.913
- revin: GEFCom-Wind h=24 0.229, GEFCom-Wind h=96 0.251, GEFCom-Wind h=336 0.364, GEFCom-Solar h=24 0.316, GEFCom-Solar h=96 6.678, GEFCom-Solar h=336 141.349
