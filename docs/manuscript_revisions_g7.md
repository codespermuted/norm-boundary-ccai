# 원고 수정 초안 (G7 item 8) — merge-ready LaTeX 스니펫 4 + 보너스 1

작성: 2026-07-20. **초안 전용** — `paper/main.tex`·`paper/sections/`는 이 문서에서 수정하지
않는다. 병합은 동결 해제 후 별도 커밋으로. 모든 수치는 동결 원천에서 재검증:
λ\*_M1 Fig-1 기준선 0.923/0.664/λ\*<0 (`sec3_theory.tex` Fig 1 캡션 = `results/summary.md`
G1), λ\*_M1 합성 매핑 0.27–0.28·λ\*_OLS/실측 0.009–0.029 (`results/gate1.md`),
τ 민감도 (`paper/tables/tab3_draft.md`), 사전 등록 (`paper/predictions.md`, commit
`cab17c1`), Fisher 재분석 (`results/g7_fisher_robustness.csv`).

프리앰블 의존성: 스니펫은 이미 로드된 패키지(amsmath, xcolor, cleveref)와 기존 매크로
(`\lamstar`, `\lps`, `\cnm`, `\inm`, `\sdel`, `\sx`)만 사용. 새 패키지 불필요
(박스는 `\fbox`+`minipage`).

---

## (a) "세 임계값" 박스 — τ를 고정하는 것이 어느 임계값인지 명시

**대상 위치**: `paper/sections/sec4_lps.tex` §4.2 (Decision rule and threshold),
"...quantified in \Cref{tab:tau}." 문단 **직후** (line 17 뒤). §3을 먼저 읽은 독자가
Fig 1의 0.923과 τ = 0.3 사이의 외견상 모순을 갖고 도착하는 지점이다.

```latex
% ---- G7(a): the three thresholds box -------------------------------------
\begin{center}
\fbox{\begin{minipage}{0.92\linewidth}
\textbf{Three thresholds, one anchor.} Three distinct crossover numbers
appear in this paper, and only one of them anchors $\tau$:
\begin{enumerate}
  \item $\lamstar_{\mathrm{M1}} = 0.923$ (at $h=24$; $0.664$ at $h=96$,
  $\lamstar<0$ at $h=336$) --- the \Cref{fig:dominance} \emph{baseline},
  computed with $\sx=0$ and $\sigma_\varepsilon=0$. Switching off
  covariate-driven level motion removes the channel that penalizes
  instance normalization within the horizon, so this is the most
  \inm-favorable stylization; it illustrates the geometry of the dominance
  map and is \emph{not} the anchor for $\tau$.
  \item $\lamstar_{\mathrm{M1}} \approx 0.27$--$0.28$ (at $h=24$) --- the
  same stylized model under the parameter mapping of the synthetic study
  (\Cref{sec:synthetic}), with covariate dynamics active. This is the
  restoration-rule \emph{upper bound} on the true crossover, and it is the
  number that anchors the pre-registered $\tau = 0.3$.
  \item $\lamstar_{\mathrm{OLS}} \approx 0.009$--$0.029$ --- the exact
  constrained-least-squares crossings (Proposition~2$'$), matched by
  trained RLinear models to within $0.015$. This is where trained linear
  predictors actually cross.
\end{enumerate}
Ordering matters: $\tau$ sits at the stylized upper bound (ii), far above
the operative crossover (iii), which is what makes the decision rule
conservative toward instance normalization.
\end{minipage}}
\end{center}
% --------------------------------------------------------------------------
```

검증 노트: (i)은 Fig 1 캡션의 기준 파라미터($V{=}1$, $w{=}96$, $\sigma_z{=}1$,
$\sigma_u^2{=}0.0036$, $\sest{=}0.02$, $\sx{=}0$, $\sigma_\varepsilon{=}0$, $\sdel{=}0$)에서
λ\* = 1 − (24·0.0036 + 1/96 − 0.02) = 0.923 재계산 일치. (ii)·(iii)은 `results/gate1.md`
표(λ\*_M1 0.271–0.283, λ\*_emp 0.009–0.029, |λ\*_emp − λ\*_OLS| 최대 0.015)와 일치.

## (b) 인식론적 지위 문단 — 정확 결과 / 상한 / 경험적 외삽의 구분

**대상 위치**: `paper/sections/sec7_discussion.tex`, limitations 논의 앞
(또는 §3 말미 §3.7 마지막 문단 뒤도 가능 — Discussion 쪽 권장: 전 블록을 인용하므로).

```latex
% ---- G7(b): epistemic status paragraph -----------------------------------
\paragraph{Epistemic status of the claims}
It is worth being explicit about which of our statements are theorems,
which are bounds, and which are extrapolations. \emph{Exact}: the
closed-form risks and the crossover identity of
Propositions~1--3 (verified against Monte Carlo to relative error below
$10^{-2}$ and pinned by unit tests), and Proposition~2$'$, which gives the
optimal predictor within each normalization's \emph{linear} function class
as a constrained least-squares solution with no stochastic optimization
involved. \emph{Bound}: the stylized model M1 brackets the crossover from
above; its threshold is an upper bound whose gap to the exact crossing
measures implicit level tracking (\Cref{sec:synthetic}). \emph{Empirical
extrapolation}: no theorem here covers nonlinear backbones. The evidence
that the same geometry governs PatchTST, SegRNN, and the SOTA covariate
architectures is the pre-registered grid and robustness blocks of
\Cref{sec:empirical}, together with a synthetic demonstration that
replacing the linear backbone by a small MLP preserves the crossover
pattern (Fig.~\ref{fig:mlpdemo}). Readers should weight the three tiers
accordingly: the boundary's existence and location are proved for linear
classes and measured--not derived--for nonlinear ones.
% --------------------------------------------------------------------------
```

병합 노트: `fig:mlpdemo`는 G7 Block E(MLP 데모, figG)의 라벨 자리표시 — Block E 트랙이
확정한 실제 라벨로 치환할 것. Block E가 논문에 실리지 않기로 결정되면 해당 절
("together with ... (Fig.~\ref{fig:mlpdemo})")을 삭제해도 문단은 자립한다.

## (c) τ–LPS 순서 공개 문단 — 무엇이 사전이고 무엇이 사후인지

**대상 위치**: `paper/sections/sec6_empirical.tex` §6.5 (LPS predicts the gap),
\Cref{tab:tau} 논의 문단(line 103, "…the pre-registered absolute-$\lps$ verdict stands.")
**직후**. 대안: `sec4_lps.tex` §4.3 말미.

```latex
% ---- G7(c): tau-LPS ordering disclosure ----------------------------------
We also make the ordering of evidence explicit, since a threshold that
performs well on a grid invites the suspicion that it was tuned on that
grid. It was not: $\tau = 0.3$ was fixed from theory--the stylized upper
bound $\lamstar_{\mathrm{M1}} \approx 0.27$--$0.28$ of
\Cref{sec:theory}--and committed together with all eight sign predictions
(commit \texttt{cab17c1}) before any grid run was launched. The
$[0.30, 0.70]$ plateau of \Cref{tab:tau} is therefore a \emph{post-hoc}
robustness finding, reported as such, not part of the pre-registered
procedure. The sensitivity table also shows that the conclusion survives
moderate misplacement of the threshold: even $\tau = 0.25$, below the
plateau, yields $7/8$ sign hits--still clearing the pre-registered
$\ge 6/8$ bar--and the rule degrades gradually ($6/8$ on $[0.00,0.10]$
and $[0.75,0.85]$) rather than collapsing outside the plateau.
% --------------------------------------------------------------------------
```

검증 노트: `paper/tables/tab3_draft.md` (a)행과 일치 — τ=0.25 → 7, [0.30,0.70] → 8,
[0.00,0.10]·[0.75,0.85] → 6, 0.90 → 4. `paper/predictions.md`가 τ=0.3의 근거로
λ\*_M1 0.27–0.28을 명시.

## (d) 개발/확증 데이터셋 각주 — Data 절

**대상 위치**: `paper/sections/sec6_empirical.tex` §6.1 (Data) 첫 문단, Jeju Wind 서술의
문장 끝 (예: "...as physically expected." 뒤 또는 문단 마지막 문장에 각주 부착).

```latex
% ---- G7(d): development vs confirmation footnote -------------------------
\footnote{A transparency note on dataset provenance: Jeju Wind was curated
by us during this study (data collection, NWP lead-matching, and quality
screening were interleaved with method development), so it should be read
as a \emph{development} dataset. GEFCom2014 and the standard long-horizon
benchmarks are third-party datasets frozen before this project began, and
play a confirmation-style role. The distinction does not affect the
pre-registration guarantee: the sign predictions for all eight
datasets--including Jeju Wind--were derived from the $\lps$ rule and
committed (commit \texttt{cab17c1}) before any grid run on any dataset
was launched.}
% --------------------------------------------------------------------------
```

검증 노트: 큐레이션 이력은 `results/summary.md` G3 절(NWP 수집·큐레이션이 본 연구 중
수행됨), 사전 등록 시각 증빙은 무결성 감사(`HANDOFF.md`: `cab17c1` 13:20:23 < 최초 run
13:24:36)와 일치.

---

## (e) 보너스 — Fisher 의존성 공개 각주 (item 7(b) 권고의 실행)

**대상 위치**: `paper/sections/sec6_empirical.tex` §6.3 (Statistical testing), 문장
"Dataset-level marks in \Cref{tab:main} combine the cell-level $p$-values by Fisher's
method ($p<0.05$)." 끝에 각주 부착.

```latex
% ---- G7(e): Fisher dependence disclosure ---------------------------------
\footnote{Fisher's method assumes independent $p$-values, whereas the
per-(backbone,\,$h$) cells within a dataset share the same test period and
are positively dependent, making the combination anti-conservative. As a
robustness check we recombined all cells with two dependence-robust rules:
under the harmonic-mean $p$-value \citep{wilson2019hmp}, valid under
arbitrary dependence, and under Simes's procedure, $31$ of the $32$ stars
in \Cref{tab:main} are unchanged. The single exception is RevIN on ETTh2
(only $3$ of $9$ cells individually significant; Fisher $p=0.026$,
harmonic-mean $p=0.077$, Simes $p=0.130$), which should be read as
indicative; the qualitative conclusion for ETTh2 (instance normalization
dominates conditional normalization by a wide margin) does not rest on
this mark.}
% --------------------------------------------------------------------------
```

병합 노트: `references.bib`에 추가 필요 —

```bibtex
@article{wilson2019hmp,
  author  = {Wilson, Daniel J.},
  title   = {The harmonic mean $p$-value for combining dependent tests},
  journal = {Proceedings of the National Academy of Sciences},
  year    = {2019},
  volume  = {116},
  number  = {4},
  pages   = {1195--1200},
}
```

수치 원천: `results/g7_fisher_robustness.csv`
(`uv run python -m experiments.g7_fisher_check`), 상세 논의 `docs/stats_hardening.md`.
