#!/usr/bin/env bash
# supervised_run.sh — OOM/충돌 내성 재시도 래퍼 (norm-boundary 공용)
#
#   scripts/supervised_run.sh <로그파일> <최대재시도> -- <명령...>
#   예: scripts/supervised_run.sh results/pytest_supervised.log 3 -- uv run pytest -q
#
# 동작:
#  - 비정상 종료(OOM SIGKILL=137 포함) 시 60s * 시도횟수 백오프 후 재실행
#  - 스레드 상한 기본값 주입: 소형 데이터 LightGBM 스레드 경합(fit당 ~90s 퇴화)과
#    과도한 동시성으로 인한 메모리 폭주 예방 (호출 측에서 이미 설정했으면 존중)
#  - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True로 CUDA 단편화 OOM 완화
#  - 각 시도의 시각·종료코드를 로그에 기록, 최종 실패 시 종료코드 그대로 전파
set -u

LOG="$1"; MAX="$2"; shift 2
[ "$1" = "--" ] && shift

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

mkdir -p "$(dirname "$LOG")"
attempt=1
while :; do
    echo "[supervised] attempt ${attempt}/${MAX} $(date '+%F %T') :: $*" | tee -a "$LOG"
    "$@" >> "$LOG" 2>&1
    code=$?
    echo "[supervised] attempt ${attempt} exit=${code} $(date '+%F %T')" | tee -a "$LOG"
    [ "$code" -eq 0 ] && exit 0
    if [ "$code" -eq 137 ]; then
        echo "[supervised] exit 137 (SIGKILL — OOM 의심). free -g:" | tee -a "$LOG"
        free -g >> "$LOG" 2>&1
    fi
    if [ "$attempt" -ge "$MAX" ]; then
        echo "[supervised] giving up after ${MAX} attempts (last exit=${code})" | tee -a "$LOG"
        exit "$code"
    fi
    sleep $((60 * attempt))
    attempt=$((attempt + 1))
done
