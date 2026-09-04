#!/usr/bin/env bash
set +e
set -uo pipefail

# Runs OpenVM recursive and WARP benchmark lanes one by one and writes:
#   - one metrics JSON per run, via OUTPUT_PATH
#   - one log per run
#   - a summary CSV with timing and parsed counters
#
# Usage:
#   scripts/run_warp_recursive_benchmarks.sh
#
# Optional environment overrides:
#   BENCHES="fibonacci sha2_bench keccak"
#   MODES="recursive warp"
#   LIFECYCLES="cold warm"           # key generation inside / outside proving
#   OUT_DIR="benchmark-results/my-run"
#   CARGO_PROFILE="--release"        # set empty for debug
#   CARGO_FEATURES="cuda"             # comma- or space-separated Cargo features
#   RUST_LOG="info,p3_=warn"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || printf 'unknown')"
GIT_BRANCH="$(git branch --show-current 2>/dev/null || printf 'unknown')"

BENCHES_STR="${BENCHES:-fibonacci sha2_bench keccak rkyv bincode}"
MODES_STR="${MODES:-recursive warp}"
# Both lanes generate proving keys lazily. `cold` charges key generation to the
# measured proving window for recursive and WARP alike; `warm` charges it to
# setup for both. Comparing a warm WARP run against a cold recursive run is not
# a like-for-like measurement, so the lifecycle is always a shared axis.
LIFECYCLES_STR="${LIFECYCLES:-cold warm}"
CARGO_PROFILE="${CARGO_PROFILE:---release}"
CARGO_FEATURES="${CARGO_FEATURES:-}"
OUT_DIR="${OUT_DIR:-benchmark-results/warp-vs-recursive-$(date +%Y%m%d-%H%M%S)}"
SUMMARY_CSV="$OUT_DIR/summary.csv"

mkdir -p "$OUT_DIR"

printf 'timestamp,benchmark,mode,lifecycle,status,exit_code,elapsed_seconds,peak_rss_kib,peak_swap_kib,benchmark_setup_time_ms,benchmark_prove_time_ms,benchmark_verify_time_ms,benchmark_verify_p95_ms,benchmark_verify_first_ms,benchmark_verify_attribution_total_ms,benchmark_proof_serialize_time_ms,segments_from_log,warp_fresh_segments,warp_shape_count,warp_invocations,warp_root_steps,warp_terminal_whir,instructions_executed,proof_size_bytes,proof_size_compressed,native_reduction_bytes,warp_root_proof_bytes,terminal_whir_bytes,warp_planning_ms,warp_main_trace_commit_ms,warp_native_reduction_ms,warp_segment_execute_ms,warp_reduction_replay_ms,warp_group_other_ms,warp_batch_other_ms,warp_history_finalize_ms,warp_completion_other_ms,warp_unattributed_ms,warp_attribution_ratio,warp_vacc_ms,warp_history_setup_keygen_ms,warp_history_keygen_ms,warp_history_prove_ms,warp_terminal_linearizer_ms,warp_terminal_whir_ms,history_certificate_bytes,terminal_opened_rows_bytes,terminal_multiproof_bytes,warp_verify_history_ms,warp_verify_terminal_whir_ms,warp_verify_linearizer_ms,metrics_path,log_path\n' > "$SUMMARY_CSV"

metric_value() {
    local metrics_file="$1"
    local metric="$2"
    [ -s "$metrics_file" ] || return 0
    awk -v name="$metric" '
        $0 ~ "\"metric\": \"" name "\"" { found = 1; next }
        found && $0 ~ "\"value\":" {
            value = $2
            gsub(/[",]/, "", value)
            print value
            exit
        }
    ' "$metrics_file"
}

last_log_number_after_colon() {
    local log_file="$1"
    local label="$2"
    local line
    line="$(grep -F "$label" "$log_file" 2>/dev/null | tail -n 1 || true)"
    [ -n "$line" ] || return 0
    printf '%s\n' "$line" | sed -E 's/.*: *([0-9]+).*/\1/' || true
}

last_log_value_after_colon() {
    local log_file="$1"
    local label="$2"
    local line
    line="$(grep -F "$label" "$log_file" 2>/dev/null | tail -n 1 || true)"
    [ -n "$line" ] || return 0
    printf '%s\n' "$line" | sed -E 's/.*: *([0-9]+(\.[0-9]+)?).*/\1/' || true
}

last_instruction_count() {
    local log_file="$1"
    grep -Eo 'instructions_executed=[0-9]+' "$log_file" 2>/dev/null \
        | tail -n 1 \
        | sed -E 's/.*=//' \
        || true
}

log_proof_size() {
    local log_file="$1"
    local field="$2"
    local line
    line="$(grep -F "Proof Size (bytes):" "$log_file" 2>/dev/null | tail -n 1 || true)"
    [ -n "$line" ] || return 0
    case "$field" in
        raw) printf '%s\n' "$line" | sed -E 's/.*Proof Size \(bytes\): *([0-9]+),.*/\1/' ;;
        compressed) printf '%s\n' "$line" | sed -E 's/.*Compressed Size: *([0-9]+).*/\1/' ;;
    esac
}

csv_append() {
    local first=1
    for value in "$@"; do
        if [ "$first" -eq 0 ]; then printf ',' >> "$SUMMARY_CSV"; fi
        printf '%s' "$value" >> "$SUMMARY_CSV"
        first=0
    done
    printf '\n' >> "$SUMMARY_CSV"
}

process_tree_memory_kib() {
    local root="$1"
    local pending="$root"
    local all=""
    local pid children
    while [ -n "$pending" ]; do
        pid="${pending%% *}"
        if [ "$pending" = "$pid" ]; then pending=""; else pending="${pending#* }"; fi
        all="$all $pid"
        children="$(pgrep -P "$pid" 2>/dev/null | tr '\n' ' ' || true)"
        pending="$pending $children"
        pending="$(printf '%s' "$pending" | xargs 2>/dev/null || true)"
    done
    local total_rss=0
    local total_swap=0
    local rss swap
    for pid in $all; do
        rss="$(awk '/VmRSS:/ {print $2}' "/proc/$pid/status" 2>/dev/null || true)"
        swap="$(awk '/VmSwap:/ {print $2}' "/proc/$pid/status" 2>/dev/null || true)"
        total_rss=$((total_rss + ${rss:-0}))
        total_swap=$((total_swap + ${swap:-0}))
    done
    printf '%s %s\n' "$total_rss" "$total_swap"
}

run_one() {
    local bench="$1"
    local mode="$2"
    local lifecycle="$3"
    local timestamp
    local safe_name
    local metrics_path
    local log_path
    local start_ns
    local end_ns
    local elapsed_seconds
    local peak_rss_kib
    local peak_swap_kib
    local status
    local exit_code

    case "$lifecycle" in
        cold|warm) ;;
        *)
            echo "Unknown lifecycle '$lifecycle' for benchmark '$bench'" >&2
            return 2
            ;;
    esac

    timestamp="$(date --iso-8601=seconds)"
    safe_name="${bench}_${mode}_${lifecycle}"
    metrics_path="$OUT_DIR/${safe_name}.metrics.json"
    log_path="$OUT_DIR/${safe_name}.log"

    local cmd=(cargo run)
    if [ -n "$CARGO_PROFILE" ]; then
        # shellcheck disable=SC2206
        local profile_parts=($CARGO_PROFILE)
        cmd+=("${profile_parts[@]}")
    fi
    if [ -n "$CARGO_FEATURES" ]; then
        cmd+=(--features "$CARGO_FEATURES")
    fi
    cmd+=(-p openvm-benchmarks-prove --bin "$bench")
    if [ "$mode" = "warp" ]; then
        cmd+=(-- --warp)
    elif [ "$mode" != "recursive" ]; then
        echo "Unknown mode '$mode' for benchmark '$bench'" >&2
        return 2
    fi

    echo
    echo "==> [$timestamp] running $bench / $mode / $lifecycle"
    echo "    metrics: $metrics_path"
    echo "    log:     $log_path"
    echo "    command: OUTPUT_PATH=$metrics_path ${cmd[*]}"
    printf '%s\n' \
        "Benchmark workspace: $ROOT_DIR" \
        "Benchmark git branch: $GIT_BRANCH" \
        "Benchmark git commit: $GIT_COMMIT" \
        > "$log_path"

    start_ns="$(date +%s%N)"
    set +e
    OUTPUT_PATH="$metrics_path" OPENVM_BENCH_LIFECYCLE="$lifecycle" \
        RUST_LOG="${RUST_LOG:-info,p3_=warn}" "${cmd[@]}" \
        > >(tee -a "$log_path") 2>&1 &
    local command_pid=$!
    peak_rss_kib=0
    peak_swap_kib=0
    while kill -0 "$command_pid" 2>/dev/null; do
        local current_rss current_swap
        read -r current_rss current_swap < <(process_tree_memory_kib "$command_pid")
        if [ "${current_rss:-0}" -gt "$peak_rss_kib" ]; then
            peak_rss_kib="$current_rss"
        fi
        if [ "${current_swap:-0}" -gt "$peak_swap_kib" ]; then
            peak_swap_kib="$current_swap"
        fi
        sleep 0.1
    done
    wait "$command_pid"
    exit_code="$?"
    set +e
    end_ns="$(date +%s%N)"
    elapsed_seconds="$(awk -v start="$start_ns" -v end="$end_ns" 'BEGIN { printf "%.3f", (end - start) / 1000000000 }')"

    if [ "$exit_code" -eq 0 ]; then
        status="ok"
    else
        status="failed"
    fi

    local segments_from_log
    local warp_fresh_segments
    local warp_shape_count
    local warp_root_steps
    local warp_invocations
    local warp_terminal_whir
    local instructions_executed
    local proof_size_bytes
    local proof_size_compressed
    local benchmark_setup_time_ms
    local benchmark_prove_time_ms
    local benchmark_verify_time_ms
    local benchmark_verify_p95_ms
    local benchmark_verify_first_ms
    local benchmark_verify_attribution_total_ms
    local warp_verify_history_ms
    local warp_verify_terminal_whir_ms
    local warp_verify_linearizer_ms
    local benchmark_proof_serialize_time_ms
    local native_reduction_bytes
    local warp_root_proof_bytes
    local terminal_whir_bytes
    local warp_planning_ms
    local warp_main_trace_commit_ms
    local warp_native_reduction_ms
    local warp_vacc_ms
    local warp_history_setup_keygen_ms
    local warp_history_keygen_ms
    local warp_history_prove_ms
    local warp_terminal_linearizer_ms
    local warp_terminal_whir_ms
    local history_certificate_bytes
    local terminal_opened_rows_bytes
    local terminal_multiproof_bytes
    local warp_segment_execute_ms
    local warp_reduction_replay_ms
    local warp_group_other_ms
    local warp_batch_other_ms
    local warp_history_finalize_ms
    local warp_completion_other_ms
    local warp_unattributed_ms
    local warp_attribution_ratio

    segments_from_log="$(grep -Ec 'Segment[[:space:]]+[0-9]+[[:space:]]+\|' "$log_path" 2>/dev/null || true)"
    benchmark_setup_time_ms="$(metric_value "$metrics_path" "benchmark_setup_time_ms")"
    benchmark_prove_time_ms="$(metric_value "$metrics_path" "benchmark_prove_time_ms")"
    benchmark_verify_time_ms="$(metric_value "$metrics_path" "benchmark_verify_time_ms")"
    benchmark_proof_serialize_time_ms="$(metric_value "$metrics_path" "benchmark_proof_serialize_time_ms")"
    warp_fresh_segments="$(metric_value "$metrics_path" "native_warp_fresh_segments.total")"
    warp_root_steps="$(metric_value "$metrics_path" "warp_root_steps.total")"
    warp_shape_count="$(last_log_number_after_colon "$log_path" "WARP exact SWIRL shape count")"
    warp_invocations="$(metric_value "$metrics_path" "warp_invocations.total")"
    warp_terminal_whir="$(metric_value "$metrics_path" "native_warp_terminal_whir.total")"
    native_reduction_bytes="$(metric_value "$metrics_path" "warp_proof_size_bytes.native_reduction")"
    warp_root_proof_bytes="$(metric_value "$metrics_path" "warp_proof_size_bytes.root")"
    terminal_whir_bytes="$(metric_value "$metrics_path" "warp_proof_size_bytes.terminal")"
    warp_planning_ms="$(metric_value "$metrics_path" "native_warp_phase_ms.planning")"
    warp_main_trace_commit_ms="$(metric_value "$metrics_path" "native_warp_phase_ms.main_trace_commit")"
    warp_native_reduction_ms="$(metric_value "$metrics_path" "native_warp_phase_ms.native_reduction")"
    warp_vacc_ms="$(metric_value "$metrics_path" "native_warp_phase_ms.vacc")"
    warp_history_setup_keygen_ms="$(metric_value "$metrics_path" "native_warp_phase_ms.history_setup_keygen")"
    warp_history_keygen_ms="$(metric_value "$metrics_path" "native_warp_phase_ms.history_stage_keygen")"
    warp_history_prove_ms="$(metric_value "$metrics_path" "native_warp_phase_ms.history_stage_prove")"
    warp_terminal_linearizer_ms="$(metric_value "$metrics_path" "native_warp_phase_ms.terminal_linearizer")"
    warp_terminal_whir_ms="$(metric_value "$metrics_path" "native_warp_phase_ms.terminal_whir")"
    warp_segment_execute_ms="$(metric_value "$metrics_path" "native_warp_phase_ms.segment_execute")"
    warp_reduction_replay_ms="$(metric_value "$metrics_path" "native_warp_phase_ms.reduction_replay")"
    warp_group_other_ms="$(metric_value "$metrics_path" "native_warp_phase_ms.group_other")"
    warp_batch_other_ms="$(metric_value "$metrics_path" "native_warp_phase_ms.batch_other")"
    warp_history_finalize_ms="$(metric_value "$metrics_path" "native_warp_phase_ms.history_finalize")"
    warp_completion_other_ms="$(metric_value "$metrics_path" "native_warp_phase_ms.completion_other")"
    warp_unattributed_ms="$(metric_value "$metrics_path" "native_warp_phase_ms.unattributed")"
    warp_attribution_ratio="$(metric_value "$metrics_path" "native_warp_attribution_ratio")"
    history_certificate_bytes="$(metric_value "$metrics_path" "warp_proof_size_bytes.history_certificate")"
    terminal_opened_rows_bytes="$(metric_value "$metrics_path" "warp_proof_size_bytes.terminal_opened_rows")"
    terminal_multiproof_bytes="$(metric_value "$metrics_path" "warp_proof_size_bytes.terminal_multiproofs")"
    benchmark_verify_p95_ms="$(metric_value "$metrics_path" "benchmark_verify_p95_ms")"
    benchmark_verify_first_ms="$(metric_value "$metrics_path" "benchmark_verify_first_ms")"
    benchmark_verify_attribution_total_ms="$(metric_value "$metrics_path" "benchmark_verify_attribution_total_ms")"
    warp_verify_history_ms="$(metric_value "$metrics_path" "native_warp_phase_ms.verify_history_certificate")"
    warp_verify_terminal_whir_ms="$(metric_value "$metrics_path" "native_warp_phase_ms.verify_terminal_whir")"
    warp_verify_linearizer_ms="$(metric_value "$metrics_path" "native_warp_phase_ms.verify_terminal_linearizer")"
    instructions_executed="$(last_instruction_count "$log_path")"
    proof_size_bytes="$(metric_value "$metrics_path" "proof_size_bytes.total")"
    proof_size_compressed="$(metric_value "$metrics_path" "proof_size_bytes.compressed")"

    if [ -z "$warp_fresh_segments" ]; then
        warp_fresh_segments="$(last_log_number_after_colon "$log_path" "WARP fresh segment source count")"
    fi
    if [ -z "$warp_root_steps" ]; then
        warp_root_steps="$(last_log_number_after_colon "$log_path" "WARP root step count")"
    fi
    if [ -z "$proof_size_bytes" ]; then
        proof_size_bytes="$(log_proof_size "$log_path" raw)"
    fi
    if [ -z "$proof_size_compressed" ]; then
        proof_size_compressed="$(log_proof_size "$log_path" compressed)"
    fi
    if [ -z "$benchmark_prove_time_ms" ]; then
        benchmark_prove_time_ms="$(last_log_value_after_colon "$log_path" "Benchmark Prove Time (ms)")"
    fi
    if [ -z "$benchmark_setup_time_ms" ]; then
        benchmark_setup_time_ms="$(last_log_value_after_colon "$log_path" "Benchmark Setup Time (ms)")"
    fi
    if [ -z "$benchmark_verify_time_ms" ]; then
        benchmark_verify_time_ms="$(last_log_value_after_colon "$log_path" "Benchmark Verify Time (ms)")"
    fi
    if [ -z "$benchmark_proof_serialize_time_ms" ]; then
        benchmark_proof_serialize_time_ms="$(last_log_value_after_colon "$log_path" "Benchmark Proof Serialize Time (ms)")"
    fi

    csv_append \
        "$timestamp" \
        "$bench" \
        "$mode" \
        "$lifecycle" \
        "$status" \
        "$exit_code" \
        "$elapsed_seconds" \
        "${peak_rss_kib:-}" \
        "${peak_swap_kib:-}" \
        "${benchmark_setup_time_ms:-}" \
        "${benchmark_prove_time_ms:-}" \
        "${benchmark_verify_time_ms:-}" \
        "${benchmark_verify_p95_ms:-}" \
        "${benchmark_verify_first_ms:-}" \
        "${benchmark_verify_attribution_total_ms:-}" \
        "${benchmark_proof_serialize_time_ms:-}" \
        "${segments_from_log:-}" \
        "${warp_fresh_segments:-}" \
        "${warp_shape_count:-}" \
        "${warp_invocations:-}" \
        "${warp_root_steps:-}" \
        "${warp_terminal_whir:-}" \
        "${instructions_executed:-}" \
        "${proof_size_bytes:-}" \
        "${proof_size_compressed:-}" \
        "${native_reduction_bytes:-}" \
        "${warp_root_proof_bytes:-}" \
        "${terminal_whir_bytes:-}" \
        "${warp_planning_ms:-}" \
        "${warp_main_trace_commit_ms:-}" \
        "${warp_native_reduction_ms:-}" \
        "${warp_segment_execute_ms:-}" \
        "${warp_reduction_replay_ms:-}" \
        "${warp_group_other_ms:-}" \
        "${warp_batch_other_ms:-}" \
        "${warp_history_finalize_ms:-}" \
        "${warp_completion_other_ms:-}" \
        "${warp_unattributed_ms:-}" \
        "${warp_attribution_ratio:-}" \
        "${warp_vacc_ms:-}" \
        "${warp_history_setup_keygen_ms:-}" \
        "${warp_history_keygen_ms:-}" \
        "${warp_history_prove_ms:-}" \
        "${warp_terminal_linearizer_ms:-}" \
        "${warp_terminal_whir_ms:-}" \
        "${history_certificate_bytes:-}" \
        "${terminal_opened_rows_bytes:-}" \
        "${terminal_multiproof_bytes:-}" \
        "${warp_verify_history_ms:-}" \
        "${warp_verify_terminal_whir_ms:-}" \
        "${warp_verify_linearizer_ms:-}" \
        "$metrics_path" \
        "$log_path"

    echo "==> completed $bench / $mode / $lifecycle: status=$status elapsed=${elapsed_seconds}s"
}

echo "Writing benchmark artifacts to: $OUT_DIR"
echo "Summary CSV: $SUMMARY_CSV"
echo "Benches: $BENCHES_STR"
echo "Modes: $MODES_STR"
echo "Lifecycles: $LIFECYCLES_STR"

for bench in $BENCHES_STR; do
    for lifecycle in $LIFECYCLES_STR; do
        for mode in $MODES_STR; do
            run_one "$bench" "$mode" "$lifecycle" \
                || echo "warning: failed to record $bench / $mode / $lifecycle" >&2
        done
    done
done

echo
echo "Summary:"
column -s, -t "$SUMMARY_CSV" 2>/dev/null || cat "$SUMMARY_CSV"
echo
echo "Wrote: $SUMMARY_CSV"
