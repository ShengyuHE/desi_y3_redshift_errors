#!/usr/bin/env bash

set -euo pipefail

SOURCE_ROOT="/global/cfs/cdirs/desi/science/cai/desi-clustering/dr2/summary_statistics/full_shape/data_splits/holi-v3-altmtl"
DEST_ROOT="/pscratch/sd/s/shengyu/galaxies/catalogs/Y3/holi-v3/altmtl"
SKIP_FILE="/global/homes/s/shengyu/Y3/desi_y3_redshift_errors/main/clustering/dubious_holi-v3-altmtl.txt"

DRY_RUN=0
OVERWRITE=0

usage() {
    cat <<EOF
Usage: $(basename "$0") [--dry-run] [--overwrite]

Copy saved holi-v3 QSO, LRG, and ELG mesh3 Sugiyama NGC/SGC measurements into
the local altmtl mpspk directories, translating filenames to the local convention.

QSO redshift bin:  z0.8-2.1
LRG redshift bins: z0.4-0.6 z0.6-0.8 z0.8-1.1
ELG redshift bins: z0.8-1.1 z1.1-1.6

Options:
  --dry-run    Print copies that would be performed without writing files.
  --overwrite  Replace target files that already exist.
  -h, --help   Show this help message.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            ;;
        --overwrite)
            OVERWRITE=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

if [[ ! -d "$SOURCE_ROOT" ]]; then
    echo "Source directory does not exist: $SOURCE_ROOT" >&2
    exit 1
fi
for tracer in QSO LRG ELG; do
    if [[ ! -d "$DEST_ROOT/$tracer" ]]; then
        echo "$tracer destination root does not exist: $DEST_ROOT/$tracer" >&2
        exit 1
    fi
done
if [[ ! -f "$SKIP_FILE" ]]; then
    echo "Skip list does not exist: $SKIP_FILE" >&2
    exit 1
fi

declare -A skip_ids=()
while IFS= read -r line || [[ -n "$line" ]]; do
    id="${line%%#*}"
    id="${id//[[:space:]]/}"
    [[ -z "$id" ]] && continue
    if [[ ! "$id" =~ ^[0-9]+$ ]]; then
        echo "Invalid mock ID in $SKIP_FILE: $line" >&2
        exit 1
    fi
    skip_ids["$id"]=1
done < "$SKIP_FILE"

copied=0
existing=0
skipped=0
missing=0

for source_dir in "$SOURCE_ROOT"/mock*; do
    [[ -d "$source_dir" ]] || continue
    mock_name="${source_dir##*/}"
    mock_id="${mock_name#mock}"
    [[ "$mock_id" =~ ^[0-9]+$ ]] || continue

    if [[ -v "skip_ids[$mock_id]" ]]; then
        ((skipped += 1))
        continue
    fi

    for tracer in QSO LRG ELG; do
        case "$tracer" in
            QSO)
                source_tracer="QSO"
                redshift_bins=("z0.8-2.1")
                ;;
            LRG)
                source_tracer="LRG"
                redshift_bins=("z0.4-0.6" "z0.6-0.8" "z0.8-1.1")
                ;;
            ELG)
                source_tracer="ELG_LOPnotqso"
                redshift_bins=("z0.8-1.1" "z1.1-1.6")
                ;;
        esac
        dest_dir="$DEST_ROOT/$tracer/$mock_name/mpspk"

        for zbin in "${redshift_bins[@]}"; do
            for region in NGC SGC; do
                source_file="$source_dir/mesh3_spectrum_sugiyama-diagonal_poles_${source_tracer}_${zbin}_${region}_weight-default-FKP.h5"
                dest_file="$dest_dir/mesh3_spectrum_poles_sugiyama_${tracer}_${zbin}_${region}_holi_v3.h5"

                if [[ ! -f "$source_file" ]]; then
                    echo "Missing source: $source_file" >&2
                    ((missing += 1))
                    continue
                fi

                if [[ -f "$dest_file" && "$OVERWRITE" -eq 0 ]]; then
                    ((existing += 1))
                    continue
                fi

                if [[ "$DRY_RUN" -eq 1 ]]; then
                    printf 'Would copy %s -> %s\n' "$source_file" "$dest_file"
                else
                    mkdir -p "$dest_dir"
                    cp -p "$source_file" "$dest_file"
                fi
                ((copied += 1))
            done
        done
    done
done

if [[ "$DRY_RUN" -eq 1 ]]; then
    action="would copy"
else
    action="copied"
fi
printf 'QSO/LRG/ELG NGC/SGC: %s %d files; skipped %d dubious mocks; kept %d existing files; missing %d source files.\n' \
    "$action" "$copied" "$skipped" "$existing" "$missing"

if [[ "$missing" -gt 0 ]]; then
    echo "Warning: source products counted as missing above were not copied." >&2
fi
