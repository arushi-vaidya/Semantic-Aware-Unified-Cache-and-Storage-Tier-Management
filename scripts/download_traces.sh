#!/usr/bin/env bash
# Download the 14 public block traces used in the paper (~430 MB compressed, ~3 GB decompressed)
# from CMU PDL's libCacheSim mirror and decompress them into data/traces/bin/<name>.bin.
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=https://ftp.pdl.cmu.edu/pub/datasets/twemcacheWorkload/cacheDatasets
mkdir -p data/traces/raw data/traces/bin
FILES=(
  msr/msr_hm_0 msr/msr_prn_0 msr/msr_proj_0 msr/msr_web_2 msr/msr_usr_2 msr/msr_prxy_0
  fiu/fiu_webmail.cs.fiu.edu-110108-113008 fiu/fiu_online.cs.fiu.edu-110108-113008
  fiu/fiu_ikki-110108-112108 fiu/fiu_casa-110108-112108
  fiu/fiu_webusers-030409-033109 fiu/fiu_webresearch-030409-033109
)
for f in "${FILES[@]}"; do
  out="data/traces/raw/$(basename "$f").oracleGeneral.zst"
  [ -f "$out" ] || curl -fsS -o "$out" "$BASE/$f.oracleGeneral.zst" &
done
for w in w22 w31; do
  out="data/traces/raw/$w.oracleGeneral.bin.zst"
  [ -f "$out" ] || curl -fsS -o "$out" "$BASE/cloudphysics/$w.oracleGeneral.bin.zst" &
done
wait
for f in data/traces/raw/*.zst; do
  n=$(basename "$f" | sed 's/\.oracleGeneral.*//; s/\.cs\.fiu\.edu//; s/-[0-9]*-[0-9]*//')
  [ -f "data/traces/bin/$n.bin" ] || zstd -q -d -f "$f" -o "data/traces/bin/$n.bin"
done
ls -lh data/traces/bin
