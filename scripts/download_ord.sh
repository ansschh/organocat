#!/usr/bin/env bash
# Download the Open Reaction Database ON A LOGIN NODE (needs internet + git-lfs).
#
#   bash scripts/download_ord.sh [DEST]
#
# Canonical ORD data is distributed as gzipped protobuf (.pb.gz) in the
# open-reaction-database/ord-data repo via git-lfs (~several GB). Our parser
# (src/data/ord_parser.py) reads .pb.gz directly; it also handles .parquet if a
# parquet export is used instead.
set -euo pipefail

DEST="${1:-data/raw}"
mkdir -p "$DEST"
cd "$DEST"

if ! command -v git-lfs >/dev/null 2>&1; then
    echo "[download] git-lfs not found. Install it first, e.g.:"
    echo "  curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash"
    echo "  (or: conda install -c conda-forge git-lfs)   then:  git lfs install"
    exit 1
fi
git lfs install

if [ ! -d ord-data ]; then
    echo "[download] cloning ord-data (metadata only first) ..."
    GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/open-reaction-database/ord-data.git
fi
cd ord-data
echo "[download] pulling LFS blobs under data/ (large) ..."
git lfs pull --include "data/**"
echo "[download] done. Files:"
find data -type f \( -name "*.pb.gz" -o -name "*.parquet" \) | head
echo "total: $(find data -type f \( -name '*.pb.gz' -o -name '*.parquet' \) | wc -l) files"
