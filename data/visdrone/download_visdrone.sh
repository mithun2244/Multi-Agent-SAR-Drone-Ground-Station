#!/usr/bin/env bash
#
# Fetch VisDrone2019-DET train and val into data/visdrone/.
#
#   bash data/visdrone/download_visdrone.sh
#
# ~2 GB. Already-extracted splits are left alone, so this is safe to re-run.
#
# The URLs point at the Ultralytics mirror on GitHub Releases, because the
# official VisDrone distribution is on Google Drive and Drive's confirm-token
# dance is not something a wget script should be pretending to do. The dataset
# itself is the same; the authoritative source and citation is
# https://github.com/VisDrone/VisDrone-Dataset — use it if the mirror moves.
set -euo pipefail

MIRROR="${VISDRONE_MIRROR:-https://github.com/ultralytics/assets/releases/download/v0.0.0}"
SPLITS=("VisDrone2019-DET-train" "VisDrone2019-DET-val")
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for tool in wget unzip; do
    command -v "$tool" >/dev/null 2>&1 || {
        echo "error: $tool is not installed" >&2
        exit 1
    }
done

for split in "${SPLITS[@]}"; do
    if [ -d "$DEST/$split" ]; then
        echo "==> $split already extracted, skipping"
        continue
    fi

    archive="$DEST/$split.zip"
    echo "==> fetching $split"
    # -c resumes a partial download: 2 GB over a hotel connection should not
    # start again from zero because the link dropped once.
    wget -c -O "$archive" "$MIRROR/$split.zip"

    # A mirror that has moved answers 200 with an HTML error page, which is not
    # a zip. Checking before extracting turns that into one clear line rather
    # than a wall of unzip errors.
    unzip -tq "$archive" >/dev/null 2>&1 || {
        echo "error: $archive is not a valid zip — the mirror may have moved." >&2
        echo "       Set VISDRONE_MIRROR, or download by hand from" >&2
        echo "       https://github.com/VisDrone/VisDrone-Dataset" >&2
        exit 1
    }

    echo "==> extracting $split"
    unzip -q "$archive" -d "$DEST"
    rm -f "$archive"
done

echo
echo "done. next:"
echo "  python data/visdrone/convert_to_yolo.py"
