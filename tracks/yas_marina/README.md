# Yas Marina Track Data

This directory stores the lightweight processed Yas Marina artifacts derived
from `TUMFTM/racetrack-database`. Do not vendor the full external database into
this repository.

Use:

```bash
python3 scripts/download_track_data.py --help
python3 scripts/process_track.py --tumftm-root /path/to/racetrack-database
```

Processed files:

```text
processed/yas_marina.csv
processed/yas_marina_raceline.csv
processed/yas_marina_metadata.yaml
```

Level-1 curbs are flat semantic full-edge zones in `curbs.yaml`. They define
curb usage penalties but do not alter physics. Track limits come from the
TUMFTM left/right widths.
