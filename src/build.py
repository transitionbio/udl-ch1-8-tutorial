"""python build.py OUTDIR [test|student] [hour|series]
   hour (default): OUTDIR/TBio_UDL_Ch1-8_OneHour.ipynb   series: OUTDIR/TBio_UDL_S{1,2,3}_*.ipynb"""
import sys, os, runpy
import nbh
PARTS = {1: ["p1_p2", "p3", "p4"], 2: ["p5", "p6"], 3: ["p7", "p8", "p9"]}
NAMES = {1: "TBio_UDL_S1_Models_Ch1-4", 2: "TBio_UDL_S2_Losses_Training_Ch5-6", 3: "TBio_UDL_S3_Backprop_Performance_Ch7-8"}
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
nbh.SOLUTIONS_FIRST = len(sys.argv) > 2 and sys.argv[2] == "test"
here = os.path.dirname(os.path.abspath(__file__))
CORE = {"2.1", "3.1", "C3", "4.1", "5.3", "5.4", "5.6", "6.4", "C6", "7.1", "8.1", "8.4"}
mode = sys.argv[3] if len(sys.argv) > 3 else "hour"
if mode == "hour":
    nbh.CELLS.clear(); nbh.SESSION = 0; nbh.KEEP = CORE
    for m in ["p0_intro", "p1_p2", "p3", "p4", "p5", "p6", "p7", "p8", "p_end"]:
        runpy.run_path(os.path.join(here, m + ".py"))
    p = os.path.join(out, "TBio_UDL_Ch1-8_OneHour.ipynb")
    nbh.build(p, "TBio_UDL_Ch1-8_OneHour.ipynb"); print(p, len(nbh.CELLS), "cells")
    sys.exit()
nbh.KEEP = None
for s in (1, 2, 3):
    nbh.CELLS.clear(); nbh.SESSION = s
    for m in ["p0_intro", "p_carry"] + PARTS[s] + ["p_end"]:
        runpy.run_path(os.path.join(here, m + ".py"))
    p = os.path.join(out, NAMES[s] + ".ipynb")
    nbh.build(p, NAMES[s] + ".ipynb"); print(p, len(nbh.CELLS), "cells")
