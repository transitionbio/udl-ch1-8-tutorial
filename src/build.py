"""python build.py OUTDIR [test|student]   ->   OUTDIR/TBio_UDL_Ch1-8_OneHour.ipynb"""
import sys, os, runpy
import nbh
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
nbh.SOLUTIONS_FIRST = len(sys.argv) > 2 and sys.argv[2] == "test"
here = os.path.dirname(os.path.abspath(__file__))
CORE = {"2.1", "3.1", "C3", "4.1", "5.3", "5.4", "5.6", "6.4", "C6", "7.1", "8.1", "8.4"}
nbh.CELLS.clear(); nbh.KEEP = CORE
for m in ["p0_intro", "p1_p2", "p3", "p4", "p5", "p6", "p7", "p8", "p_end"]:
    runpy.run_path(os.path.join(here, m + ".py"))
p = os.path.join(out, "TBio_UDL_Ch1-8_OneHour.ipynb")
nbh.build(p, "TBio_UDL_Ch1-8_OneHour.ipynb"); print(p, len(nbh.CELLS), "cells")
