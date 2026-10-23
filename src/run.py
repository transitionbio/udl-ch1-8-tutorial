import sys, time, nbformat
from nbclient import NotebookClient
nb = nbformat.read(sys.argv[1], 4)
t=time.time()
c = NotebookClient(nb, timeout=900, kernel_name="python3", allow_errors=True)
c.execute()
nbformat.write(nb, sys.argv[2])
print("exec time", round(time.time()-t,1))
for i, cell in enumerate(nb.cells):
    if cell.cell_type != "code": continue
    for o in cell.get("outputs", []):
        if o.output_type == "error":
            print("ERROR cell", i, o.ename, o.evalue[:300]); print(cell.source[:200])
        if o.output_type == "stream":
            for line in o.text.splitlines():
                if any(s in line for s in ["Puzzle", "❌", "💥", "🕳", "MSE", "steps", "accuracy", "recovered", "simulated", "bias²", "naive", "error", "joints", "regions", "Score", "GD", "start", "classical", "trained"]):
                    print(f"[{i}]", line[:220])
