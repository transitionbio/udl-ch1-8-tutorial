"""Tiny helpers for building the Colab notebook as a list of cells."""
from textwrap import dedent
import nbformat as nbf

CELLS = []
SOLUTIONS_FIRST = False   # test mode: put solutions before their checks


def _clean(s):
    return dedent(s).strip("\n")


def md(s):
    CELLS.append(nbf.v4.new_markdown_cell(_clean(s)))


def code(s, form=False):
    c = nbf.v4.new_code_cell(_clean(s))
    if form:
        c.metadata["cellView"] = "form"
    CELLS.append(c)


def _sol_cell(pid, s):
    body = (f"#@title 🔑 Solution {pid} — double-click to peek (have a go first!)\n"
            f"peek('{pid}')\n" + _clean(s))
    c = nbf.v4.new_code_cell(body)
    c.metadata["cellView"] = "form"
    return c


KEEP = None            # None = every puzzle; otherwise the set of puzzle ids on the core path
ANSWER_ONLY = {"1.1", "1.2", "3.2", "3.4", "4.3", "6.2", "8.1", "8.4", "8.5"}   # nothing downstream depends on these


def _drop_prompt(pid):
    """Remove the markdown prompt for a puzzle that's not on the core path."""
    import re
    head = (f"### 🏆 Challenge {pid[1:]}" if pid.startswith("C") else f"### 🧩 Puzzle {pid} ")
    for c in reversed(CELLS[-3:]):
        if c.cell_type == "markdown" and head in c.source:
            c.source = c.source[:c.source.index(head)].rstrip()
            if not c.source:
                CELLS.remove(c)
            return


def exercise(pid, stub, check, solution):
    """stub cell (student) -> check cell -> hidden solution cell."""
    if KEEP is not None and pid not in KEEP:
        _drop_prompt(pid)
        if pid.startswith("C") or pid in ANSWER_ONLY:
            return
        import re
        names = ", ".join(f"`{n}`" for n in re.findall(r"^def (\w+)", _clean(solution), re.M)) or "helper"
        c = nbf.v4.new_code_cell(f"#@title ⚙️ Ready-made helper: {names} (just run me)\n" + _clean(solution))
        c.metadata["cellView"] = "form"
        CELLS.append(c)
        return
    stub_c = nbf.v4.new_code_cell(_clean(stub))
    check_c = nbf.v4.new_code_cell(f"# 🧪 Check puzzle {pid}\n" + _clean(check))
    sol_c = _sol_cell(pid, solution)
    if SOLUTIONS_FIRST:
        CELLS.extend([stub_c, sol_c, check_c])
    else:
        CELLS.extend([stub_c, check_c, sol_c])


def build(path, title):
    nb = nbf.v4.new_notebook()
    nb.cells = CELLS
    nb.metadata = {
        "colab": {"provenance": [], "toc_visible": True, "name": title},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    }
    nbf.write(nb, path)

SESSION = 1
