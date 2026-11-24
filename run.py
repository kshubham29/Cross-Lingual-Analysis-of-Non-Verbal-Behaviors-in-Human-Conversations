"""Run notebooks by number and show their results neatly in the terminal.
  python run.py 24          -> runs notebooks/24_fusion.ipynb
  python run.py 14 17 24    -> runs several, one after another
  python run.py all         -> runs all 27 in order
The results are also saved inside each notebook; charts are also saved as PNG files in charts/<notebook>/."""
import asyncio
import glob
import os
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore")
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())   # avoids a Windows warning

import nbformat
from nbclient import NotebookClient
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

console = Console()


def cell_titles(nb):
    """for every code cell: the first line of the text cell just above it (its description)"""
    titles, last = {}, ""
    for i, c in enumerate(nb.cells):
        if c.cell_type == "markdown":
            last = c.source.strip().splitlines()[0].lstrip("# ").replace("**", "").replace("`", "") if c.source.strip() else ""
        else:
            titles[i] = last
            last = ""
    return titles


def output_text(cell, chart_dir, counter):
    """text of the cell's outputs; every chart is saved as a PNG in chart_dir and its path is shown"""
    import base64
    parts = []
    for out in cell.get("outputs", []):
        kind = out.get("output_type")
        if kind == "stream" and out.get("name") == "stdout":
            parts.append(out["text"].rstrip("\n"))
        elif kind in ("execute_result", "display_data"):
            data = out.get("data", {})
            if "image/png" in data:
                os.makedirs(chart_dir, exist_ok=True)
                counter[0] += 1
                file = os.path.join(chart_dir, f"chart_{counter[0]}.png").replace("\\", "/")
                with open(file, "wb") as f:
                    f.write(base64.b64decode(data["image/png"]))
                parts.append(f"[chart saved: {file}  (Ctrl + click to open)]")
            elif "text/plain" in data and not data["text/plain"].startswith("<Figure size"):
                parts.append(data["text/plain"])
        elif kind == "error":
            parts.append(f"ERROR: {out.get('ename')}: {out.get('evalue')}")
    return "\n".join(p for p in parts if p.strip())


def short(title, n=72):
    return title if len(title) <= n else title[:n].rsplit(" ", 1)[0] + " ..."


def run(path):
    name = os.path.basename(path)[:-6]
    nb = nbformat.read(path, as_version=4)
    # hidden first cell: switch on chart display so even the first chart is captured (removed again before saving)
    nb.cells.insert(0, nbformat.v4.new_code_cell("%matplotlib inline"))
    titles = cell_titles(nb)
    code_cells = [i for i, c in enumerate(nb.cells) if c.cell_type == "code"][1:]
    heading = nb.cells[1].source.strip().splitlines()[0].lstrip("# ") if nb.cells[1].cell_type == "markdown" else name
    console.print()
    console.rule(f"[bold]Notebook {name[:2]}  -  {name[3:]}[/bold]")
    console.print(f"[dim]{heading}[/dim]\n")

    start = time.time()
    status = console.status("starting ...")
    chart_dir, counter = os.path.join("charts", name), [0]

    def on_start(cell, cell_index, **_):
        if cell_index not in code_cells:
            return
        status.update(f"running cell {code_cells.index(cell_index) + 1} of {len(code_cells)}   "
                      f"[dim]({(time.time() - start) / 60:.1f} min so far)[/dim]")

    def on_done(cell, cell_index, **_):
        if cell_index not in code_cells:
            return
        text = output_text(cell, chart_dir, counter)
        if text:
            title = short(titles.get(cell_index) or "Output")
            style = "red" if text.startswith("ERROR") else "cyan"
            console.print(Panel(Text(text), title=title, title_align="left", border_style=style, box=box.ROUNDED))

    client = NotebookClient(nb, timeout=-1, resources={"metadata": {"path": "notebooks"}},
                            on_cell_start=on_start, on_cell_executed=on_done)
    ok, msg = True, ""
    status.start()
    try:
        client.execute(stderr=subprocess.DEVNULL)          # hides the kernel's own warnings
    except Exception as e:
        ok, msg = False, str(e).strip().splitlines()[-1][:120]
    finally:
        status.stop()
    nb.cells.pop(0)                                       # remove the hidden cell again
    nbformat.write(nb, path)
    minutes = (time.time() - start) / 60
    if ok:
        console.print(f"[bold green]OK[/bold green]  {name} finished in {minutes:.1f} min (results saved in the notebook)")
    else:
        console.print(f"[bold red]FAILED[/bold red]  {name} after {minutes:.1f} min: {msg}")
    return name, ok, minutes


args = sys.argv[1:] or ["all"]
if args == ["all"]:
    paths = sorted(glob.glob("notebooks/*.ipynb"))
else:
    paths = []
    for n in args:
        found = glob.glob(f"notebooks/{int(n):02d}_*.ipynb")
        if found:
            paths.append(found[0])
        else:
            console.print(f"[red]No notebook with number {n}[/red]")

summary = [run(p) for p in paths]

if len(summary) > 1:
    t = Table(title="Summary", box=box.SIMPLE_HEAVY)
    t.add_column("Notebook"); t.add_column("Status"); t.add_column("Time", justify="right")
    for name, ok, minutes in summary:
        t.add_row(name, "[green]OK[/green]" if ok else "[red]FAILED[/red]", f"{minutes:.1f} min")
    console.print(t)
