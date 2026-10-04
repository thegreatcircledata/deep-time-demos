# ep04-coincidence — random dots make "alignments"

The demonstration in the Deep Time Research Institute video *Why Almost Every Ancient "Alignment" Is a Coincidence* (2026).
It is an illustration of a method, not a research result: the dots are random, and nothing here tests any real place.

## What it computes

- **Demo A, random dots make lines.** 300 dots placed uniformly at random on a 100 km × 100 km square map. A *k-dot line* is a
  straight line drawn between two dots (its end points) that passes within 50 m of at least k − 2 other dots lying between
  those ends (a strip 100 m wide). All 44,850 end-point pairs are tried. On the first map: **486** lines pass through four dots
  or more, **28** through five, none through six.
- **Demo B, what random gives.** The same count on 1,000 independent random maps: lines through four or more dots, median
  **571** (range 444–744); through five, median **39** (14–74); **859** of the 1,000 maps have a line through six dots and
  **96** a line through seven. An analytic approximation (independent binomial) gives 577.3 and 39.9 expected.
- **Demo C, how many things were tried.** 10,000 simulated "studies", each running 20 two-sided z-tests (50 samples each) on
  pure noise at the usual 5 % level: 5.02 % of single tests come out "significant", and **64.2 %** of studies find at least
  one "significant" result (textbook value 1 − 0.95²⁰ = 64.15 %).

## Seed

Master seed **20261003** (`numpy.random.SeedSequence`). It spawns one stream for the maps (one child per map; map 0 is the
first map) and one for the studies, so every number above is fixed.

## How to run

```sh
python3 -m venv .venv
.venv/bin/pip install numpy==2.5.3
.venv/bin/python demo_alignments.py
```

The script writes `demo_results.json` and `demo_map_first.json` (the first map's 300 dots and its 486 lines) next to itself
and prints their sha256. It takes a few minutes (1,000 maps × 44,850 pairs). With numpy 2.5.3 (tested on Python 3.14.5) the
output reproduces the `demo_results.json` in this folder byte for byte:

```
demo_results.json sha256 283d2d4a2654ece26d3df0e906c609bfc6031a9c13d99d7855a327a0bdf9bd52
```

The file records the numpy version it was made with, so another numpy version gives the same numbers in a file that is not
byte-identical. (The run note in the script's docstring refers to the video project's own environment; the commands above are
all you need.)

## Files

| file | what |
|---|---|
| `demo_alignments.py` | the demonstration (Demos A–C), fixed seed |
| `demo_results.json` | its results: the first map, the 1,000-map baseline, the analytic check, the noise studies |
| `LICENSE` | MIT |

## Limits

Real ancient sites are not scattered evenly: they cluster along rivers and coasts, so a fair baseline for a real map has to copy
that clustering. These uniform random maps show the idea; they do not test any real place.

## Licence

MIT (see `LICENSE`).
