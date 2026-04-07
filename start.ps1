Set-Location $PSScriptRoot
cls


# Park et al. 2021 (MARL):
#   Po treningu domyślnie: park_population.png + park_population.csv (--no-plot / --no-csv wyłącza)
#   --plot-live = okno matplotlib z krzywymi populacji w trakcie iteracji
#   --viz = okno Pygame (siatka) podczas rolloutu
# python train_park.py --smoke
# python train_park.py --grid 30 --iters 10 --horizon 50 --plot-live --viz --viz-delay 0.01
# python train_park.py --grid 30 --iters 5 --horizon 50 --viz --viz-delay 0.01

# Fig. 3–5 jak w paperze (populacja vs krok środowiska): python reproduce_paper_figures.py --smoke
# Pełny eksperyment (wolno): python reproduce_paper_figures.py --out-dir picks

python reproduce_paper_figures.py --out-dir picks --eval-steps 2000 --train-iters 30 --seeds 0 1 2

# Random moves (bez RL): python watch_park.py --grid 20


# python main.py
# python main.py --width 100 --height 100

# py -3.13 main.py