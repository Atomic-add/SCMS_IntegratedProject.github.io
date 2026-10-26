# Parallel sim runs per batch (--workers). Every run renders the 30 Hz camera, so one process per core
# (20 on a laptop) takes all CPU, GPU and memory and freezes the machine; 4 keeps it usable.
DEFAULT_WORKERS = 4
