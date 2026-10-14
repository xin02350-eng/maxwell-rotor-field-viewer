"""Finite remaining regional/quantity controls; no test or all60 fitting."""
import os
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[name]="2"
from pathlib import Path
from threadpoolctl import threadpool_limits
import prepare_cycles
import search_cycles
import search_blocks

HERE=Path(__file__).resolve().parent
with threadpool_limits(limits=2):
    for region,magnitude in (("rotor",True),("stator",False),("airgap",False),("stator",True)):
        suffix=region+("_magnitude" if magnitude else "")
        prepare_cycles.main(region,magnitude)
        search_cycles.run(HERE/("prepared_"+suffix),HERE/(suffix+"_cycle_coarse"))
    search_blocks.main("rotor",HERE/"rotor_vector_blocks6",HERE/"rotor_cycle_coarse/selection.json",6,False)
