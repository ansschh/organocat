# OrganoEnzymeGen — Z_cat-CLIP / RAS-Bench v0 on Caltech HPC

Decisive question (no protein, no RFD2): **given an organometallic reaction, can
a learned representation `Z_cat` retrieve the correct catalyst / active-state
family — beating dumb baselines and separating chemically adversarial hard
negatives?**

Pipeline: `ORD → parse → clean+atom-map → 3D catalyst pairs → Z_cat-CLIP → decision table`.

---

## STEP 0 — (relay to me) cluster diagnostics

Run on a **login node** and paste me the output so I can finalize the SLURM
account/partition and confirm the toolchain:

```bash
sinfo -s ; echo '---' ; sacctmgr -nP show assoc user=$USER format=account,partition ; echo '---' ; command -v conda git-lfs ; echo "scratch=$SCRATCH" ; df -h $HOME | tail -1
```

(Defaults in the sbatch files are `--partition=gpu`; account is commented out —
fill them from this output.)

---

## STEP 1 — get the code on the cluster

On the cluster (login node), from wherever you keep code (prefer scratch/group
space, not `$HOME`, since the data is large):

```bash
git clone https://github.com/ansschh/organocat.git
cd organocat
```

## STEP 2 — build the environment (login node — needs internet)

```bash
bash setup_env.sh          # creates conda env "organocat"; verifies imports
conda activate organocat
```

If any import line at the end prints `FAILED`, paste it to me.

## STEP 3 — download ORD (login node — needs internet, ~several GB)

```bash
bash scripts/download_ord.sh data/raw
# sanity: confirm files + that the parser can read one
python scripts/parse_ord.py --ord-root data/raw/ord-data/data --inspect
```

The `--inspect` line prints the first file's columns. If the full build (STEP 5)
later reports *"could not auto-detect serialized-protobuf column"*, paste me the
`--inspect` output and I'll patch `ord_parser.py`.

## STEP 4 — point the SLURM jobs at your allocation

Edit `slurm/build_dataset.sbatch` and `slurm/run_bench.sbatch`: set
`--partition` and uncomment/set `--account` to match STEP 0.

## STEP 5 — build the dataset (GPU job)

```bash
mkdir -p logs
sbatch slurm/build_dataset.sbatch
squeue --me                       # watch state
tail -f logs/build_*.out          # progress: parse %, atom-map kept, pairs saved
```

Produces `data/reactions/ord_tm_clean.pkl` and `data/pairs/contrastive_pairs.pt`.

## STEP 6 — run the decision table (GPU job)

```bash
sbatch slurm/run_bench.sbatch
tail -f logs/bench_*.out
```

Prints, per split (leave-catalyst / leave-metal / leave-ligand-out):
- learned retrieval (exact + family hits@k, mrr)
- baselines: random · popularity prior · Morgan-kNN(diff/reactant)
- hard-negative ranking accuracy
- **VERDICT**: PASS / PARTIAL / WEAK / FAIL

and writes `data/ras_bench_v0.json`.

---

## What to relay back to me at each step
- STEP 0 output (to finalize SLURM headers)
- any `FAILED` import from STEP 2
- the `--inspect` output **only if** STEP 5 complains about the protobuf column
- the tail of `logs/bench_*.out` (the decision table) when STEP 6 finishes

## Notes / known risk points
- **`setup_env.sh`** (dependency resolution for `rxnmapper` + `torch_geometric`)
  is the most likely thing to need a tweak on a fresh cluster — relay failures.
- **`ord_parser.py`** is reconstructed; it auto-detects format (.pb.gz / .parquet).
  The `--inspect` safety valve covers the parquet-column case.
- tmQM is **not** required for v0 (the catalyst encoder trains from scratch on
  the 3D catalyst graphs). `scripts/pretrain_complex_encoder.py` is optional and
  only helps if you later add tmQM pretraining.
- Compute nodes usually have **no internet** — that's why STEP 2/3 run on a login
  node and STEP 5/6 run via `sbatch`.
