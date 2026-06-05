#!/usr/bin/env python3
"""RAS-Bench v0 driver — the decisive go/no-go for r -> Z_cat.

For each split (leave-catalyst-out, leave-metal-out, leave-ligand-out):
  1. Train Z_cat-CLIP  (ReactionGNN + ComplexEGNN + projection heads, InfoNCE,
     optional hard negatives).
  2. Evaluate on the held-out test set, all on the SAME (reactions, pool):
       - learned retrieval        : exact + family hits@k, mrr
       - baselines                : random, popularity prior, Morgan-kNN(diff/reactant)
       - hard-negative ranking    : does the model rank the true catalyst above
                                    its chemically-adversarial negatives?
       - latent structure         : does Z_cat cluster by ligand-class/metal
                                    (proxy until mechanism labels land)?
  3. Apply the decision rule and print the table.

Pass requires (per the doc): beat popularity AND Morgan-kNN on family hits@k,
and win the hard-negative ranking clearly above chance.
"""
from __future__ import annotations
import argparse, json, pickle, sys, time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

sys.path.insert(0, __import__("os").environ.get("ORGANOCAT_ROOT", "."))
from src.encoders.reaction_gnn import ReactionGNN
from src.encoders.complex_egnn import ComplexEGNN
from src.contrastive.info_nce import info_nce_loss, retrieval_metrics, ProjectionHead
from src.contrastive.hard_negatives import CatalystBank, HardNegativeSampler
from src.data.splits import make_split, ALL_SPLITS
from src.data.catalyst_descriptors import catalyst_descriptors
from src.data import baselines as B


# ----------------------- catalyst batch reconstruction -----------------------

class PairData(Data):
    """Each pair holds a reaction graph (x/edge_index/edge_attr) AND a catalyst
    graph (cat_*) in one object. PyG auto-increments any key containing 'index'
    by the MAIN graph's node count, which corrupts cat_edge_index/cat_metal_idx
    (they index the catalyst nodes, not the reaction nodes). Override the
    increments so batching offsets them by the catalyst node count instead."""
    def __inc__(self, key, value, *args, **kwargs):
        if key in ("cat_edge_index", "cat_metal_idx"):
            return self.cat_pos.size(0)
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "cat_edge_index":
            return -1
        return super().__cat_dim__(key, value, *args, **kwargs)


def as_pairdata(pairs):
    """Re-class plain Data pairs (as saved on disk) to PairData in place (zero
    copy) so correct batch increments apply."""
    for p in pairs:
        p.__class__ = PairData
    return pairs


def cat_batch_from(combined_batch):
    device = combined_batch.cat_pos.device
    n_per = combined_batch.cat_n_atoms
    n_list = n_per.tolist() if torch.is_tensor(n_per) else list(n_per)
    cb = torch.cat([torch.full((n,), i, dtype=torch.long, device=device)
                    for i, n in enumerate(n_list)])
    cat = Data(pos=combined_batch.cat_pos, z=combined_batch.cat_z,
               charges=combined_batch.cat_charges,
               edge_index=combined_batch.cat_edge_index,
               edge_attr=combined_batch.cat_edge_attr,
               metal_mask=combined_batch.cat_metal_mask)
    cat.batch = cb
    return cat


def representative_cat_data(pairs, idxs):
    """One catalyst Data per unique catalyst SMILES in idxs (for pool/bank
    encoding). Uses STANDARD PyG keys (pos/z/edge_index/...) so these
    catalyst-only graphs batch normally — no cat_* keys fighting PyG."""
    seen = {}
    for i in idxs:
        p = pairs[i]
        s = p.catalyst_smi
        if s not in seen:
            d = Data(
                pos=p.cat_pos, z=p.cat_z, charges=p.cat_charges,
                edge_index=p.cat_edge_index, edge_attr=p.cat_edge_attr,
                metal_mask=p.cat_metal_mask)
            d.num_nodes = p.cat_pos.size(0)
            seen[s] = d
    return seen


# ----------------------------- model bundle -----------------------------

class ZCatModel(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.rxn = ReactionGNN(embed_dim=embed_dim)
        self.cat = ComplexEGNN(embed_dim=embed_dim, update_coords=False)
        self.pr = ProjectionHead(embed_dim, embed_dim)
        self.pc = ProjectionHead(embed_dim, embed_dim)

    def enc_rxn(self, batch):
        return self.pr(self.rxn(batch))

    def enc_cat(self, batch):
        # combined (reaction+catalyst) pair batch: extract the catalyst subgraph
        return self.pc(self.cat(cat_batch_from(batch)))

    def enc_cat_std(self, std_batch):
        # catalyst-only batch already in standard PyG keys (pool/bank)
        return self.pc(self.cat(std_batch))


def encode_pool(model, cat_dict, device, bs=128):
    """Encode each unique catalyst once -> (P, D) tensor + smi list."""
    smis = list(cat_dict.keys())
    datas = [cat_dict[s] for s in smis]
    embs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(datas), bs):
            from torch_geometric.data import Batch
            batch = Batch.from_data_list(datas[i:i + bs]).to(device)
            embs.append(model.enc_cat_std(batch))
    return torch.cat(embs), smis


# ----------------------------- training -----------------------------

def train_split(pairs, train_idx, device, epochs=15, bs=64, lr=3e-4, temp=0.1,
                use_hard_neg=True, hard_k=8, embed_dim=128):
    model = ZCatModel(embed_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    train_ds = [pairs[i] for i in train_idx]
    loader = DataLoader(train_ds, batch_size=bs, shuffle=True)

    bank = sampler = None
    if use_hard_neg:
        train_cat_dict = representative_cat_data(pairs, train_idx)
        bank = CatalystBank(list(train_cat_dict.keys()))
        sampler = HardNegativeSampler(bank, k=hard_k, seed=0)
        bank_datas = [train_cat_dict[s] for s in bank.smis]

    for ep in range(epochs):
        model.train()
        t0 = time.time(); tot = 0.0; n = 0
        # per-epoch memory bank of catalyst embeddings (detached negatives)
        bank_emb = None
        if use_hard_neg:
            from torch_geometric.data import Batch
            embs = []
            model.eval()
            with torch.no_grad():
                for i in range(0, len(bank_datas), 128):
                    b = Batch.from_data_list(bank_datas[i:i + 128]).to(device)
                    embs.append(model.enc_cat_std(b))
            bank_emb = torch.cat(embs)             # (Nbank, D) detached
            model.train()
        for batch in loader:
            batch = batch.to(device)
            z_r = model.enc_rxn(batch)
            z_c = model.enc_cat(batch)
            hn = None
            if use_hard_neg:
                neg_idx = sampler.sample(list(batch.catalyst_smi))   # (B, K)
                hn = bank_emb[neg_idx.to(device)]                    # (B, K, D)
            loss = info_nce_loss(z_r, z_c, temperature=temp, hard_negatives=hn)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * batch.num_graphs; n += batch.num_graphs
        print(f"    epoch {ep+1:2d}/{epochs} loss={tot/max(1,n):.4f} ({time.time()-t0:.0f}s)", flush=True)
    return model


# ----------------------------- evaluation -----------------------------

def eval_split(model, pairs, eval_idx, train_idx, rxn_lookup, device, top_ks=(1, 5, 10),
               max_eval=6000):
    # Cap the eval set so learned + baselines are scored on the SAME bounded
    # sample (leave-metal-out can put tens of thousands of reactions in test;
    # the Morgan-kNN baseline would be far too slow over all of them).
    if len(eval_idx) > max_eval:
        g = torch.Generator().manual_seed(0)
        sel = torch.randperm(len(eval_idx), generator=g)[:max_eval].tolist()
        eval_idx = [eval_idx[i] for i in sel]
        print(f"    [eval capped to {max_eval} of {len(sel)} sampled reactions]", flush=True)
    # Build eval reaction dicts (for baselines + pool) from the SAME pairs.
    eval_rxns = [rxn_lookup[pairs[i].reaction_id] for i in eval_idx]
    train_rxns = [rxn_lookup[pairs[i].reaction_id] for i in train_idx]
    pool, target = B.build_pool(eval_rxns)

    # --- learned retrieval ---
    cat_dict = representative_cat_data(pairs, eval_idx)
    pool_emb, pool_smis = encode_pool(model, cat_dict, device)
    pool_pos = {s: i for i, s in enumerate(pool_smis)}
    # reaction embeddings for eval set
    from torch_geometric.data import Batch
    r_embs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(eval_idx), 128):
            b = Batch.from_data_list([pairs[j] for j in eval_idx[i:i + 128]]).to(device)
            r_embs.append(model.enc_rxn(b))
    r_emb = torch.cat(r_embs)
    # align pool order to baseline pool
    reorder = torch.tensor([pool_pos[s] for s in pool])
    S = (r_emb @ pool_emb.t())[:, reorder].cpu()
    learned = B.retrieval_from_scores(S, target, pool, top_ks)

    # --- baselines ---
    base = B.run_all_baselines(train_rxns, eval_rxns, pool, target, top_ks)

    # --- hard-negative ranking (sharpest test) ---
    hn_acc = hard_negative_ranking(model, pairs, eval_idx, device)

    # --- latent structure (proxy: ligand-class / metal separability) ---
    struct = latent_structure(pool_emb.cpu(), pool_smis)

    return {"learned": learned, "baselines": base,
            "hard_neg_rank_acc": hn_acc, "latent": struct,
            "n_eval": len(eval_rxns), "pool_size": len(pool)}


def hard_negative_ranking(model, pairs, eval_idx, device, k=8):
    """For each eval reaction, does sim(r, true_cat) exceed sim(r, hard_negs)?
    Builds the bank from the EVAL catalysts so negatives are unseen-pool members."""
    cat_dict = representative_cat_data(pairs, eval_idx)
    bank = CatalystBank(list(cat_dict.keys()))
    sampler = HardNegativeSampler(bank, k=k, seed=1)
    pool_emb, pool_smis = encode_pool(model, cat_dict, device)
    pos_idx = {s: i for i, s in enumerate(pool_smis)}
    from torch_geometric.data import Batch
    wins = tot = 0
    model.eval()
    with torch.no_grad():
        for i in range(0, len(eval_idx), 128):
            sub = eval_idx[i:i + 128]
            b = Batch.from_data_list([pairs[j] for j in sub]).to(device)
            zr = model.enc_rxn(b)                      # (B, D)
            smis = list(b.catalyst_smi)
            neg_idx = sampler.sample(smis)             # (B, K) into bank
            for bi, s in enumerate(smis):
                pos = pool_emb[pos_idx[s]]
                negs = pool_emb[[pos_idx[bank.smis[j]] for j in neg_idx[bi].tolist()]]
                sp = (zr[bi] @ pos).item()
                sn = (zr[bi] @ negs.t()).max().item()
                wins += int(sp > sn); tot += 1
    return wins / max(1, tot)


def latent_structure(pool_emb, pool_smis):
    """Silhouette of catalyst embeddings under metal and ligand-class labels.
    Mechanism-clustering is the real target (doc); flagged TODO until labels."""
    try:
        from sklearn.metrics import silhouette_score
    except Exception:
        return {"note": "sklearn unavailable"}
    descs = [catalyst_descriptors(s) for s in pool_smis]
    X = pool_emb.numpy()
    out = {}
    for key in ("metal", "ligand_class"):
        labs = [d[key] if d else "?" for d in descs]
        uniq = sorted(set(labs))
        if 2 <= len(uniq) < len(labs):
            y = np.array([uniq.index(l) for l in labs])
            try:
                out[f"silhouette_{key}"] = float(silhouette_score(X, y))
            except Exception:
                out[f"silhouette_{key}"] = None
    out["note"] = "mechanism-clustering TODO (needs mechanism_labels)"
    return out


# ----------------------------- decision -----------------------------

def decide(res):
    L = res["learned"]; base = res["baselines"]
    pop = base["popularity_prior"]["fam_hits@5"]
    knn = max(base["morgan_knn_diff"]["fam_hits@5"], base["morgan_knn_reactant"]["fam_hits@5"])
    rnd = base["random"]["fam_hits@5"]
    learned = L["fam_hits@5"]
    hn = res["hard_neg_rank_acc"]
    beats_strong = learned > pop + 0.02 and learned > knn + 0.02
    beats_weak = learned > rnd + 0.02
    hard_ok = hn > 0.60
    if beats_strong and hard_ok:
        verdict = "PASS: Z_cat signal exists -> proceed to active-state generation"
    elif beats_strong and not hard_ok:
        verdict = "PARTIAL: beats baselines but fails hard negatives -> representation shallow, fix negatives/data"
    elif beats_weak:
        verdict = "WEAK: beats random only -> mostly metal/popularity prior, improve data/negatives (not protein)"
    else:
        verdict = "FAIL: no signal above random -> pivot representation or get QM/mechanism data"
    return {"verdict": verdict, "learned_fam@5": learned, "popularity_fam@5": pop,
            "morgan_fam@5": knn, "random_fam@5": rnd, "hard_neg_acc": hn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-pkl", default="data/pairs/contrastive_pairs.pt")
    ap.add_argument("--clean-pkl", default="data/reactions/ord_tm_clean.pkl")
    ap.add_argument("--splits", nargs="+", default=ALL_SPLITS)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--no-hard-neg", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-json", default="data/ras_bench_v0.json")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)

    pairs = pickle.load(open(args.pairs_pkl, "rb"))["pairs"]
    if args.limit:
        pairs = pairs[:args.limit]
    as_pairdata(pairs)        # correct catalyst-index increments on batching
    clean = pickle.load(open(args.clean_pkl, "rb"))
    rxn_lookup = {r["reaction_id"]: {"reactants_smi": r["reactants_smi"],
                                     "products_smi": r["products_smi"],
                                     "catalyst_smi": r["catalyst_smi"]} for r in clean}
    print(f"pairs={len(pairs)}  clean_lookup={len(rxn_lookup)}", flush=True)

    report = {}
    for kind in args.splits:
        print(f"\n===== SPLIT: {kind} =====", flush=True)
        tr, va, te = make_split(pairs, kind=kind)
        if not tr or not te:
            print(f"  skip {kind}: empty split"); continue
        model = train_split(pairs, tr, device, epochs=args.epochs,
                             bs=args.batch_size, embed_dim=args.embed_dim,
                             use_hard_neg=not args.no_hard_neg)
        res = eval_split(model, pairs, te, tr, rxn_lookup, device)
        dec = decide(res)
        report[kind] = {"result": res, "decision": dec}
        print(f"  -- {kind} --", flush=True)
        print(f"     learned   : exact@1={res['learned']['hits@1']:.3f} fam@5={res['learned']['fam_hits@5']:.3f} mrr={res['learned']['mrr']:.3f}", flush=True)
        for bn, bm in res["baselines"].items():
            print(f"     {bn:20s}: fam@5={bm['fam_hits@5']:.3f} exact@1={bm['hits@1']:.3f}", flush=True)
        print(f"     hard-neg rank acc: {res['hard_neg_rank_acc']:.3f}", flush=True)
        print(f"     latent: {res['latent']}", flush=True)
        print(f"     VERDICT: {dec['verdict']}", flush=True)

    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nsaved -> {args.out_json}", flush=True)
    print("\n================ DECISION TABLE ================", flush=True)
    for kind, r in report.items():
        d = r["decision"]
        print(f"  {kind:22s} learned_fam@5={d['learned_fam@5']:.3f} "
              f"pop={d['popularity_fam@5']:.3f} morgan={d['morgan_fam@5']:.3f} "
              f"hardneg={d['hard_neg_acc']:.3f}  -> {d['verdict'].split(':')[0]}", flush=True)


if __name__ == "__main__":
    main()
