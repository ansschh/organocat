#!/usr/bin/env python3
"""Use Claude API to generate mechanism labels for parsed reactions.

Strategy: for each (reactants, products, catalyst) tuple, prompt Claude to
produce a structured mechanism description:
  - reaction_class: e.g. "asymmetric_transfer_hydrogenation", "buchwald_hartwig", etc.
  - elementary_steps: list of "oxidative_addition / migratory_insertion / ..." labels
  - electron_flow: per-step description of bond breaks/forms
  - rate_determining_step: best guess
  - transition_state_geometry: hint at TS (e.g., "6-membered cyclic TS with hydride transfer")

These labels go into the contrastive dataset as auxiliary supervision signals
(predict mechanism class from (reaction, catalyst) → multi-task with
the contrastive loss).

Cost notes:
  Claude-3.5-Sonnet at ~$3/M input + $15/M output tokens.
  Typical prompt 500 tokens, response 300 tokens → ~$5 per 1000 reactions.
  For 10k TM reactions, expect ~$50.
"""
from __future__ import annotations
import json, os, time
from typing import Optional, List, Dict

try:
    from anthropic import Anthropic
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False


MECHANISM_PROMPT = """You are an expert in organometallic catalysis. Analyze the following reaction:

Reactants (SMILES): {reactants}
Products (SMILES):  {products}
Catalyst (SMILES or name): {catalyst}
{extra_info}

Provide a structured analysis in JSON only, no surrounding prose:

{{
  "reaction_class": "<canonical class, e.g. buchwald_hartwig_amination, transfer_hydrogenation, olefin_metathesis, hydroformylation, suzuki_coupling, c_h_activation, etc>",
  "elementary_steps": ["<step1>", "<step2>", ...],
  "key_bond_changes": ["<X-Y forms>", "<A-B breaks>", ...],
  "active_metal_oxidation_states": ["<e.g. Pd(0) -> Pd(II) -> Pd(0)>"],
  "rate_determining_step": "<one of the elementary steps>",
  "ts_geometry_hint": "<short description of the TS, e.g. '6-membered cyclic TS with hydride transfer to C=N face'>",
  "confidence": <0.0 to 1.0, your confidence in this analysis>
}}

If the reaction or catalyst is unclear or non-organometallic, output:
{{"reaction_class": "unknown", "confidence": 0.0}}
"""


def label_one_reaction(client: "Anthropic", reactants_smi: str, products_smi: str,
                       catalyst_smi: str, model: str = "claude-sonnet-4-6",
                       extra_info: str = "") -> Optional[dict]:
    """Generate a mechanism label for one reaction."""
    prompt = MECHANISM_PROMPT.format(
        reactants=reactants_smi, products=products_smi,
        catalyst=catalyst_smi, extra_info=extra_info,
    )
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
        )
        text = resp.content[0].text.strip()
        # Strip code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            if text.startswith("json"):
                text = text[4:].strip()
        return json.loads(text)
    except Exception as e:
        return {"error": str(e), "reaction_class": "parse_error", "confidence": 0.0}


def label_reactions_batch(reactions: List[dict], api_key: str = None,
                          out_path: str = None, model: str = "claude-sonnet-4-6",
                          checkpoint_every: int = 50, max_n: int = None) -> List[dict]:
    """Label a batch of reactions. Checkpoints to disk every N reactions.

    Args:
      reactions: list of dicts with keys reactants/products/catalysts (from ord_parser)
      api_key: Anthropic API key (defaults to env ANTHROPIC_API_KEY)
      out_path: where to checkpoint results
      max_n: limit number of reactions to process
    """
    if not _HAS_ANTHROPIC:
        raise ImportError("pip install anthropic")
    client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    if max_n: reactions = reactions[:max_n]
    out = []
    # Resume from checkpoint if exists
    if out_path and os.path.isfile(out_path):
        with open(out_path) as f:
            out = json.load(f)
        print(f"resuming from {len(out)} previously labeled reactions")
    skip = len(out)

    for i, rxn in enumerate(reactions[skip:], start=skip):
        reactants = ".".join(r.get("smiles", "") for r in rxn.get("reactants", []) if r.get("smiles"))
        products  = ".".join(p.get("smiles", "") for p in rxn.get("products",  []) if p.get("smiles"))
        catalysts = rxn.get("catalysts", [])
        if not catalysts:
            # Try first reactant as catalyst proxy if it contains a TM
            catalyst_smi = ""
            catalyst_name = ""
        else:
            catalyst_smi = catalysts[0].get("smiles", "")
            catalyst_name = catalysts[0].get("name", "")

        if not (reactants and products):
            label = {"reaction_class": "incomplete", "confidence": 0.0}
        else:
            label = label_one_reaction(client, reactants, products,
                                        catalyst_smi or catalyst_name, model=model)
        entry = {"reaction_id": rxn.get("reaction_id"), "label": label}
        out.append(entry)
        if (i + 1) % checkpoint_every == 0:
            if out_path:
                with open(out_path, "w") as f:
                    json.dump(out, f, indent=2)
            print(f"  {i+1}/{len(reactions)}  last_class={label.get('reaction_class')}  saved checkpoint")

    if out_path:
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"saved {len(out)} labels to {out_path}")
    return out


if __name__ == "__main__":
    import argparse, pickle
    ap = argparse.ArgumentParser()
    ap.add_argument("--reactions-pkl", default="data/reactions/ord_tm_reactions.pkl")
    ap.add_argument("--out", default="data/reactions/mechanism_labels.json")
    ap.add_argument("--max-n", type=int, default=100, help="for cost control during testing")
    ap.add_argument("--model", default="claude-sonnet-4-6")
    args = ap.parse_args()

    with open(args.reactions_pkl, "rb") as f:
        rxns = pickle.load(f)
    print(f"loaded {len(rxns)} reactions; labeling up to {args.max_n}")
    label_reactions_batch(rxns, out_path=args.out, model=args.model, max_n=args.max_n)
