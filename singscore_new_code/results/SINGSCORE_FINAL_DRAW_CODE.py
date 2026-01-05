
import sys
import os
import json
import re
import inspect
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from singscore_main import score, permutate, empiricalpval
# from common.constants import DEFAULT_DATA_PATHS  constants path issue workaround is to comment this out

def run_singscore_all_drugs(sample, geneset=None, out_path=None, n_perm=10000, norm_method="theoretical"):
    """
    Compute singscore total score and empirical p-value for every pathway in a master geneset.

    Parameters
    - sample: pandas.DataFrame (indexed by gene symbol) or path to expression TSV/CSV.
    - geneset: dict (preloaded) or path to geneset JSON. If None, uses
      DEFAULT_DATA_PATHS['drug_genesets'] from src.common.constants.
    - out_path: optional path to write result JSON. If None, does not write.
    - n_perm: number of permutations for empirical p-value.
    - norm_method: passed to `score()` as `norm_method`.

    Returns
    - final_result: nested dict preserving master geneset keys with
      {"total_score": float|None, "empirical_p_value": float|None}

    Notes: Accepts genesets with structure: drug -> category -> pathway -> [genes]
    """

    def _load_sample(sample_input):
        gene_keywords = ["gene", "genes", "symbol", "symbols", "name", "genename"]
        if isinstance(sample_input, pd.DataFrame):
            df = sample_input.copy()
        else:
            try:
                df = pd.read_csv(sample_input, sep=None, engine="python", header=0)
            except Exception:
                try:
                    df = pd.read_csv(sample_input, header=0)
                except Exception:
                    df = pd.read_csv(sample_input, sep="\t", header=0)

        # find matching column for gene names
        gene_col = next(
            (col for col in df.columns if any(re.search(k, col, re.IGNORECASE) for k in gene_keywords)),
            df.columns[0]
        )
        df = df.set_index(gene_col)
        return df

    def _load_geneset(geneset_input):
        if geneset_input is None:
            path = DEFAULT_DATA_PATHS.get("drug_genesets")
            if path is None:
                raise ValueError("No default drug_genesets path found in DEFAULT_DATA_PATHS")
            geneset_input = path

        if isinstance(geneset_input, dict):
            return geneset_input

        # assume path
        with open(geneset_input, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def run_permutations(sample_df, n_up, n_down=False, n_perm=10000):
        sig = inspect.signature(permutate)
        kwargs = {}

        for cand in ["n_perm", "n_permutations", "permutations", "n"]:
            if cand in sig.parameters:
                kwargs[cand] = n_perm
                break

        if "n_up" in sig.parameters:
            kwargs["n_up"] = n_up
        if "n_down" in sig.parameters:
            kwargs["n_down"] = n_down

        try:
            return permutate(sample_df, **kwargs)
        except TypeError:
            return permutate(sample_df, n_up=n_up, n_down=n_down)

    # load inputs
    expr = _load_sample(sample)
    # print(f"exprestion data: {expr.head()}")
    all_drugs = _load_geneset(geneset)
    # print(f"Loaded geneset with drugs: {list(all_drugs.items())}")


    final_result = {}

    for drug_name, drug_block in tqdm(all_drugs.items(), desc="Drugs"):
        final_result[drug_name] = {}

        for category_name, category_block in drug_block.items():
            if not isinstance(category_block, dict) or len(category_block) == 0:
                final_result[drug_name][category_name] = {}
                continue

            final_result[drug_name][category_name] = {}

            for pathway_name, genes in tqdm(category_block.items(), desc=f"{drug_name} :: {category_name}", leave=False):
                if not isinstance(genes, list) or len(genes) == 0:
                    final_result[drug_name][category_name][pathway_name] = {"total_score": None, "empirical_p_value": None}
                    continue

                up_present = [g for g in genes if g in expr.index]

                if len(up_present) == 0:
                    final_result[drug_name][category_name][pathway_name] = {"total_score": None, "empirical_p_value": None}
                    continue

                scored = score(up_gene=up_present, down_gene=False, sample=expr, norm_method=norm_method, full_data=True)
                total_score = float(scored["total_score"].to_list()[0])

                permd = run_permutations(expr, n_up=len(up_present), n_down=False, n_perm=n_perm)
                pvals = empiricalpval(permutations=permd, score=scored)
                empirical_p_value = float(pvals["empirical p value"].to_list()[0])

                final_result[drug_name][category_name][pathway_name] = {"total_score": total_score, "empirical_p_value": empirical_p_value}

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(final_result, f, indent=2, ensure_ascii=False)

    return final_result

# Example usage (commented)

# 1) Call with file path for sample and default geneset from constants, write output
# result = run_singscore_all_drugs(
#     sample="/path/to/sample_expression.tsv",
#     geneset=DEFAULT_DATA_PATHS['drug_genesets'],
#     out_path="/path/to/output/singscore.json",
#     n_perm=10000
# )


# Example usage implemented
# result = run_singscore_all_drugs(
#     sample="../../../patients/1UK-65-F/preprocessing/transcriptomics/output/1UK-65-F_preprocessed 1.csv",
#     geneset="../../../utils/singscore/master_drug_pathways_genesets.json",
#     out_path="../../../patients/1UK-65-F/inference/singscore/output/singscore.json",
#     n_perm=10000
# )
# print("Completed run_singscore_all_drugs; wrote output:" , "patients/1UK-65-F/inference/singscore/output/singscore.json")
