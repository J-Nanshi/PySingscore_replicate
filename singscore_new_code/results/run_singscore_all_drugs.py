import os
import json
import inspect
import pandas as pd
from tqdm import tqdm
from singscore.singscore import score, permutate, empiricalpval

# -----------------------------
# 1) Load expression (UK65)
# -----------------------------
uk65 = pd.read_csv(
    "singscore_new_code/UK65_clean_gene_log_tpm.txt",
    header="infer",
    sep="\t"
)
uk65 = uk65.set_index("GeneName")  # rows = genes, cols = samples

# -----------------------------
# 2) Load the "master" geneset JSON
# -----------------------------
geneset_json_path = "singscore_new_code/results/Deep_search_results/master_drug_pathways_genesets.json"
all_drugs = json.load(open(geneset_json_path, "r", encoding="utf-8"))

# Output config
N_PERM = 10000
out_path = "singscore_new_code/results/Deep_search_results/singscore_results_all_drugs.json"

# -----------------------------
# 3) Robust permutation call (handles different permutate() signatures)
# -----------------------------
def run_permutations(sample_df, n_up, n_down=False, n_perm=10000):
    """
    Calls permutate() with whatever signature your singscore version exposes.
    Tries common parameter names for #permutations.
    """
    sig = inspect.signature(permutate)
    kwargs = {}

    # Some versions use: n_perm / n_permutations / permutations / n
    for cand in ["n_perm", "n_permutations", "permutations", "n"]:
        if cand in sig.parameters:
            kwargs[cand] = n_perm
            break

    # n_up / n_down are present in most versions
    if "n_up" in sig.parameters:
        kwargs["n_up"] = n_up
    if "n_down" in sig.parameters:
        kwargs["n_down"] = n_down

    # Some versions just accept positional (sample, n_up, n_down)
    try:
        return permutate(sample_df, **kwargs)
    except TypeError:
        return permutate(sample_df, n_up=n_up, n_down=n_down)

# -----------------------------
# 4) Compute singscore + empirical p-value for EVERY pathway in JSON
#     - Keeps SAME JSON nesting keys
#     - Each pathway becomes:
#         {
#           "total_score": <float or null>,
#           "empirical_p_value": <float or null>
#         }
# -----------------------------
final_result = {}

for drug_name, drug_block in tqdm(all_drugs.items(), desc="Drugs"):
    final_result[drug_name] = {}

    # drug_block contains categories like:
    # pathways_sensitive_upregulated, pathway_sensitive_downregulated, pathway_resistant_upregulated, ...
    for category_name, category_block in drug_block.items():
        # Preserve empty dicts as empty dicts
        if not isinstance(category_block, dict) or len(category_block) == 0:
            final_result[drug_name][category_name] = {}
            continue

        final_result[drug_name][category_name] = {}

        for pathway_name, genes in tqdm(
            category_block.items(),
            desc=f"{drug_name} :: {category_name}",
            leave=False
        ):
            # genes is a list of symbols
            if not isinstance(genes, list) or len(genes) == 0:
                final_result[drug_name][category_name][pathway_name] = {
                    "total_score": None,
                    "empirical_p_value": None
                }
                continue

            # Filter genes to those present in expression matrix
            up_present = [g for g in genes if g in uk65.index]

            # If nothing overlaps, we can’t compute a meaningful score/p-value
            if len(up_present) == 0:
                final_result[drug_name][category_name][pathway_name] = {
                    "total_score": None,
                    "empirical_p_value": None
                }
                continue

            # Compute singscore (UP only)
            scored = score(
                up_gene=up_present,
                down_gene=False,
                sample=uk65,
                norm_method="theoretical",
                full_data=True
            )
            total_score = float(scored["total_score"].to_list()[0])

            # Permutations + empirical p-value
            permd = run_permutations(
                uk65,
                n_up=len(up_present),
                n_down=False,
                n_perm=N_PERM
            )
            pvals = empiricalpval(permutations=permd, score=scored)
            empirical_p_value = float(pvals["empirical p value"].to_list()[0])

            # Store ONLY the 2 entries per pathway
            final_result[drug_name][category_name][pathway_name] = {
                "total_score": total_score,
                "empirical_p_value": empirical_p_value
            }

# -----------------------------
# 5) Save JSON (same nesting keys)
# -----------------------------
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(final_result, f, indent=2, ensure_ascii=False)

print(f"Saved: {out_path}")
