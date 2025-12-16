┌──────────────────────────────────────────────────────────────┐
│                         START (Drug X)                        │
└───────────────┬──────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 1a: Deep Research (LLM)                                 │
│  - Load Word prompt1a                                          │
│  - Query ChatGPT (deep-research) with drug name + prompt1                     │
│  - Save LLM output as PDF                                    │
   - Output - DRUG_NAME.pdf
└───────────────┬──────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 1b: Extract Pathway Table from PDF (Currently manual)                     │                                                 │
│  - Locate "Pathway Table" section                            │
│  - Extract/paste rows + columns                                    │
│  - Save as Excel (.xlsx)                                     │
    - Output - DRUG_NAME.xlsx
└───────────────┬──────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 2a: Pathway Table Correction (LLM)                              │                                      │
│  - Send table + correction prompt2a to LLM                     │
│  - Receive corrected table                                   │
│  - Save corrected table (CSV/XLSX)                           │
   - Output - DRUG_NAME_ADMINISTRATION_CORRECTED.xlsx
└───────────────┬──────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 2b: Pathway Mapping to MsigDB (Total Manual work right now, needs automation)
│  - Normalize pathway names based on the rationale                                    │
   - Remove the pathways which are repeated in same context
   - Remove the pathways to which are inverse of each other
   - Remove/tag the pathways which are inferred/contextual/non-validated
│  - Requirement: Map to exact MsigDB gene set based on                   │
│        • rationale text                                      │
│        • synonyms/databases                                  │
└───────────────┬──────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 3a: JSON Output Generation (LLM)                        │
│  - Provide drug name + mapped pathways + prompt3a + DRUG_NAME.pdf        │
│  - LLM returns JSON                                           │
│  - Output: DRUG_NAME.json                                         │
└───────────────┬──────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 3b: 8 combinations (before/after, resistance/sensitive, up/down regulation) generated using prompt3b Tabular Output (LLM)
│  - Provide drug name + mapped pathways + prompt3b                   │
│  - Receive structured table                                   │
│  - Output: DRUG_NAME_administrated_combinations.csv                                   │
└───────────────┬──────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 3c: Final JSON Schema Assembly (Automated - code inplace)                               │
│  - Convert DRUG_NAME_administed.csv to json and append to DRUG_NAME.json under respective pathway keys.                           │
│  - Output:  DRUG_NAME_final.json                             │
└───────────────┬──────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 4: Fetch MsigDB Gene Sets (Automated - code inplace)                               │
│  - For each mapped pathway                                   │
│  - Fetch gene set data from MsigDB                            │
│  - Save gene sets for further analysis                       │
└───────────────┬──────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────┐
│                         END (Drug X)                         │
└──────────────────────────────────────────────────────────────┘
