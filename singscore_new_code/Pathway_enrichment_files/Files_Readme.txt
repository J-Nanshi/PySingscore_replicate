Step1a:NEEDS AUTOMATION (Semi-manual)
prompt1a_updated_drug_effect - prompt to be given to LLM(chatgpt) for deepsearch along with <DRUG_NAME>
Output: - <Drug_name>.pdf - LLM output saved in a pdf.

Step1b:NEEDS AUTOMATION (Semi-manual)
Deepsearch_raw_dostralimab - A table "Pathway evidence table" mostly or in generale a table the end of the pdf pasted into an excel for better understanding.

Step2a:
Input - NEEDS AUTOMATION (Semi-manual)
1)Deepsearch_raw_dostralimab
2)prompt2a - prompt for fact check.
Output - Deepsearch_administration_corrected - A tablular form where the pathways present in input is subjected for correction based on "baseline cancer/ before drug administration"


Step2b:NEEDS AUTOMATION (manual)
Dostralimab_PATHWAY_MAPPED - Pathway coloumn names mapped to MSigDB database based on the rationale. some of the pathways can be elemenated as well.

Step3a:NEEDS AUTOMATION (Semi-manual)
Input - 
1) <DRUG_NAME>.pdf - output of step1a
2) Mapped_pathway_list_<DRUG_NAME>.txt - list of pathway names selected and mapped to the msigdb. 
3) prompt3a - prompt to extract info from pdf and convert to json.
Output - <DRUG_NAME>.json -json containing the information based on prompt schema

Step3b:NEEDS AUTOMATION (Semi-manual)
Input - 
1) prompt3b_before_after_adminstration_matrix - prompt to have a tabular format of output which gives 8 rows per pathway for a given drug.
2) Mapped_pathway_list_<DRUG_NAME>.txt
Output - DRUG_NAME_administrated_combinations - a table with (n,5) dimension where n = number of pathway mapped x 8.

Step3c: AUTOMATED (code in place)
Input - 
1) <DRUG_NAME>.json - output of step3a
2) DRUG_NAME_administrated_combinations - output of step3b 
Otuput - <DRUG_NAME>_final.json - here its another drug paclitaxel file for just an example of the json schema.


