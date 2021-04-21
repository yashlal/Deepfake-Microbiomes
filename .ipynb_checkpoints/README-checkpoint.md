# ProbioticDesign

Predict invasion of microorganisms into a stable community using the model of TOWARDS A MATHEMATICAL UNDERSTANDING OF COLONIZATION RESISTANCE (Gjini and Madec, 2020).

Generate pairwise interactions via "observation" of equilibrium relative abundance in pairwise growth experiments. We use the predictions of pairwise metabolic models as "observation".

Must provide excel file with sheet labeled "Relative_Abundance" which contains these observations.

This sheet should have rows \& columns corresponding to taxa, and include all taxa present in the communities, if possible. The module will attempt to match the taxa names from the community data file (miceData) to these labels, and will return the number found and the proportion of reads covered by those found.

The module will predict "experiments" given a community and an invader, attempting to match the names in the community with the taxa in the observation file.

Community data should be given as a .csv file whose columns correspond to samples/communities and rows correspond to taxa, and entries corresponding to read counts/relative abundances.

TO DO:
All in one script for user read files.