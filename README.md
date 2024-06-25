# galaxy_segmentation

The file `gz2_hart16.csv` contains the galaxyzoo2 catalog where each object has a label according to the categorization used by https://arxiv.org/pdf/1308.3496v2 (Appendix A).

The files `via_region_data.json` under `original/train` and `original/val` are taken from https://github.com/hfarias/mask_galaxy as also most of the dataset python objects in `src/dataset.py`.

The file `galaxy_segment_classes.json` contains the data used for training with the classes derived from galaxyzoo2 and is built with the notebook `data_conversion.ipynb`. 
From an initial aggregation, 15 classes are extracted, but the dataset results too unbalanced so an additional aggregation is performed.
The resulting classes are 5 with the following meaning:
5 categories
- A: star or artifact 
- E: Elliptical, which contains (c)igar-shaped, (i)n-between, (r)ounded
- S: Spiral, whic groups all bulge prominence (‘Sd’ = none, ‘Sc’ = just noticeable, ‘Sb’ = obvious, ‘Sa’ = dominant). 
- SB: Spiral Barred, which groups all barred with bulge prominence (‘SBd’ = none, ‘SBc’ = just noticeable, ‘SBb’ = obvious, ‘SBa’ = dominant).
- Se: Spiral edge-on disks, which groups all edge-on with bulge shape (‘Sen’ = none, ‘Seb’ = boxy, ‘Ser’ = rounded).