# galaxy_segmentation

The file `gz2_hart16.csv` contains the galaxyzoo2 catalog where each object has a label according to the categorization used by https://arxiv.org/pdf/1308.3496v2 (Appendix A).

The files `via_region_data.json` under `original/train` and `original/val` are taken from https://github.com/hfarias/mask_galaxy as also most of the dataset python objects in `src/dataset.py`.

The file `galaxy_segment_classes.json` contains the data used for training with the added classes derived from galaxyzoo2 and is built with the notebook `data_conversion.ipynb`. The resulting classes are 15 with the following meaning:
- A: star or artifact 
- Ec, Ei, Er: Elliptical (c)igar-shaped, (i)n-between, (r)ounded
- Sa, Sb, Sc, Sd: Spiral with bulge prominence  (‘d’ = none, ‘c’ = just noticeable, ‘b’ = obvious, ‘a’ = dominant). 
- SBa, SBb, SBc, SBd: Spiral Barred with bulge prominence (‘d’ = none, ‘c’ = just noticeable, ‘b’ = obvious, ‘a’ = dominant).
- Seb, Sen, Ser: Spiral edge-on disks with bulge shape (‘n’ = none, ‘b’ = boxy, ‘r’ = rounded).