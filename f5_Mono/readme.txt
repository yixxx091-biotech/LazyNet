1. use origdata.rds and origprocess.R to create the sampling data trainset_bigc_7_053024.

2. use trainset_bigc_7_053024 go through preprocess.py to get trainset140_bigc_quad_7_053024 and trainset_Seurat.csv.

3. use trainset140_bigc_quad_7_053024 to train the LazyNet1.py to get the model.

4. the model use eval.py and to generate the prediction bigc_trainset140_053024_clamp_quad_pred_700dp.csv.

5. the origdata.rds, trianset_Seurat and  bigc_trainset140_053024_clamp_quad_pred_700dp.csv goes to figure.R to generate the figures and P56vs1234genes.txt.
