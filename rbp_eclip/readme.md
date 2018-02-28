## Predictive models for RBP's

112 eCLIP models from: [https://doi.org/10.1093/bioinformatics/btx727](Avsec et al, Bioinformatics 2017)

Associated code is located at: [github.com/gagneurlab/Manuscript_Avsec_Bioinformatics_2017](https://github.com/gagneurlab/Manuscript_Avsec_Bioinformatics_2017)

### Folder structure

- folder name represents the protein name of the RNA-binding protein (RBP)
- yaml/py code resides in: `template/`
- each RBP has a different model (residing in `<rbp>/model_files/model.h5`) and a different transformer for the relative distance (residing in `<rbp>/dataloader_files/position_transformer.pkl`)
- everyhing else is softlinked from `template/` to the individual `<rbp>` folders
