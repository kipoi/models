from kipoiseq.dataloaders import SingleVariantUTRDataLoader

class SingleVariantFramepoolDataloader(SingleVariantUTRDataLoader):
    def __init__(
        self,
        gtf_file,
        fasta_file,
        vcf_file,
        feature_type="5UTR",
        vcf_file_tbi=None,
        infer_from_cds=False,
        on_error_warn=True,
        **kwargs
    ):
        super().__init__(
            gtf_file=gtf_file,
            fasta_file=fasta_file,
            vcf_file=vcf_file,
            feature_type=feature_type,
            vcf_file_tbi=vcf_file_tbi,
            infer_from_cds=infer_from_cds,
            on_error_warn=on_error_warn,
            **kwargs
        )
