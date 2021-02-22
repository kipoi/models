from kipoiseq.dataloaders import SingleVariantUTRDataLoader

class SingleVariantFramepoolDataloader(SingleVariantUTRDataLoader):
    def __init__(self, gtf_file, fasta_file, vcf_file, vcf_file_tbi=None, infer_from_cds=False, on_error_warn=True, **kwargs):
        kwargs["feature_type"] = "5UTR"
        super().__init__(
            gtf_file=gtf_file,
            fasta_file=fasta_file,
            vcf_file=vcf_file,
            vcf_file_tbi=vcf_file_tbi,
            infer_from_cds=infer_from_cds,
            on_error_warn=on_error_warn,
            **kwargs
        )
