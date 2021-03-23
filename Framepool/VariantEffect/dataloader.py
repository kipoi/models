from kipoiseq.dataloaders import SingleVariantUTRDataLoader

class SingleVariantFramepoolDataloader(SingleVariantUTRDataLoader):
    def __init__(self, gtf_file, fasta_file, vcf_file, feature_type, vcf_file_tbi=None, infer_from_cds=False, on_error_warn=True, **kwargs):
        # please specify your "feature_type" as given in the GTF file which you are using!
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
