"""
Dataloader
"""
import pickle
from kipoi.data import Dataset
import pandas as pd
import numpy as np
import os
import gffutils

import inspect
this_file_path = os.path.abspath(inspect.getframeinfo(inspect.currentframe()).filename)
this_dir = os.path.dirname(this_file_path)

class IntronsDataset(Dataset):
    """
    Args:
    gtf_file: file path; genome annotation GTF file.
    fasta_file: file path; genome sequence
    """
    
    def __init__(self, gtf_file, fasta_file, bp_idx_file, create_introns = False):
        self.gtf_file = ''.join(["file://", this_dir, os.path.sep, gtf_file])
        self.fasta_file = ''.join([this_dir, os.path.sep, fasta_file])
        with open(''.join([this_dir, os.path.sep, bp_idx_file]), 'r') as bp_file:
            self.bp_indexes = bp_file.readlines()
        self.create_introns = create_introns
        
        # get intronic features
        self.introns = self._get_introns()
        
    def __len__(self):
        return len(self.introns)

    def __getitem__(self, idx):
        """
        Get a single found intron.
        
        :param idx: Index in the list of found introns
        :return: Intron information as a dict: ["soi"] is a SOI (input of the model), ["metadata"] is various other information.
        """
        
        intron = self.introns[idx]
        #if not got_introns is list:
        #    got_introns = [got_introns]
        out = dict()
        out["inputs"] = dict()
        # input sequence as string
        out["inputs"]["soi"] = np.array(intron.attributes["SOI"])
        out["inputs"]["bp_index"] = np.array(self.bp_indexes[idx])
        #out["metadata"] = list()
        #for intron in got_introns:
        #    intron_meta = dict()
        #    # metadata for the output
        #    for term in ["gene_id", "transcript_id", "number"]:
        #        intron_meta[term] = intron.attributes[term]

        #    intron_meta["start"] = intron.start
        #    intron_meta["end"] = intron.end
        #    intron_meta["seqid"] = intron.seqid
        #    intron_meta["strand"] = intron.strand
        #    out["metadata"].append(intron_meta)
        out["metadata"] = dict()
        
        intron_meta = dict()
        # metadata for the output
        for term in ["gene_id", "transcript_id", "number"]:
            intron_meta[term] = np.array(intron.attributes[term])

        intron_meta["start"] = np.array(intron.start)
        intron_meta["end"] = np.array(intron.end)
        intron_meta["seqid"] = np.array(intron.seqid)
        intron_meta["strand"] = np.array(intron.strand)
        out["metadata"] = intron_meta
        
        return out
        
    
    def _get_introns(self):
        """
        Get introns from the specified seq and annotation files.
        
        :return: List of all detected introns as gffutils.Feature objects.
        """
        # create a gffutils database
        self.db = gffutils.create_db(data = self.gtf_file, dbfn = ":memory:", \
                                force = True, id_spec= {'gene': 'gene_id', 'transcript': 'transcript_id'}, \
                                disable_infer_transcripts= True, disable_infer_genes=True, verbose= False, \
                                merge_strategy="merge" )
        
        if not self.create_introns:
            #load introns from gtf, don't create them
            introns = list(self.db.features_of_type('intron', order_by= ('seqid', 'start', 'end'))) # exons are sorted start-coord. asc.
            self._add_SOI(introns)
            return introns
        
        exons = list(self.db.features_of_type('exon', order_by= ('seqid', 'start', 'end'))) # exons are sorted start-coord. asc.
        
        #group exons in a dict by gene id
        transcript_to_exon = self._get_tr_to_exon_dict(exons)
        
        collected_introns=self._build_introns(transcript_to_exon)
                
        self._add_SOI(collected_introns)
        
        return collected_introns
    
    
    def _get_tr_to_exon_dict(self,exons_list):
        out = dict()
        for exon in exons_list:
            [exon_transcript] = exon.attributes["transcript_id"]
            if not exon_transcript in out:
                out[exon_transcript] = [exon]
            else:
                out[exon_transcript].append(exon)
        return out
        
    def _build_introns(self,transcript_to_exon):
        """
        Build, number and pack introns into the list.
        """
        out = []
        for transcript, exons in transcript_to_exon.items():
            # exons are already sorted in start end ascending order
            # depending on the strand, number them (reverse numbering on minus)
            if exons[0].strand == "-":
                for i in range(1,len(exons) + 1):
                    exons[-i].attributes["number"] = i
            else:
                for i in range(0,len(exons)):
                    exons[i].attributes["number"] = i + 1
                    
            #interfeatures() builds intronic features between exons on transcripts
            introns = list(self.db.interfeatures(exons, 'intron', merge_attributes=True))

            for i in range(0, len(introns)):
                introns[i].attributes["number"] = str(i+1)
            out.extend(introns)
        return out
        
        
    def _add_SOI(self,introns):
        """
        Loads intron sequences from fasta file.
        """
        for intron in introns:
            # extend region not depending on strand
            intron.start -=21
            intron.end +=21
            # slice first 17 bases from donor, because we don't need them in our model
            intron.attributes["SOI"] = intron.sequence(self.fasta_file, use_strand = True)[21 - 3:]
            # get true intronic coordinates back in place
            intron.end -=21
            intron.start +=21
