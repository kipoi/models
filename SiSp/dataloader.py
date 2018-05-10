"""SiSp dataloader
"""
# python2, 3 compatibility
from __future__ import absolute_import, division, print_function

import pandas as pd
import numpy as np
import pysam
import itertools
import subprocess
import shlex
import os
from optparse import OptionParser, OptionGroup
import sys
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO
from kipoi.data import PreloadedDataset

class FastaFile:
    """docstring for FastaFile"""
    def __init__(self, fasta_file):
        self.f = pysam.FastaFile(fasta_file)

    def get_seq(self, qref, start, stop):
        """get the sequence in a given region, the start is from 1.
        The start and stop index may still need double check."""
        return self.f.fetch(qref, start-1, stop)

def rev_seq(seq):
    _tmp = []
    _tmp[:] = seq
    for j in range(len(_tmp)):
        if _tmp[j] == "A": _tmp[j] = "T"
        elif _tmp[j] == "T": _tmp[j] = "A"
        elif _tmp[j] == "G": _tmp[j] = "C"
        elif _tmp[j] == "C": _tmp[j] = "G"
    RV = "".join(_tmp[::-1])
    return RV

class Transcript:
    def __init__(self, chrom, strand, start, stop, tran_id, tran_name="*", 
        biotype="*"):
        """a general purpose transcript object with the basic information.
        """
        self.chrom  = chrom
        self.strand = strand
        self.start  = int(start)
        self.stop   = int(stop)
        self.tranID = tran_id
        self.exons  = np.zeros((0,2), "int")
        self.seglen = None
        self.tranL  = 0
        self.exonNum = 0
        self.biotype = biotype
        self.tranName = tran_name
        

    def add_exon(self, chrom, strand, start, stop):
        if strand != self.strand or chrom != self.chrom:
            print("The exon has different chrom or strand to the transcript.")
            return
        _exon = np.array([start, stop], "int").reshape(1,2)
        self.exons = np.append(self.exons, _exon, axis=0)
        self.exons = np.sort(self.exons, axis=0)
        self.tranL += abs(int(stop) - int(start) + 1)
        self.exonNum += 1


        self.seglen = np.zeros(self.exons.shape[0] * 2 - 1, "int")
        self.seglen[0] = self.exons[0,1]-self.exons[0,0] + 1
        for i in range(1, self.exons.shape[0]):
            self.seglen[i*2-1] = self.exons[i,0]-self.exons[i-1,1] - 1
            self.seglen[i*2] = self.exons[i,1]-self.exons[i,0] + 1

        if ["-","-1","0",0,-1].count(self.strand) > 0:
            self.seglen = self.seglen[::-1]


class Gene:
    def __init__(self, chrom, strand, start, stop, gene_id, gene_name="*",
        biotype="*"):
        """
        """
        self.chrom  = chrom
        self.strand = strand
        self.start  = int(start)
        self.stop   = int(stop)
        self.exons  = np.zeros((0,2), "int")
        self.tranL  = 0
        self.exonNum = 0
        self.geneID = gene_id
        self.trans  = []
        self.tranNum = 0
        self.biotype = biotype
        self.geneName = gene_name
        
    def add_transcipt(self, transcript):
        self.trans.append(transcript)
        self.tranNum += 1

    def get_gene_info(self):
        RV = [self.geneID, self.geneName, self.chrom, self.strand, self.start,
              self.stop, self.biotype]
        _trans = []
        for t in self.trans:
            _trans.append(t.tranID)
        RV.append(",".join(_trans))
        return RV

    def add_premRNA(self):
        _tran = Transcript(self.chrom, self.strand, self.start, self.stop, 
                           self.geneID+".p", self.geneName, self.biotype)
        _tran.add_exon(self.chrom, self.strand, self.start, self.stop)
        self.trans.append(_tran)
        self.tranNum += 1
        
    def get_exon_max_num(self):
        exonMax = 0
        for _tran in self.trans:
            exonMax = max(exonMax, _tran.exonNum)
        return exonMax

    def gene_ends_update(self):
        for t in self.trans:
            self.start = min(self.start, np.min(t.exons))
            self.stop = max(self.stop, np.max(t.exons))
    
    def add_exon(self, chrom, strand, start, stop):
        if strand != self.strand or chrom != self.chrom:
            print("The exon has different chrom or strand to the transcript.")
            return
        _exon = np.array([start, stop], "int").reshape(1,2)
        self.exons = np.append(self.exons, _exon, axis=0)
        self.exons = np.sort(self.exons, axis=0)
        self.tranL += abs(int(stop) - int(start) + 1)
        self.exonNum += 1


        self.seglen = np.zeros(self.exons.shape[0] * 2 - 1, "int")
        self.seglen[0] = self.exons[0,1]-self.exons[0,0] + 1
        for i in range(1, self.exons.shape[0]):
            self.seglen[i*2-1] = self.exons[i,0]-self.exons[i-1,1] - 1
            self.seglen[i*2] = self.exons[i,1]-self.exons[i,0] + 1

        if ["-","-1","0",0,-1].count(self.strand) > 0:
            self.seglen = self.seglen[::-1]






def parse_attribute(attStr, default="*", 
    ID_tags="ID,gene_id,transcript_id,mRNA_id",
    Name_tags="Name,gene_name,transcript_name,mRNA_name",
    Type_tags="Type,gene_type,gene_biotype,biotype",
    Parent_tags="Parent"):
    """
    Parse attributes in GTF or GFF3
    Parameters
    ----------
    attStr: string
        String containing attributes either in GTF or GFF3 format.
    default: string
        default value for ID, Name, Type and Parent.
    ID_tags: string
        Multiple tags for ID. Use comma for delimit. 
        If multiple tags found, use the last one.
    Name_tags: string
        Multiple tags for Name. Use comma for delimit. 
        If multiple tags found, use the last one.
    Type_tags: string
        Multiple tags for Type. Use comma for delimit. 
        If multiple tags found, use the last one.
    Parent_tags: string
        Multiple tags for Parent. Use comma for delimit. 
        If multiple tags found, use the last one.
    Returns
    -------
    RV: library of string
        Library of all tags, always including ID, Name, Type, Parenet.
    """
    RV = {}
    RV["ID"] = default
    RV["Name"] = default
    RV["Type"] = default
    RV["Parent"] = default
    ID_tags = ID_tags.split(",")
    Name_tags = Name_tags.split(",")
    Type_tags = Type_tags.split(",")
    Parent_tags = Parent_tags.split(",")

    attList = attStr.rstrip().split(";")
    for att in attList:
        while len(att) > 0 and att[0] == " ": 
            att = att[1:]
        if len(att) == 0: 
            continue
        if att.find("=") > -1:
            _att = att.split("=") #GFF3
        else:
            _att = att.split(" ") #GTF

        if len(_att) < 2:
            print("Can't pase this attribute: %s" %att)
            continue

        if _att[1][0] == '"':
            _att[1] = _att[1].split('"')[1]

        if ID_tags.count(_att[0]) == 1:
            RV["ID"] = _att[1]
        elif Name_tags.count(_att[0]) == 1:
            RV["Name"] = _att[1]
        elif Type_tags.count(_att[0]) == 1:
            RV["Type"] = _att[1]
        elif Parent_tags.count(_att[0]) == 1:
            RV["Parent"] = _att[1]
        else: RV[_att[0]] = _att[1]

    return RV



def loadgene(anno_file, comments="#,>", geneTag="gene", 
        tranTag="transcript,mRNA", exonTag="exon"):
    """
    Load genes from gtf or gff3 file.
    Parameters
    ----------
    anno_file: str
        path for the annotation file in GTF or GFF3 format.
    comments: string
        Multiple comments. Use comma for delimit. 
    geneTag: string
        Multiple tags for gene. Use comma for delimit. 
    tranTag: string
        Multiple tags for transcript. Use comma for delimit. 
    exonTag: string
        Multiple tags for exon. Use comma for delimit. 
    Return
    ------
    genes: list of ``pyseqlib.Gene``
        a list of loaded genes
    """

    #TODO: load gzip file
    fid = open(anno_file, "r")
    anno_in = fid.readlines()
    fid.close()

    geneTag = geneTag.split(",")
    tranTag = tranTag.split(",")
    exonTag = exonTag.split(",")
    comments = comments.split(",")

    genes = []
    genenames = []
    _gene = None
    for _line in anno_in:
        
        if comments.count(_line[0]):
            continue
        
        
        aLine = _line.split("\t")
        
        
        if len(aLine) < 8:
            continue
       
        
        elif geneTag.count(aLine[2]) == 1:
            RVatt = parse_attribute(aLine[8], ID_tags="ID,gene_id",
                Name_tags="Name,gene_name")
            _gene = Gene(aLine[0], aLine[6], aLine[3], aLine[4],
                RVatt["ID"], RVatt["Name"], RVatt["Type"])
            
            if _gene is not None: 
                genes.append(_gene)
             
                
        elif tranTag.count(aLine[2]) == 1:
            RVatt = parse_attribute(aLine[8],ID_tags="ID,transcript_id,mRNA_id",
                Name_tags="Name,transcript_name,mRNA_name")
            _tran  = Transcript(aLine[0], aLine[6], aLine[3], aLine[4],
                RVatt["ID"], RVatt["Name"], RVatt["Type"])

            if _gene is not None:
                _gene.add_transcipt(_tran)
            else:
                print("Gene is not ready before transcript.")

                
        elif exonTag.count(aLine[2]) == 1:
            #if aLine[0] != _gene.trans[-1].chrom:
            #    print("Exon from a different chrom of transcript.")
            #    continue
            #if aLine[6] != _gene.trans[-1].strand:
            #    print("Exon from a different strand of transcript.")
            #    continue
            RVatt = parse_attribute(aLine[8], ID_tags="ID,gene_id",
                                Name_tags="Name,gene_name")
            _gene = Gene(aLine[0], aLine[6], aLine[3], aLine[4],
                     RVatt["ID"], RVatt["Name"], RVatt["Type"])
            
            if _gene is not None:
                _gene.add_exon(aLine[0], aLine[6], aLine[3], aLine[4])
            else:
                print("Gene or transcript is not ready before exon.")
         
                
        if genenames is not None:
            genenames.append(_gene.geneID)
        
            
        if _gene is not None: 
            genes.append(_gene)    

    return genes, genenames



def get_one_hot(sequence, nucleo):
    #also replace non excisting nucleos with 0
    repl='TGCAN'
    sequence=sequence.replace(nucleo, '1')
    for nucl in repl:
        sequence=sequence.replace(nucl, '0')
    t=[i for i in sequence]
    return t



def get_one_hot_C(sequence, region_dict  ):
    #also replace non excisting nucleos with 0
    repl='TGAN'
    seq_Cmeth_new=""
    seq_C_new=""

    for nucl in repl:
        sequence=sequence.replace(nucl, '0')
    # split the Cs in C meth and C unmeth
    for dict_key in sorted(region_dict):
        meth_true=region_dict[dict_key][2]
        start_meth=int(region_dict[dict_key][0])
        end_meth=int(region_dict[dict_key][1])
        if meth_true==1:
            seqsnip_Cmeth=sequence[start_meth:end_meth+1].replace("C", '1')
            seqsnip_C=sequence[start_meth:end_meth+1].replace("C", '0')
        else:
            seqsnip_Cmeth=sequence[start_meth:end_meth+1].replace("C", '0')
            seqsnip_C=sequence[start_meth:end_meth+1].replace("C", '1')
        seq_C_new=seq_C_new + seqsnip_C
        seq_Cmeth_new=seq_Cmeth_new+seqsnip_Cmeth

    return seq_C_new, seq_Cmeth_new



def get_methylation_dict(df_output):
    # detact methylation changes 
    output_list=list(df_output[3])
    output_changes=[output_list[n]!=output_list[n-1] for n in range(len(output_list))]
    output_changes[0]=False

    # get the methylation regions, if methylation changes start new region
    # save in dict with the start of the region, the end of the region and the methylation
    region_dict={}
    #use relative lenght in comparison to start of sequence
    start_region=0
    end_region=seq_end-seq_start-1
    n=0  # counter for splice entries
    m=1 # counter for dict entries 
    for entry in output_changes:
        if entry==True:
            #get the middle point of the splicing change
            middle_region=(df_output[1][n]+df_output[1][n-1])/2-seq_start
            # get the methylation (of previous region)
            methylation=df_output[3][n-1]
            name="key_"+str(m)
            region_dict[name]=[start_region, middle_region, methylation ]
            #update the start region
            start_region=middle_region+1
            m=m+1
        n=n+1
        
    # finish with last region  (-1 because we already added 1 to n)       
    methylation=df_output[3][n-1]
    name="key_"+str(m)
    region_dict[name]=[start_region, end_region, methylation ]
    return region_dict  



def get_hot_seq_meth(meth_file, genes, fastaFile):  
    seqshot2=np.ndarray(shape=(len(genes),800,5), dtype=float)
    n=0  
    for g in genes:

        #if (n+1)%100==0:
        #    break
        #    print n
        gene_ids=g.geneID
        exons = g.exons
        global chrom
        chrom = g.chrom
        global strand
        strand=g.strand
        alt=exons[0]
        cons_gene=[]
    
        #get the center of the exon 
        center=(alt[1]+alt[0])/2
        #get the 800 bp around the exon
        global seq_start
        seq_start=center-400
        global seq_end
        seq_end=center+400
        #seq_ex=np.ndarray((2,), buffer=np.array([seq_start,seq_end]), dtype=int)
    
        ######    
        #get sequences 
        ######
    
        fastaFilefasta = FastaFile(fastaFile)
        Seq_nuc=fastaFilefasta.get_seq(chrom, seq_start+1, seq_end)
        Seq_nuc=Seq_nuc.upper()
        if strand != "+" and strand != "1":
            Seq_nuc=rev_seq(Seq_nuc)
        
        ######    
        #get methylation pattern
        ######       
    
        #check if we have methylation knowlege from the region itsef
        
        bashCommand = "tabix %s %s:%d-%d" %(meth_file, chrom, seq_start+1, seq_end)
        bashCommand2=bashCommand
        pro = subprocess.Popen(shlex.split(bashCommand), stdout=subprocess.PIPE)
        output_2 = pro.communicate()
    
        # get in a nicer representation
        output_repres=StringIO(output_2[0].decode("utf-8"))
       
        # sometime the output is empty then enlarge the regio be 100bp in each direction till we have the knwlege
        try:
            df_output = pd.read_csv(output_repres, sep="\t", header=None)
        except:
            # loop till the output is not empty 
            #maximal looping is 8
            m=1
            while len(output_2[0])==0 and m<9:
                bashCommand = "tabix %s %s:%d-%d" %(meth_file, chrom, seq_start+1-100*m, seq_end+100*m)
                bashCommand2=bashCommand
                pro = subprocess.Popen(shlex.split(bashCommand), stdout=subprocess.PIPE)
                output_2 = pro.communicate()
                m=m+1
            
            if m<9: #methylation was succesful
                #get the median methylation
                output_repres_dummi=StringIO(output_2[0])
                df_output_dummi = pd.read_csv(output_repres_dummi, sep="\t", header=None)
                meth_state= np.median(df_output_dummi[3])
        
                output_2=(chrom+'\t'+str(seq_start+1)+'\t'+str(seq_start+2)+'\t'+str(meth_state))
            
            else: #methylation was not succesful, set to 0 
                #set to unmetylated
                output_2=(chrom+'\t'+str(seq_start+1)+'\t'+str(seq_start+2)+'\t0')
            
            # get in a nicer representation
            output_repres=StringIO(output_2)
            df_output = pd.read_csv(output_repres, sep="\t", header=None)
    
        region_dict=get_methylation_dict(pd.DataFrame(df_output.loc[0]).transpose())
    
        ######    
        #one-hot encodement 
        ######
    
        seq_A=get_one_hot(Seq_nuc, 'A')
        seq_G=get_one_hot(Seq_nuc, 'G')
        seq_T=get_one_hot(Seq_nuc, 'T')
        seq_C, seq_Cmeth=get_one_hot_C(Seq_nuc, region_dict)
        
        seq_C = [character for character in seq_C]
        seq_Cmeth = [character for character in seq_Cmeth]
        #seq_C=np.array([character for character in seq_C])
        #seq_Cmeth=np.array([character for character in seq_Cmeth])
    
        ######    
        #weight with conservation score 
        ######
        #try:
        #    bashCommand = "/homes/stlinker/ucsc/bigWigSummary %s %s %d %d %d" %(phast_file, chrom, 
        #            seq_start, seq_end, seq_end-seq_start )
        #    pro = subprocess.Popen(shlex.split(bashCommand), stdout=subprocess.PIPE)
        #    output = pro.communicate()
        #    a=str(output[0]).split('\t') 
        #    #if no conservation score is available replace with lowest value 0.001
        #    b=['0.001' if x=='n/a' or x=='n/a\n' else x for x in a]
        #    cons_gene=[float(entry) for entry in b]
        #except:
        #    cons_gene=list(np.zeros(800))
    
        # make lowest value of conservation score 0.001
        #cons_gene=[float('0.001') if x==0 else x for x in cons_gene]

        # multiply with one hot cold
        #seq_A=np.array(cons_gene)*np.array(seq_A).astype(int)
        #seq_G=np.array(cons_gene)*np.array(seq_G).astype(int)
        #seq_T=np.array(cons_gene)*np.array(seq_T).astype(int)
        
        
        #list_seq_C=np.array([int(character) for character in seq_C])
        #seq_C=np.array(cons_gene)*list_seq_C
        # 
        #list_seq_Cmeth=np.array([int(character) for character in seq_Cmeth])
        #seq_Cmeth=np.array(cons_gene)*list_seq_Cmeth
    
        seqshot2[n,:,0] = seq_A
        seqshot2[n,:,1] = seq_C
        seqshot2[n,:,2] = seq_Cmeth
        seqshot2[n,:,3] = seq_G
        seqshot2[n,:,4] = seq_T
        n=n+1
    return seqshot2



def data(anno_file, fasta_file, meth_file, target_file=None):
    """
    Args:
        anno_file: file path; gtf file containing genes and exons
        fasta_file: file path; Genome sequence
        target_file: file path; path to the targets in the csv format
	meth_file: file path; methylation information
    """
    bt, genes = loadgene(anno_file)
    SEQ_WIDTH = 800
       
    if target_file is not None:
        targets = pd.read_csv(target_file, header=None, index_col=0)
        targets = targets.loc[genes]
        targets = targets.values
    else:
        targets = None
    # Run the fasta extractor
    seq = get_hot_seq_meth(meth_file, bt, fasta_file)
    # import pdb
    # pdb.set_trace()
    return {
            "inputs": seq,
            "targets": targets,
            "metadata": {
                "gene_id": np.array(genes),
             }
           }
