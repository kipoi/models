# python2, 3 compatibility
from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
import inspect
import os

filename = inspect.getframeinfo(inspect.currentframe()).filename
DATALOADER_DIR = os.path.dirname(os.path.abspath(filename))


def data_preprocessing(dataframe,conservation_data,process_name):
  start_time=time.time()
  print 'start time'+ str(start_time)
  seq_length=51
  flank=(seq_length-1)/2
  #
  conservation_data[['primatet','mammalt','vertebratet']]=conservation_data[['primate','mammal','vertebrate']].applymap(lambda x :\
    np.concatenate((np.zeros((flank,20)),x,np.zeros((flank,20))),axis=0).flatten())
  conservation_data['sequencet']=conservation_data.apply(lambda x : 'Z'*flank+x['sequence']+'Z'*flank,axis=1)
  #
  #
  dataframe=dataframe.reset_index()
  del dataframe['index']
  dataframe['index_sort']=dataframe.index
  dataframe=dataframe.merge(conservation_data,on='gene_name').sort_values('index_sort')
  #sequence
  dataframe['extractseq']=dataframe.apply(lambda x : x['sequencet']\
    [x['change_position_1based']-flank+(flank-1):x['change_position_1based']+(flank-1)+(flank+1)],axis=1)
  dataframe['orig_sequence']=dataframe.apply(lambda x : x['extractseq'][:flank] +x['ref_aa']+x['extractseq'][(flank+1):],axis=1)
  dataframe['snp_sequence']=dataframe.apply(lambda x : x['extractseq'][:flank] +x['alt_aa']+x['extractseq'][(flank+1):],axis=1)
  dataframe[['ref_seq','alt_seq']]=dataframe[['orig_sequence','snp_sequence']].applymap(lambda x : x.replace('-','Z').replace('*','Z').\
  replace('Z','00000000000000000000').replace('Y','00000000000000000001').replace('W','00000000000000000010').\
  replace('V','00000000000000000100').replace('T','00000000000000001000').replace('S','00000000000000010000').\
  replace('R','00000000000000100000').replace('Q','00000000000001000000').replace('P','00000000000010000000').\
  replace('N','00000000000100000000').replace('M','00000000001000000000').replace('L','00000000010000000000').\
  replace('K','00000000100000000000').replace('I','00000001000000000000').replace('H','00000010000000000000').\
  replace('G','00000100000000000000').replace('F','00001000000000000000').replace('E','00010000000000000000').\
  replace('D','00100000000000000000').replace('C','01000000000000000000').replace('A','10000000000000000000'))
  X_test_orig_1=dataframe['ref_seq'].as_matrix().astype(str).view('S1').reshape(len(dataframe),-1,20)   #change as needed
  print X_test_orig_1.shape
  X_test_snp_1=dataframe['alt_seq'].as_matrix().astype(str).view('S1').reshape(len(dataframe),-1,20) #change as needed
  dataframe=dataframe.drop(['extractseq','orig_sequence','snp_sequence','ref_seq','alt_seq'], axis=1)
  #
  temp_index_changepositon=dataframe.columns.get_loc('change_position_1based')+1
  #
  temp_index_primatet=dataframe.columns.get_loc('primatet')+1
  X_train_conservation_onlyprimates=numpy.empty((len(dataframe),seq_length*20))
  for i, el in enumerate((x[temp_index_primatet][(x[temp_index_changepositon]+(flank-1)-flank)*20:(x[temp_index_changepositon]+(flank-1)+(flank+1))*20] for x in dataframe.itertuples())):
    X_train_conservation_onlyprimates[i] = el
  X_train_conservation_onlyprimates=X_train_conservation_onlyprimates.reshape(-1,seq_length,20)
  #
  temp_index_mammalt=dataframe.columns.get_loc('mammalt')+1
  X_train_conservation_mammals=numpy.empty((len(dataframe),seq_length*20))
  for i, el in enumerate((x[temp_index_mammalt][(x[temp_index_changepositon]+(flank-1)-flank)*20:(x[temp_index_changepositon]+(flank-1)+(flank+1))*20] for x in dataframe.itertuples())):
    X_train_conservation_mammals[i]=el
  X_train_conservation_mammals=X_train_conservation_mammals.reshape(-1,seq_length,20)
  #
  temp_index_vertebratet=dataframe.columns.get_loc('vertebratet')+1
  X_train_conservation_vertebrates=numpy.empty((len(dataframe),seq_length*20))
  for i, el in enumerate((x[temp_index_vertebratet][(x[temp_index_changepositon]+(flank-1)-flank)*20:(x[temp_index_changepositon]+(flank-1)+(flank+1))*20] for x in dataframe.itertuples())):
    X_train_conservation_vertebrates[i]=el
  X_train_conservation_vertebrates=X_train_conservation_vertebrates.reshape(-1,seq_length,20)
  #
  dataframe['label']=dataframe['label'].apply(lambda x : x.replace('Unknown','1').replace('Benign','0').replace('likely benign','1')).astype(int)
  y_train=dataframe['label'].as_matrix()
  print ' the time for the data preprocessing is ' + str(time.time()-start_time)
  return (X_test_orig_1,X_test_snp_1,X_train_conservation_mammals,X_train_conservation_onlyprimates,X_train_conservation_vertebrates,y_train)



def loaddata(vcf_file):
    required_data=pd.read_csv(os.path.join(DATALOADER_DIR, 'dataloader_files/full_data_coverage_species.csv'))
    required_data=required_data[required_data['mean_coverage_bins']!=0.0]
    #
    conservation_sequence_data=pd.DataFrame(np.load(os.path.join(DATALOADER_DIR, 'dataloader_files/conservation_without_msa_full.npy')))
    conservation_sequence_data.columns=['gene_name','sequence','primate','mammal','vertebrate']
    #filter variants 
    filtered_set=pd.read_csv(open(vcf_file,'r'),sep='\t')
    filtered_set=filtered_set[[0,1,3,4]]
    filtered_set.columns = ['chr', 'pos', 'ref', 'alt']
    filted_set=filted_set.merge(required_data,left_on=['chr','pos','ref','alt'],right_on=['chr','pos','non_flipped_ref','non_flipped_alt'])
    #
    data=data_preprocessing(filtered_set,conservation_sequence_data.copy(),process_name)
    y_train=data[5]
    X_train_orig_1=data[0]
    X_train_snp_1=data[1]
    X_train_conservation_full=data[2]
    X_train_conservation_onlyprimates=data[3]
    X_train_conservation_otherspecies=data[4]
    return {
        "inputs": {
            "orig_seq": X_train_orig_1,
            "snp_seq": X_train_snp_1,
            "conservation_full": X_train_conservation_full ,
            "conservation_primates": X_train_conservation_onlyprimates,
            "conservation_otherspecies": X_train_conservation_otherspecies,
        }
    }
