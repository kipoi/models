import torch
import collections
import pdb
import csv
from kipoi.data import Dataset
import math
import numpy as np

class HMData(Dataset):
	# Dataset class for loading data
	def __init__(self, input_file, bin_size=100):
		self.hm_data = self.loadData(input_file, bin_size)


	def loadData(self,filename,windows):
		with open(filename) as fi:
			csv_reader=csv.reader(fi)
			data=list(csv_reader)

			ncols=(len(data[0]))
		fi.close()
		nrows=len(data)
		ngenes=nrows/windows
		nfeatures=ncols-1
		print("Number of genes: %d" % ngenes)
		print("Number of entries: %d" % nrows)
		print("Number of HMs: %d" % nfeatures)

		count=0
		attr=collections.OrderedDict()

		for i in range(0,nrows,windows):
			hm1=torch.zeros(windows,1)
			hm2=torch.zeros(windows,1)
			hm3=torch.zeros(windows,1)
			hm4=torch.zeros(windows,1)
			hm5=torch.zeros(windows,1)
			for w in range(0,windows):
				hm1[w][0]=int(data[i+w][2])
				hm2[w][0]=int(data[i+w][3])
				hm3[w][0]=int(data[i+w][4])
				hm4[w][0]=int(data[i+w][5])
				hm5[w][0]=int(data[i+w][6])
			geneID=str(data[i][0].split("_")[0])

			thresholded_expr = int(data[i+w][7])

			attr[count]={
				'geneID':geneID,
				'expr':thresholded_expr,
				'hm1':hm1,
				'hm2':hm2,
				'hm3':hm3,
				'hm4':hm4,
				'hm5':hm5
			}
			count+=1

		return attr


	def __len__(self):
		return len(self.hm_data)

	def __getitem__(self,i):
		final_data=torch.cat((self.hm_data[i]['hm1'],self.hm_data[i]['hm2'],self.hm_data[i]['hm3'],self.hm_data[i]['hm4'],self.hm_data[i]['hm5']),1)
		final_data = final_data.numpy()
		label = self.hm_data[i]['expr']
		geneID = self.hm_data[i]['geneID']


		return_item={
					'inputs': final_data,
					'metadata': {'geneID':geneID,'label':label}
					}

		return return_item
