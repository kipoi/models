from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pdb import set_trace as stop

def batch_product(iput, mat2):
		result = None
		for i in range(iput.size()[0]):
			op = torch.mm(iput[i], mat2)
			op = op.unsqueeze(0)
			if(result is None):
				result = op
			else:
				result = torch.cat((result,op),0)
		return result.squeeze(2)


class rec_attention(nn.Module):
	# attention with bin context vector per HM and HM context vector
	def __init__(self,hm,args):
		super(rec_attention,self).__init__()
		self.num_directions=2 if args.bidirectional else 1
		if (hm==False):
			self.bin_rep_size=args.bin_rnn_size*self.num_directions
		else:
			self.bin_rep_size=args.bin_rnn_size
	
		self.bin_context_vector=nn.Parameter(torch.Tensor(self.bin_rep_size,1),requires_grad=True)
	

		self.softmax=nn.Softmax(dim=1)

		self.bin_context_vector.data.uniform_(-0.1, 0.1)

	def forward(self,iput):
		alpha=self.softmax(batch_product(iput,self.bin_context_vector))
		[batch_size,source_length,bin_rep_size2]=iput.size()
		repres=torch.bmm(alpha.unsqueeze(2).view(batch_size,-1,source_length),iput)
		return repres,alpha



class recurrent_encoder(nn.Module):
	# modular LSTM encoder
	def __init__(self,n_bins,ip_bin_size,hm,args):
		super(recurrent_encoder,self).__init__()
		self.bin_rnn_size=args.bin_rnn_size
		self.ipsize=ip_bin_size
		self.seq_length=n_bins

		self.num_directions=2 if args.bidirectional else 1
		if (hm==False):
			self.bin_rnn_size=args.bin_rnn_size
		else:
			self.bin_rnn_size=args.bin_rnn_size // 2
		self.bin_rep_size=self.bin_rnn_size*self.num_directions


		self.rnn=nn.LSTM(self.ipsize,self.bin_rnn_size,num_layers=args.num_layers,dropout=args.dropout,bidirectional=args.bidirectional)

		self.bin_attention=rec_attention(hm,args)
	def outputlength(self):
		return self.bin_rep_size
	def forward(self,single_hm,hidden=None):
		bin_output, hidden = self.rnn(single_hm,hidden)
		bin_output = bin_output.permute(1,0,2)
		hm_rep,bin_alpha = self.bin_attention(bin_output)
		return hm_rep,bin_alpha


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class att_chrome(nn.Module):
	def __init__(self,args):
		super(att_chrome,self).__init__()
		self.n_hms=args.n_hms
		self.n_bins=args.n_bins
		self.ip_bin_size=1
		
		self.rnn_hms=nn.ModuleList()
		for i in range(self.n_hms):
			self.rnn_hms.append(recurrent_encoder(self.n_bins,self.ip_bin_size,False,args))
		self.opsize = self.rnn_hms[0].outputlength()
		self.hm_level_rnn_1=recurrent_encoder(self.n_hms,self.opsize,True,args)
		self.opsize2=self.hm_level_rnn_1.outputlength()
		self.diffopsize=2*(self.opsize2)
		self.fdiff1_1=nn.Linear(self.opsize2,1)

	def forward(self,iput):

		bin_a=None
		level1_rep=None
		[batch_size,_,_]=iput.size()

		for hm,hm_encdr in enumerate(self.rnn_hms):
			hmod=iput[:,:,hm].contiguous()
			hmod=torch.t(hmod).unsqueeze(2)

			op,a= hm_encdr(hmod)
			if level1_rep is None:
				level1_rep=op
				bin_a=a
			else:
				level1_rep=torch.cat((level1_rep,op),1)
				bin_a=torch.cat((bin_a,a),1)
		level1_rep=level1_rep.permute(1,0,2)
		final_rep_1,hm_level_attention_1=self.hm_level_rnn_1(level1_rep)
		final_rep_1=final_rep_1.squeeze(1)
		prediction_m=((self.fdiff1_1(final_rep_1)))
		
		return torch.sigmoid(prediction_m)

args_dict = {'lr': 0.0001, 'model_name': 'attchrome', 'clip': 1, 'epochs': 2, 'batch_size': 10, 'dropout': 0.5, 'cell_1': 'Cell1', 'save_root': 'Results/Cell1', 'data_root': 'data/', 'gpuid': 0, 'gpu': 0, 'n_hms': 5, 'n_bins': 200, 'bin_rnn_size': 32, 'num_layers': 1, 'unidirectional': False, 'save_attention_maps': False, 'attentionfilename': 'beta_attention.txt', 'test_on_saved_model': False, 'bidirectional': True, 'dataset': 'Cell1'}
att_chrome_args = AttrDict(args_dict)
att_chrome_model = att_chrome(att_chrome_args)

