
import torch
import torch.nn as nn

from kipoi.model import BaseModel

# Simple CNN, taking SeqVec embeddings for a given protein as input.
# Returns predictions for 3-state and 8-state secondary structure
# as well as for disorder for each residue in the given protein.
class CNN( nn.Module ):
    def __init__( self ):
        super(CNN, self).__init__()
        
        self.elmo_feature_extractor = nn.Sequential(
                        nn.Conv2d( 1024, 32, kernel_size=(7,1), padding=(3,0) ),
                        nn.ReLU(),
                        nn.Dropout( 0.25 ),
                        )
        

        n_final_in = 32
        self.dssp3_classifier = nn.Sequential(
                        nn.Conv2d( n_final_in, 3, kernel_size=(7,1), padding=(3,0))
                        )
        
        self.dssp8_classifier = nn.Sequential(
                        nn.Conv2d( n_final_in, 8, kernel_size=(7,1), padding=(3,0))
                        )
        self.diso_classifier = nn.Sequential(
                        nn.Conv2d( n_final_in, 2, kernel_size=(7,1), padding=(3,0))
                        )

    def forward( self, x):
        x         = self.elmo_feature_extractor( x ) 
        
        d3_Yhat   = self.dssp3_classifier( x )
        d8_Yhat   = self.dssp8_classifier( x ) 
        diso_Yhat = self.diso_classifier(  x )

        return d3_Yhat, d8_Yhat, diso_Yhat


class SeqVec2structure( BaseModel ):
    def __init__(self, weights):
        self.model = self.load_weights(weights)

    def load_weights( self, weights ):
        model = CNN()
        weights = torch.load(weights, map_location=None if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(weights)
        return model.eval()


    def get_predictions( self, Yhat ):
        Yhat = Yhat.squeeze(dim=-1) # remove last empty dim
        Yhat = torch.max( Yhat.data, dim=1 )[1] # get index of most reliable class prediction
        DSSP3_MAPPING = {0:'H',1:'E',2:'C'} # map indices to labels
        predictions = list() # container for predictions
        
        for sample_idx in range(Yhat.shape[0]): # for all samples in a batch
            sample = Yhat[sample_idx].long().cpu().numpy()# single sample      
            prediction = ''.join([ DSSP3_MAPPING[ sample[idx] ] 
                                   for idx in range(sample.size)] )
            predictions.append( prediction )
        return predictions


    def predict_on_batch( self, x ):
        x = custom_collate( x )
        Yhat = self.model(x)
        predictions = self.get_predictions(Yhat[0])
        Yhat = [ yhat.cpu().detach().numpy() for yhat in Yhat ]
        return Yhat


def custom_collate( batch ):
    '''
        This custom collate function creates batches based on the longest
        sequence in every batch. This removes the necessity of having to pad
        all sequence to the longest sequence in the whole data set.
    '''
    
    max_len = 0 # get the length of the longest sequence in the batch
    for sample in batch: # for all samples in the batch
        # each sample has the shape [L,1024]
        seq_len = sample.shape[0]
        if seq_len > max_len: # get longest sequence in this batch
            max_len = seq_len
    
    X = list()
    for x in batch: # for all samples in the batch
        x = torch.from_numpy(x)
        # get seq_len to calculate the number of residues for zero-padding
        n_to_be_padded = max_len - x.shape[0]

        # pad zeros along dimensions Left, Right, Top, Bottom/Seq
        padder = torch.nn.ConstantPad2d( (0, 0, 0, n_to_be_padded), 0)
        
        # Pad missing elements with zeros
        x = padder(x)
        x = x.transpose(1,0)

        # Add singleton-dimension to fit batch-requirement
        x = x.unsqueeze(dim=0)
        
        X.append( x )    
    
    # create single batch with same dimensions after padding
    X = torch.cat( X, 0).unsqueeze(dim=-1)

    return X
