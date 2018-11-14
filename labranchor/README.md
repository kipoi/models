## LSTM Branchpoint Retriever (LaBranchoR)

LaBranchoR predicts RNA splicing branchpoints using a Long Short-Term Memory network.

### Description

A model to predict branchpoint from sequence by Paggi et al.. 2017(http://bejerano.stanford.edu/labranchor/).

Input of the model is one-hot-encoded 70bp sequence upstream a 3' acceptor site. Output are scores per basepair the probability to be a branchpoint. 
