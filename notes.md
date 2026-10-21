(Pdb) p max\_length
114                                                                                                                    

(Pdb) p input\_ids
tensor([[ 3220,  5798,  4159,   512,   436, 32463,  4877,  2127,   323,   247,
           747,  3448,  1566,   285]], device='cuda:0')

Hidden states after first pass:

(Pdb) p hidden\_states
tensor([[[-0.3105,  1.1016, -0.0789,  ..., -0.1013, -0.0513,  0.3782]]],
       device='cuda:0', dtype=torch.float16)
(Pdb) p hidden\_states.shape[0]
1
(Pdb) p hidden\_states.shape[1]
1
(Pdb) p hidden\_states.shape[2]


