import torch        

def compute_norms(param_list,p):
    norms = []
    product=1
    for item in param_list:
        if item is not None:
            #   print(item) 
            val=torch.norm(item, p)
            product=product*val
        else:
            val=None
        norms.append(val)
    
    return norms, product
    
def get_complexity_weights(model):
    conv_filters = []
    bn_std = []
    bn_scaling = []
    
    conv_filters_names = []
    bn_std_names = []
    bn_scaling_names = []
    
    for name, mod in model.named_modules():
        #print(name, mod)
        if isinstance(mod, torch.nn.modules.conv.Conv2d):
            conv_filters.append(mod.weight.data)
            conv_filters_names.append(name)
        elif isinstance(mod, torch.nn.modules.batchnorm.BatchNorm2d):
            bn_std.append(torch.sqrt(mod.running_var)) # mod.running_var--variance; mod.runnining_mean--mean.
            if mod.weight is not None:
                bn_scaling.append(torch.sqrt(mod.weight))
            else:
                bn_scaling.append(torch.ones(1))               
            bn_std_names.append(name + '_std')
            bn_scaling_names.append(name + '_scaling')     
    # print(conv_filters, bn_std, bn_scaling)
    return conv_filters, bn_std, bn_scaling, conv_filters_names, bn_std_names, bn_scaling_names

# rho,A,B,C, A_names, B_names, C_names =  complexity.get_complexities(net,'fro')      

def get_complexities(model,p): 
    A, B, C, A_names, B_names, C_names = get_complexity_weights(model)
    A, prod_A = compute_norms(A,p) # product of norms of weight matrices
    B, prod_B = compute_norms(B,p)
    C, prod_C = compute_norms(C,p)
    rho = (prod_A * prod_C) / prod_B
    #  conv_filters, bn_std, bn_scaling
    return rho, A, B, C, A_names, B_names, C_names

# prod_rho without BN
def get_complexities_2(model,p): 
    A, B, C, A_names, B_names, C_names = get_complexity_weights(model)
    A, prod_A = compute_norms(A,p) # product of norms of weight matrices
    B, prod_B = compute_norms(B,p)
    C, prod_C = compute_norms(C,p)
    rho = prod_A
    #  conv_filters, bn_std, bn_scaling
    return rho, A, A_names
def get_complexities_conv(model,p):     
    A,B,C, A_names, B_names, C_names = get_complexity_weights(model)
    A,prod_A = compute_norms(A,p)
    # B,prod_B = compute_norms(B,p)
    # C,prod_C = compute_norms(C,p)
    # rho = (prod_A * prod_C) / prod_B
    rho = prod_A     
    #  conv_filters  
    return rho, A , A_names

def get_complexities_grads_conv(model,p):                   
    A,B,C, A_names, B_names, C_names =get_complexity_grads(model)     
    A,prod_A = compute_norms(A,p)
    # B,prod_B = compute_norms(B,p)
    # C,prod_C = compute_norms(C,p)
    # rho = (prod_A * prod_C) / prod_B
    rho = prod_A     
    #  conv_filters  
    return rho, A , A_names


def get_complexity_grads(model):
    conv_filters = []
    bn_std = []
    bn_scaling = []    
    conv_filters_names = []
    bn_std_names = []
    bn_scaling_names = []
    
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.modules.conv.Conv2d):
            conv_filters.append(mod.weight.grad.data)      
            conv_filters_names.append(name)
        elif isinstance(mod, torch.nn.modules.batchnorm.BatchNorm2d):
            pass  
            # bn_std.append(torch.sqrt(mod.running_var))
            # if mod.weight.grad.data is not None:
            #     bn_scaling.append(torch.sqrt(mod.weight.grad.data))
            # else:
            #     bn_scaling.append(torch.zeros(1))                 
            # bn_std_names.append(name + '_std')
            # bn_scaling_names.append(name + '_scaling')     
    
    return conv_filters, bn_std, bn_scaling, conv_filters_names, bn_std_names, bn_scaling_names


import torch.nn as nn
import torch.nn.functional as F
import torch

import extarget
  
def compnorm_periter(model,init_scale,bn_affine=False):
    init_0_scale, init_1_scale =  extarget.parse_init_scale(init_scale)
    
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            #if compnorm_type == 'periter':
            if hasattr( m,'lastLayer') and m.lastLayer:
                init_scale__ = init_1_scale
            else:
                init_scale__ = init_0_scale            

            rho = torch.norm(m.weight.data,2)
            m.weight.data = init_scale__ * m.weight.data / rho
            m.bias.data   = init_scale__ * m.bias.data / rho
            
        elif isinstance(m, nn.BatchNorm2d):
            # assert(not bn_affine)
            # if bn_affine is False:
            #     m.weight = None
            #     m.bias   = None
            #     m.affine = False
            # else:
            #     m.affine = True
            #     m.weight = nn.Parameter(m.running_var.clone().detach(), requires_grad=True)        
            #     m.weight.data.fill_(1)
            #     m.bias   = nn.Parameter(m.running_var.clone().detach(), requires_grad=True)                
            #     m.bias.data.zero_()
            pass
        
        elif isinstance(m, nn.Linear):
            #if compnorm_type == 'periter':
            
            if hasattr( m,'lastLayer') and m.lastLayer:
                init_scale__ = init_1_scale
            else:
                init_scale__ = init_0_scale            

            rho =  torch.norm(m.weight.data,2)
            m.weight.data = init_scale__ * m.weight.data / rho
            m.bias.data   = init_scale__ * m.bias.data / rho            
            




    
    
#
# import numpy as np
# i=5;r=3;pp={'xsinx':i*np.sin(i/r), 'xcosx':i*np.cos(i/r), 'tanx': np.tan(i/r)}

# i=5;r=3;
# import numpy as np
# {'xsinx':i*np.sin(i/r), 'xcosx':i*np.cos(i/r), 'tanx': np.tan(i/r)}


# keys = ['a', 'b', 'c']
# values = [1, 2, 3]
# dictionary = dict(zip(keys, values))

# print(dictionary)

# result2 = ['conv_' + str(x) for x in numpy.arange(10)]
