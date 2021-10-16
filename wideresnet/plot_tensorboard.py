from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import pdb
def load_file(filepath):
    import os
    event_acc = EventAccumulator(os.path.expanduser(filepath))
    event_acc.Reload()
    # tags = event_acc.Tags()
    # result = {}
    return event_acc

import matplotlib.pyplot as plt
import numpy

def ylim_auto(percentile,data):
    lower = numpy.percentile(data, percentile[0])
    upper = numpy.percentile(data, percentile[1])
    plt.ylim([lower, upper]) 

def event_to_numpy(events):
    #arry = []
    max_size = 0
    for item in events:
        if max_size < item.step:
            max_size = item.step
    arry = numpy.zeros(shape=(max_size+1,))      
    for item in events:
        arry[item.step] = item.value
    return arry
        

import numpy as np
# object_methods = [method_name for method_name in dir(object)
#                   if callable(getattr(object, method_name))]    

    # ('solid', 'solid'),      # Same as (0, ()) or '-'
    #  ('dotted', 'dotted'),    # Same as (0, (1, 1)) or '.'
    #  ('dashed', 'dashed'),    # Same as '--'
    #  ('dashdot', 'dashdot')] 

import contextlib
import numpy as np

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

import itertools

# plot_tensorboard.plot_scalar(event_acc,['train_acc','val_acc'],xlabel='Epoch',ylabel='Accuracy',markevery=50,linestyle='solid',color=['r','g','b'],marker=['.','o'])
# plot_tensorboard.plot_scalar(event_acc,['train_acc','val_acc'],xlabel='Epoch',ylabel='Accuracy',markevery=50,linestyle='solid',color=['r','g','b'],marker=['.','s'])
# plot_tensorboard.plot_scalar(event_acc,['train_acc','val_acc'],xlabel='Epoch',ylabel='Accuracy',markevery=80,linestyle='solid',color=['r','g','b'],marker=['.','s'])    
def line_style_selector(num,**kwargs):                       
    kwargs.setdefault('linestyle', ['dashdot','dotted', 'dashed', 'solid'] )     
    kwargs.setdefault('color', ['g','c','b','r','m','y','k'] )         
    #kwargs.setdefault('linestyle', [ 'solid'] )           
    #kwargs.setdefault('marker', [".",",","o","v","^","<",">","s","p","P","*","h","+","x","X","D","d","|","_"]  )
    #kwargs.setdefault('marker', [".",",","o","v","^","<",">","s","p","P","*","+","x","d"]  )
    kwargs.setdefault('marker', [".",",","o","v","^","<",">","s","p","P","*","+","x","d"]  )       
    #kwargs.setdefault('markersize',['6'])

    # convert single element to a list for selection
    kwargs =   {k: v if isinstance(v,list) else [v]  for (k,v) in kwargs.items()}
    
    with temp_seed(1):
        selected_styles =   {k: np.random.choice(v,num,replace=False) if (len(v) >= num) else np.random.choice(v,num,replace=True) for (k,v) in kwargs.items()}                
        # some_key if condition else default_key 
        #     selected_styles = []
        #     for i in range(num):                
    dl = selected_styles
    ld = [{key:value[index] for key,value in dl.items()}
         for index in range(max(map(len,dl.values())))]            
    
    return ld

# plot_tensorboard.plot_scalar_simple(event_acc,['train_acc','val_acc'],xlabel='Epoch',ylabel='Accuracy',markevery=50)     
# def plot_scalar_simple(event_acc,tags,xlabel='xlabel',ylabel='ylabel',**kwargs):     
#     if isinstance(event_acc,str):
#         load_file(event_acc)    
#     #event_acc.Scalars()
#     selected_styles = line_style_selector(len(tags),**kwargs)     
#     for i in range(len(tags)):
#         events = event_acc.Scalars(tags[i])  
#         arry = event_to_numpy(events)        
#         #plt.plot(np.arange(len(arry))+1, arry, 'r--',  epoch_stats[:,0]+1,  epoch_stats[:,3], 'b--') 
#         plt.plot(np.arange(len(arry))+1, arry, ** selected_styles[i] ) 
#     plt.grid()
#     plt.ylabel(ylabel)
#     plt.xlabel(xlabel)        
#     plt.legend(tags)             
#     plt.ion();
#     plt.show();

# def forceAspect(ax,aspect=1):
#     im = ax.get_images()
#     extent =  im[0].get_extent()
#     ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

def forceAspect(ax,ratio=1.0):  
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)    

def set_aspect(fig,aspect_ratio):  
    ax_list = fig.axes
    for item in ax_list:
        #item.set_aspect(*args)
        forceAspect(item,aspect_ratio)  
    
def process_tags(tags):
    '''
     standard format:
    [(fileString1,tagString1), (fileString2,tagString2) ,...] 
     short cuts:
     if tags is not a list, it will be converted to a list with one element
     element 'train_acc' will be converted to ('1','train_acc')
     element (2,'train_acc')  will be converted to ('2','train_acc')
    '''
    
    if not isinstance(tags,list):
        tags = [tags]
    
    for i in range(len(tags)):
        if not isinstance(tags[i],tuple):
            tags[i] = ('1',tags[i])
        if not isinstance(tags[i][0],str):
            tags[i][0] = str(tags[i][0])

    return tags
        
def draw_rect(plt,ylim,xlim,**kwargs):     
    import matplotlib.patches as patches
    rect=patches.Rectangle( [xlim[0],ylim[0]], xlim[1]-xlim[0], ylim[1]-ylim[0], **kwargs )
    currentAxis = plt.gca()
    currentAxis.add_patch(rect) 


def get_tensorboard_data(event_acc,tags):
    '''    
    get_tensorboard_data 
    '''
    if not isinstance(event_acc,dict):
        event_acc = {'1':event_acc}
    
    tags=process_tags(tags)

    # load files
    for k,v in event_acc.items():          
        if isinstance(v,str):
            event_acc[k] = load_file(v)
         
    all_data = []  
    for i in range(len(tags)): 
        fileString = tags[i][0]
        tagString  = tags[i][1]
        events = event_acc[fileString].Scalars(tagString)     
        arry = event_to_numpy(events)
        all_data.append(arry)
        
    return (all_data, event_acc, tags)      


def plot_scalar(event_acc,tags,**kwargs):
    '''    
    Example:
    plot_tensorboard.plot_scalar({'1':event_acc1,'2',event_acc2},['train_acc',(2,'val_acc')],xlabel='Epoch',ylabel='Accuracy',markevery=50)       
    '''
    #
    if isinstance(tags, np.ndarray) or  isinstance( tags[0], np.ndarray):
        # can pass data directly through tags     
        all_data = tags
    else:
        all_data, event_acc, tags =  get_tensorboard_data(event_acc,tags)    
    
    plot_scalar_general(all_data,**kwargs)    

    
def plot_scalar_general(all_data,legends=None,ylabel='ylabel',xlabel='xlabel',ylim=None,xlim=None,**kwargs):          

    # import pdb; pdb.set_trace()  
    
    if not isinstance(all_data,list):
        all_data = [all_data]
    
    selected_styles = line_style_selector(len(all_data),**kwargs)      

    for i in range(len(all_data)): 
        #plt.plot(np.arange(len(arry))+1, arry, 'r--',  epoch_stats[:,0]+1,  epoch_stats[:,3], 'b--')
        arry = all_data[i]
        plt.plot(np.arange(len(arry))+1, arry, ** selected_styles[i] )

    # esthetics
    ax_ = plt.gca() 
    ax_.spines['right'].set_color((.8,.8,.8))
    ax_.spines['top'].set_color((.8,.8,.8))           

    plt.grid(linestyle='--')         
    if xlim is not None:
        plt.xlim(xlim)
    
    if ylim is not None: 
        if hasattr(ylim, '__call__'):
            # automatic ylim   , not yet debugged   
            ylim(  numpy.concatenate(all_data, axis=0) )
        else:
            plt.ylim(ylim)
        
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if legends is not None:
        plt.legend(legends) 
    plt.ion();
    plt.show();

    #return all_data