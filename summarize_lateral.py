# read logs.
import pickle
import os
from itertools import product
from os.path import isfile
results = []
for trial in range(5):
    for lateral in ['default','none','rand']:#' random']:
        for let_inhibition_learn in [False,True]:
            for the_arch in ['dense','wrn']:
                for dataset in ['100','10']:
                
                    if lateral=='none' and let_inhibition_learn: 
                        # nothing to learn where there's no "inhibition"
                        continue 
                                        
                    if the_arch == 'dense':
                        curOutDir = '~/checkpoints/cifar{}_inhibition/densenet-bc-100-12-IN_{}_IL_{}-run_{}'
                        epochs = 300
                    elif the_arch == 'wrn':
                        curOutDir = '~/checkpoints/cifar{}_inhibition/wrn-28-10-IN_{}_IL_{}-run_{}'
                    curOutDir = curOutDir.format(dataset,lateral,let_inhibition_learn,trial)
                    ff = os.path.join(curOutDir,'log.txt')
                    ff = os.path.expanduser(ff)
                    if isfile(ff):
                        results.append(dict(trial=trial,lateral=lateral,let_inhibition_learn=let_inhibition_learn,arch=the_arch,dataset=dataset,flines = open(ff).readlines()))

pickle.dump(results,open('/data/results.pkl','w'))


# read logs for full runs.

results = []
for trial in range(5):
    for lateral in ['default','none','rand']:#' random']:
        for let_inhibition_learn in [False,True]:
            for the_arch in ['dense','wrn']:
                for dataset in ['100','10']:
                
                    if lateral=='none' and let_inhibition_learn: 
                        # nothing to learn where there's no "inhibition"
                        continue 
                                        
                    if the_arch == 'dense':
                        curOutDir = '~/checkpoints/cifar{}_inhibition/densenet-bc-100-12-IN_{}_IL_{}-run_{}'
                        epochs = 300
                    elif the_arch == 'wrn':
                        curOutDir = '~/checkpoints/cifar{}_inhibition/wrn-28-10-IN_{}_IL_{}-run_{}'
                    
                    curOutDir = curOutDir.format(dataset,lateral,let_inhibition_learn,trial)
                    curOutDir+='_FULL'
                    ff = os.path.join(curOutDir,'log.txt')
                    ff = os.path.expanduser(ff)
                    if isfile(ff):
                        results.append(dict(trial=trial,lateral=lateral,let_inhibition_learn=let_inhibition_learn,arch=the_arch,dataset=dataset,flines = open(ff).readlines()))
                    else:
                        print 'file ff is missing!'

pickle.dump(results,open('/data/results_full.pkl','w'))

