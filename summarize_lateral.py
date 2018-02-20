# read logs.
import pickle
import os
from os.path import isfile
results = []
all_opts = list(product(trials,laterals,let_inhibition_learns,the_archs,datasets))
for trial,lateral,let_inhibition_learn,the_arch,dataset in all_opts:
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
        results.append(dict(trial=trial,let_inhibition_learn=let_inhibition_learn,arch=the_arch,dataset=dataset,flines = open(ff).readlines()))

pickle.dump(results,open('/data/results.pkl','w'))