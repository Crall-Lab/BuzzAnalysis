import os
for dir, subdir, files in os.walk('testNest'):
    for f in files:
        if '.csv' in f and 'worker' in f:
            y, d1, d2, d3, id, ext = f.split('-')
            new = '_'.join([id, '-'.join([d1, d2, d3]), ext.replace('_', '-')])
            os.rename(os.path.join(dir,f), os.path.join(dir,new))