# This python script depends on nifti2dicom

# nifti2dicom can be installed by the following two methods
    # reference https://www.jianshu.com/p/bb075bdfdf6b
    # 1. 'sudo apt install nifti2dicom'
    # 2. build from the source code at https://github.com/biolab-unige/nifti2dicom
    # command line in shell: nifti2dicom -i FLAIR.nii.gz -o FLAIR_DICOM -a 123456

# two ways to use this python script
    # python nii2dicom.py -p ${path2nii} 
    # python nii2dicom.py -d ${directory} [-i ${target-string1 target-string2 ...}]

# in way1, the script will directly convert the single nii file specified by the ${path2nii} to dicom
# in way2, the script will look for all nii files under $directory and convert all or those with target-strings if specified by -i. 
# One example for way2: python nii2dicom.py -d /mnt/data/public/MRI-TCGA -i N4 WSnorm

import os
import re
os.system('pip install pathlib --user')
from pathlib import Path


def convert1file(nii_fp: str, test = False):

    pwd = os.getcwd()
    nii_fp = Path(nii_fp)
    this_dir = nii_fp.parent
    os.chdir(this_dir)
    nii_fn = nii_fp.name
    dcm_dir = re.sub(r'.nii(\.gz)?', '', nii_fp.stem) + '_DCM'
    
    command = 'nifti2dicom -i %s -o %s -a 123456' %(nii_fn, dcm_dir)
    print('execute:', command)
    if not test: os.system(command)
    print('Done, please find dcm at ', str(this_dir/dcm_dir))
    os.chdir(pwd)


def run(args):
    assert args.dir or args.path, \
         ('Please give at least one type of path to convert: --dir or --path')
    
    target_strs = args.include
    single_fp = args.path
    data_rt = args.dir

    if single_fp:
        convert1file(single_fp)

    if data_rt:
        
        fps_list = []
        for subr, subd, subf in os.walk(data_rt, followlinks=True):
            if len(subf) > 0:
                fps = [os.path.join(subr, f) for f in subf if 'nii' in f]
                fps_list.extend(fps)
        print('In %s found %d nii files' %(data_rt, len(fps_list)))
        if target_strs: 
            target_strs = list(target_strs) 
            fps_list = [p for p in fps_list if all([str(s) in p for s in target_strs])]

        if target_strs: print('\t %d files contain %s' %(len(fps_list), target_strs))

        for fp in fps_list:
            convert1file(fp)

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Convert Nii to Dicom')

    parser.add_argument('-d', '--dir', default = None, help='directory which contains niis', type=str)
    parser.add_argument('-p', '--path', default = None, help='absolute path to 1 nii', type=str)
    parser.add_argument('-i', '--include', nargs='+', default = None, help='string included in targeted files')
    args = parser.parse_args()
    print('inputs: ', args)
    run(args)
