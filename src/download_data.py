import argparse
import deepchem as dc
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_spec", default="engd", help="Image specification to load")
    parser.add_argument("--save_dir", help='Directory to save the dataset')
    parser.add_argument('--dataset', help='chembl/ tox21/ hiv/ freesolv')
    args = parser.parse_args()

    datasets_list = ['chembl', 'tox21', 'hiv', 'freesolv']
    '''load_fn = dc.molnet.load_chembl25
    
    
    load_fn = dc.molnet.load_chembl25
    tasks, dataset, transformers = load_fn(featurizer="smiles2img", data_dir=DIRNAME, 
                                           save_dir=DIRNAME+'Featurized/', img_spec='engd', split="stratified")'''
    DIRNAME = 'Raman Data/'
    path = os.path.join(DIRNAME, args.dataset)
    
    if(args.dataset == 'chembl'):
        
        load_fn = dc.molnet.load_chembl25
        if(os.path.exists(path)):
            tasks, dataset, transformers = load_fn(featurizer="smiles2img", data_dir=path, 
                                            save_dir=path, img_spec=args.img_spec, split="stratified")
        else:
            os.mkdir(path)
            tasks, dataset, transformers = load_fn(featurizer="smiles2img", data_dir=path, 
                                            save_dir=path, img_spec=args.img_spec, split="stratified")

    elif(args.dataset == 'tox21'):
        load_fn = dc.molnet.load_tox21
        if(os.path.exists(path)):
            tasks, dataset, transformers = load_fn(featurizer="smiles2img", data_dir=path, 
                                            save_dir=path, img_spec=args.img_spec)
        else:
            os.mkdir(path)
            tasks, dataset, transformers = load_fn(featurizer="smiles2img", data_dir=path, 
                                            save_dir=path, img_spec=args.img_spec)
    elif(args.dataset == 'hiv'):
        load_fn = dc.molnet.load_hiv
        if(os.path.exists(path)):
            tasks, dataset, transformers = load_fn(featurizer="smiles2img", data_dir=path, 
                                            save_dir=path, img_spec=args.img_spec)
        else:
            os.mkdir(path)
            tasks, dataset, transformers = load_fn(featurizer="smiles2img", data_dir=path, 
                                            save_dir=path, img_spec=args.img_spec)

    elif(args.dataset == 'freesolv'):
        print("NA :/")
        pass
        
    else:
        print("Please select one of the following datasets: ", datasets_list)
    
    
    
if __name__ == "__main__":
    main()
    
