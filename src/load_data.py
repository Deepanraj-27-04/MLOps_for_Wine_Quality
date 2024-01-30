import os
from getData import read_params,get_data
import argparse

def load_save(config_path):
    config=read_params(config_path)
    df=get_data(config_path)
    new_cols=[col.replace(" ","_") for col in df.columns]
    raw_data_path=config["load_data"]["raw_dataset_csv"]
    
    
    df.to_csv(raw_data_path,sep=",",index=False,header=new_cols)
    
    
if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--config",default="params.yaml")
    parse_arg=args.parse_args()
    load_save(config_path=parse_arg.config)