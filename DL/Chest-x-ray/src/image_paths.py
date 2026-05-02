import pandas as pd 
import os 
import logging

def load_data(data_path):
    df=pd.read_csv(data_path)
    return df
def get_path(row):
    Data_dir=r"C:\Users\nice\Desktop\final project for college\DL\Chest-x-ray\data\Coronahack-Chest-XRay-Dataset"
    folder="train" if row["Dataset_type"]=="TRAIN" else "TEST"
    return os.path.join(Data_dir,folder,row["X_ray_image_name"])

def main():
    df=load_data(r"C:\Users\nice\Desktop\final project for college\DL\Chest-x-ray\data\Chest_xray_Corona_Metadata.csv")
    df['image_path']=df.apply(get_path, axis=1)
    df.to_csv(r"C:\Users\nice\Desktop\final project for college\DL\Chest-x-ray\data\spilt_data.csv")

main()