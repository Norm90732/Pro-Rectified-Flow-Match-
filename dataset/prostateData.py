#dataset/prostateData . py 
from torch.cuda._pin_memory_utils import pin_memory
from torch._inductor.runtime.cache_dir_utils import cache_dir
import os
import glob
import pandas as pd
import monai
from monai.transforms import (
    Compose,
    ScaleIntensityRangeD,
    Resized
)
from monai.data import PersistentDataset, DataLoader
from omegaconf import DictConfig
from monai.transforms import LoadImage, MapTransform
from sklearn.model_selection import train_test_split
import numpy as np
import SimpleITK as sitk
import torch
from typing import Tuple



def loadDicomSeries(seriesDir):
    reader = sitk.ImageSeriesReader()
    dicomNames = reader.GetGDCMSeriesFileNames(seriesDir)
    
    if len(dicomNames) == 0:
        dicomNames = sorted(glob.glob(os.path.join(seriesDir, "**/*.dcm"), recursive=True))

        if len(dicomNames) == 0:
            raise ValueError(f"No DICOM files found in {seriesDir} or its subdirectories")
    
    reader.SetFileNames(dicomNames)
    image = reader.Execute()
    
    array = sitk.GetArrayFromImage(image)
    array = np.transpose(array, (1, 2, 0))
    return array

def volumesToTripletSlices(volumeList):
    expandedList = []
    for volumeDict in volumeList:
        volumePath = volumeDict["image"]
        subjectID = volumeDict["subject_id"]
        
        fullVolume = loadDicomSeries(volumePath)
        Depth = fullVolume.shape[-1]
        
        for centerSliceIndex in range(1,Depth-1):
            expandedList.append({
                "image": volumePath,
                "center_slice_idx": centerSliceIndex,
                "subject_id": subjectID,
            })
        
    return expandedList



def obtainMRIdict(cfg:DictConfig):
   metDataDir = cfg.data.metaData
   prostateImageDir = cfg.data.prostateMRI
   
   df = pd.read_csv(metDataDir)
   
   mriDF = df[df['Modality'] == 'MR']
   
   t2MriDF = mriDF[mriDF["Series Description"].str.contains("t2",case=False,na=False)].copy()
   
   collectionName = "Prostate-MRI-US-Biopsy"
   t2MriDF['Series Dir Relative'] = t2MriDF['File Location'].apply(
        lambda x: x.lstrip('./').replace(f'{collectionName}/', '', 1)
    )
    
   t2MriDF['Absolute Series Path'] = t2MriDF['Series Dir Relative'].apply(
        lambda x: os.path.join(prostateImageDir, x)
    )
    
   def seriesDirDicom(path):
       if os.path.isfile(path):
           path = os.path.dirname(path)
       if os.path.exists(path):
            dcmFiles = [f for f in os.listdir(path) if f.endswith('.dcm')]
            if len(dcmFiles) > 0: 
                return path
            return None
   
   t2MriDF['Series_Dir_Fixed'] = t2MriDF['Absolute Series Path'].apply(seriesDirDicom)
   t2MriDF = t2MriDF[t2MriDF['Series_Dir_Fixed'].notna()].copy()

    
   uniqueVolumesDf = t2MriDF.drop_duplicates(subset=['Series_Dir_Fixed'])
   uniqueVolumePaths = uniqueVolumesDf[[
        'Series_Dir_Fixed', 'Subject ID'
    ]].rename(columns={'Series_Dir_Fixed': 'image', 'Subject ID': 'subject_id'}).to_dict('records')
   
   trainingValidation, testVolumes = train_test_split(
       uniqueVolumePaths,
       test_size=cfg.data.trainingValidationTest,
       random_state = cfg.seed,
   )
   
   trainingVolumes, validationVolumes = train_test_split(
       trainingValidation,
       test_size = cfg.data.trainingValidation,
       random_state= cfg.seed
   )

   print("Total Train Volumes", len(trainingVolumes))
   print("Total Validation Volumes", len(validationVolumes))
   print("TotalTestVolumes",len(testVolumes))
   
   trainList = volumesToTripletSlices(trainingVolumes)
   validationList = volumesToTripletSlices(validationVolumes)
   testList = volumesToTripletSlices(testVolumes)
   
   print("Final Total Train Volumes", len(trainList))
   print("Final Total Validation Volumes", len(validationList))
   print("Final TotalTestVolumes",len(testList))
   

   return trainList,validationList,testList
        
class CustomDicomLoadD(MapTransform):
    """Loads the volume using the custom SITK logic."""
    def __call__(self, data):
        data = dict(data)
        volumePath = data["image"]
        volumeArray = loadDicomSeries(volumePath)
        data["image_volume"] = torch.from_numpy(volumeArray).unsqueeze(0).float()
        
        
        return data


class ExtractSliceTripletsD(MapTransform):
    def __call__(self,data):
        data = dict(data)
        
        img = data["image_volume"]
        centerIndex = data["center_slice_idx"]
        
        img3D = img.squeeze(0)
        sliceMinusOne = img3D[..., centerIndex - 1]
        sliceI = img3D[..., centerIndex]
        slicePlusOne = img3D[..., centerIndex + 1]
        
        data["sliceMinusOne"] = sliceMinusOne.unsqueeze(0).float()
        data["sliceI"] = sliceI.unsqueeze(0).float()
        data["slicePlusOne"] = slicePlusOne.unsqueeze(0).float()
        
        del data["image_volume"]
        del data["center_slice_idx"]
        
        return data
class ZScoreNormalizeD(MapTransform):
    def __init__(self, keys, percentile_clip=True, lower=0.5, upper=99.5):
        super().__init__(keys)
        self.percentile_clip = percentile_clip
        self.lower = lower
        self.upper = upper
        
    def __call__(self, data):
        data = dict(data)
        
        sliceMinusOne = data[self.keys[0]]
        sliceI = data[self.keys[1]]
        slicePlusOne = data[self.keys[2]]
        
        allSlices = torch.cat([sliceMinusOne, sliceI, slicePlusOne], dim=0)
        
       
        nonzeroValues = allSlices[allSlices > 0]
        
        if nonzeroValues.numel() > 0:
            if self.percentile_clip:
                lower_bound = torch.quantile(nonzeroValues, self.lower / 100.0).item()
                upper_bound = torch.quantile(nonzeroValues, self.upper / 100.0).item()
                
               
                for key in self.keys:
                    data[key] = torch.clamp(data[key], lower_bound, upper_bound)
                
                
                allSlices = torch.cat([data[self.keys[0]], data[self.keys[1]], 
                                      data[self.keys[2]]], dim=0)
                nonzeroValues = allSlices[allSlices > 0]
            
           
            meanVal = nonzeroValues.mean().item()
            stdVal = nonzeroValues.std().item()
            
            
            if stdVal < 1e-8:
                stdVal = 1.0
            
            
            for key in self.keys:
                sliceTensor = data[key]
                normalized = (sliceTensor - meanVal) / stdVal
                data[key] = normalized
        else:
            pass
        
        return data
            
def CreatePersistentDataSet(cfg: DictConfig, trainList, validationList, testList):
    TargetSliceSize = (256,256)
    totalTransforms = Compose([
        CustomDicomLoadD(keys=["image"]), 
        ExtractSliceTripletsD(keys=["image_volume", "center_slice_idx"]),
        Resized(keys=['sliceMinusOne', 'sliceI', 'slicePlusOne'], spatial_size=TargetSliceSize, mode="bilinear"),
        ZScoreNormalizeD(keys=['sliceMinusOne', 'sliceI', 'slicePlusOne'], percentile_clip=True, lower=0.5, upper=99.5)
    ]) 
    
    
    trainingDataset = PersistentDataset(
        data=trainList,
        transform=totalTransforms,
        cache_dir=cfg.data.persistentDataDir.trainingDir
    )
    
    validationDataset = PersistentDataset(
        data=validationList,
        transform=totalTransforms,
        cache_dir=cfg.data.persistentDataDir.validationDir
    )
    
    testingDataset = PersistentDataset(
        data=testList,
        transform=totalTransforms,
        cache_dir=cfg.data.persistentDataDir.testingDir
    )
    
    
    
    
    print("Finished Persistent Definitions")
    
    cacheTraining = DataLoader(trainingDataset,batch_size=1,shuffle=False,num_workers=0)
    totalTrain = len(cacheTraining)
    for i, _ in enumerate(cacheTraining):
        print(f"Current Cache Train {i}/{totalTrain}")
    print("Finished Training Cache")
    
    
    cacheValidation = DataLoader(validationDataset,batch_size=1,shuffle=False,num_workers=0)
    totalVal = len(cacheValidation)
    for i, _ in enumerate(cacheValidation):
        print(f"Current Cache Train {i}/{totalVal}")
        
    print("Finished Validation Cache")
    
    cacheTest = DataLoader(testingDataset,batch_size=1,shuffle=False,num_workers=0)
    totalTest = len(cacheTest)
    for i, _ in enumerate(cacheTest):
        print(f"Current Cache Test {i}/{totalTest}")
    print("Finished Test")

def fetchLoaders(cfg:DictConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    TargetSliceSize = (256,256)
    trainList,validationList,testList = obtainMRIdict(cfg)
    totalTransforms = Compose([
        CustomDicomLoadD(keys=["image"]), 
        ExtractSliceTripletsD(keys=["image_volume", "center_slice_idx"]),
        Resized(keys=['sliceMinusOne', 'sliceI', 'slicePlusOne'], spatial_size=TargetSliceSize, mode="bilinear"),
        ZScoreNormalizeD(keys=['sliceMinusOne', 'sliceI', 'slicePlusOne'], percentile_clip=True, lower=0.5, upper=99.5)
    ]) 
    
    
    trainingDataset = PersistentDataset(
        data=trainList,
        transform=totalTransforms,
        cache_dir=cfg.data.persistentDataDir.trainingDir
    )
    
    validationDataset = PersistentDataset(
        data=validationList,
        transform=totalTransforms,
        cache_dir=cfg.data.persistentDataDir.validationDir
    )
    
    testingDataset = PersistentDataset(
        data=testList,
        transform=totalTransforms,
        cache_dir=cfg.data.persistentDataDir.testingDir
    )
    
    
    batchSize = cfg.batchSize
    prefetchFactor = cfg.preFetch
    numWorkers = cfg.numDataWorkers
    
    trainingDataloader = DataLoader(
        trainingDataset,
        num_workers=numWorkers,
        batch_size=batchSize,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetchFactor
    )
    validationDataloader = DataLoader(
        validationDataset,
        num_workers=numWorkers,
        batch_size=batchSize,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetchFactor
    )
    
    testingDataloader = DataLoader(
        testingDataset,
        num_workers=numWorkers,
        batch_size=batchSize,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetchFactor
    )
    
    return trainingDataloader,validationDataloader,testingDataloader

def visualizeDataloader(cfg:DictConfig) -> None:
    import matplotlib.pyplot as plt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainingDataloader,validationDataloader,testingDataloader = fetchLoaders(cfg=cfg)
    slicePlusOne=None
    sliceI= None
    sliceMinusOne= None
    for batch in trainingDataloader:
        slicePlusOne = batch["slicePlusOne"].to(device)
        sliceI = batch["sliceI"].to(device)
        sliceMinusOne = batch["sliceMinusOne"].to(device)
        break
    
    
    minusOneSlice = sliceMinusOne[0, 0].cpu().numpy()
    sliceI = sliceI[0, 0].cpu().numpy() 
    plusOneSlice = slicePlusOne[0, 0].cpu().numpy()
    
    print("Visualization")
    
    fig,axes = plt.subplots(1,3,figsize=(15,15))
    
    axes[0].imshow(minusOneSlice,cmap="gray")
    axes[0].set_title("Minus One Slice")
    axes[0].axis("off")
    
    
    axes[1].imshow(sliceI,cmap='gray')
    axes[1].set_title("Slice I")
    axes[1].axis("off")
    
    axes[2].imshow(plusOneSlice,cmap="gray")
    axes[2].set_title("Plus one Slice")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.savefig("Training Visual.png") 
    
    
    
    
        
        
        
if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig 
    @hydra.main(version_base=None, config_path="../configs", config_name="config")
    def main(cfg:DictConfig) -> None:
        visualizeDataloader(cfg=cfg)
    main()
