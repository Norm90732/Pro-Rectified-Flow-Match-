import torch
import torch.nn as nn
import torch.nn.functional as F 
from omegaconf import DictConfig,OmegaConf
from dataset.prostateData import fetchLoaders
from models.RectifiedFlow.RecFlow import RectifiedConditionalFlow
from torchdiffeq import odeint #solver Library 
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity #LPIPS
from torchmetrics.image import StructuralSimilarityIndexMeasure #SSIM
from torchmetrics.image import PeakSignalNoiseRatio #PSNR
from torchmetrics.image.fid import FrechetInceptionDistance #FID
#Implementation of validation Scripts To test performance 
class VectorRectified(nn.Module):
    def __init__(self,checkpoint:str,cfg:DictConfig):
        super().__init__()
        
        self.velocityModel = RectifiedConditionalFlow(
            input_channel=cfg.model.input_channels,
            output_channel=cfg.model.output_channels, 
            timeEmbedDim=cfg.model.timeEmbedDim,
            )
        
        checkpoint = torch.load(checkpoint,map_location="cpu",weights_only=False)
        self.velocityModel.load_state_dict(checkpoint["model_state_dict"])
        self.velocityModel.eval()
        #Estimating Velocity Field 
        for param in self.velocityModel.parameters(): #precaution
            param.requires_grad=False
    def forward(self,t,x):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=x.device, dtype=x.dtype)
        
        if t.dim() == 0: 
            t = t.expand(x.shape[0])  
        
        middle = self.velocityModel.forward(x,t)
        zeros = torch.zeros_like(middle)
        velocityFull = velocityFull = torch.cat([zeros, middle, zeros], dim=1)
        return velocityFull


    
    
def chooseBestCheckpointValidation(cfg:DictConfig):
    import os
    import glob
    from numpy import argmin
    _, validationLoader, _ = fetchLoaders(cfg=cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpointDir = cfg.evaluation.checkpointDir
    checkpointPattern = os.path.join(checkpointDir, "checkpoint_*", "checkpoint_*.pt")
    checkpoints: list = sorted(glob.glob(checkpointPattern))
    numSteps = cfg.evaluation.validationPicker.steps
    solver:str = cfg.evaluation.validationPicker.solver
    lpipScoreList = []
    for checkpoint in checkpoints:
        #Defining Loop 
        model = VectorRectified(checkpoint,cfg=cfg)
        model.to(device)
        lpipsMetric = LearnedPerceptualImagePatchSimilarity(net_type="alex",normalize=True).to(device)
        lpipsMetric.reset()
        with torch.no_grad():
            for batchIdx, batch in enumerate(validationLoader):
                slicePlusOne = batch["slicePlusOne"].to(device)
                sliceI = batch["sliceI"].to(device)
                sliceMinusOne = batch["sliceMinusOne"].to(device)
                
                #Estimate Forward 
                batchSize = sliceI.shape[0]
                X0 = torch.randn_like(sliceI,device=device)
                
                #Concat
                concat = torch.cat([slicePlusOne,X0,sliceMinusOne],dim=1)
                times = torch.linspace(0,1,steps=numSteps,device=device)
                OdeResult = odeint(model,concat,times,method=solver)
                
                Xt = OdeResult[-1]
                XtMiddle = Xt[:, 1:2, :, :]
                XtMiddle = torch.clamp(XtMiddle, 0, 1)
                sliceI = torch.clamp(sliceI,0,1)
                
                rgbXtMiddle = XtMiddle.repeat(1,3,1,1)
                targetRgb = sliceI.repeat(1,3,1,1)
                #LPIPS Computation for Selection 
                lpipsMetric.update(rgbXtMiddle,targetRgb)
            
        
        lpipsScore = lpipsMetric.compute().item()
        print(F"Checkpoint{checkpoint} LPIPS VAL: {lpipsScore:.4f}")
        lpipScoreList.append(lpipsScore)
    
    
    #Best checkpoint
    bestNum = argmin(lpipScoreList)
    bestCheckpoint = checkpoints[bestNum]

    print(f"Best Checkpoint:",bestCheckpoint)
    
        
def TestingFunction(cfg:DictConfig):
    from pathlib import Path
    import os 
    import json
    import matplotlib.pyplot as plt
    import time
    _, _, testingLoader = fetchLoaders(cfg=cfg)
    bestCheckpoint = cfg.evaluation.bestCheckpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VectorRectified(checkpoint=bestCheckpoint,cfg=cfg)
    model.to(device)
    
    numSteps = cfg.evaluation.solver.steps
    solver:str = cfg.evaluation.solver.type
    
    #Metric Init
    lpipsMetric = LearnedPerceptualImagePatchSimilarity(net_type="alex",normalize=True).to(device)
    SSIMMetric = StructuralSimilarityIndexMeasure(data_range=None).to(device)
    PSNRMetric = PeakSignalNoiseRatio(data_range=16).to(device)
    
    lpipsGlobal = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
    SSIMGlobal = StructuralSimilarityIndexMeasure(data_range=None).to(device)
    PSNRGlobal = PeakSignalNoiseRatio(data_range=16).to(device)  
    FIDMetric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
     
    
    allScores = [] #lpips, index, psnr, ssim, FID
    
    inferenceTimes= []
    totalImages = 0
    
    #Warm Up for timing (suggested by Fei)
    with torch.no_grad():
        for i, batch in enumerate(testingLoader):
            if i >=3:
                break
            slicePlusOne = batch["slicePlusOne"].to(device)
            sliceI = batch["sliceI"].to(device)
            sliceMinusOne = batch["sliceMinusOne"].to(device)
  
            batchSize = sliceI.shape[0]
            
        
            X0 = torch.randn_like(sliceI,device=device)
            concat = torch.cat([slicePlusOne,X0,sliceMinusOne],dim=1)
            times = torch.linspace(0,1,steps=numSteps,device=device)
            OdeResult = odeint(model,concat,times,method=solver)
            Xt = OdeResult[-1]
    
    with torch.no_grad():
    #Obtain Best index by Lpips 
        for batchIdx, batch in enumerate(testingLoader):
            slicePlusOne = batch["slicePlusOne"].to(device)
            sliceI = batch["sliceI"].to(device)
            sliceMinusOne = batch["sliceMinusOne"].to(device)

            batchSize = sliceI.shape[0]
            totalImages += batchSize
            
            X0 = torch.randn_like(sliceI,device=device)
            concat = torch.cat([slicePlusOne,X0,sliceMinusOne],dim=1)
            times = torch.linspace(0,1,steps=numSteps,device=device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            startTime = time.time()
            OdeResult = odeint(model,concat,times,method=solver)
            
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            endTime = time.time()
            batchTime = endTime-startTime
            inferenceTimes.append(batchTime)
            
            Xt = OdeResult[-1]
            XtMiddle = Xt[:, 1:2, :, :]
            
            XtMiddleClamp = torch.clamp(XtMiddle, 0, 1)
            sliceIClamp = torch.clamp(sliceI,0,1)
                
            rgbXtMiddle = XtMiddleClamp.repeat(1,3,1,1)
            targetRgb = sliceIClamp.repeat(1,3,1,1)
            #Global
            lpipsGlobal.update(rgbXtMiddle,targetRgb)
            SSIMGlobal.update(XtMiddle,sliceI)
            PSNRGlobal.update(XtMiddle,sliceI)
            FIDMetric.update(rgbXtMiddle,real=False)
            FIDMetric.update(targetRgb,real=True)
            
            
            for i in range(batchSize):
                generatedImg = XtMiddle[i].unsqueeze(0)
                targetImg = sliceI[i].unsqueeze(0)
                
                #lpips i need to repeat and make it rgb and clamp for lpips 
                
                generatedImgClamp = torch.clamp(generatedImg,0,1)
                targetImgClamp = torch.clamp(targetImg,0,1)
                
                generatedImgRGB = generatedImgClamp.repeat(1,3,1,1)
                targetImgRGB = targetImgClamp.repeat(1,3,1,1)
                
                lpipsScore = lpipsMetric(generatedImgRGB,targetImgRGB).item()
                ssimScore = SSIMMetric(generatedImg,targetImg).item()
                psnrScore = PSNRMetric(generatedImg,targetImg).item()
                
                
                globalIndex = batchIdx * batchSize + i
                
                allScores.append({
                    "lpips": lpipsScore,
                    "ssim": ssimScore,
                    "psnr": psnrScore,
                    "globalIDX": globalIndex,
                    "batchIDX": batchIdx,
                    "inBatchIDX": i
                })
            
    totalTime = sum(inferenceTimes)
    avgTimePerBatch = totalTime / len(inferenceTimes)
    avgTimePerImage = totalTime/totalImages
    globalLPIPS= lpipsGlobal.compute().item()
    globalSSIM = SSIMGlobal.compute().item()
    globalPSNR =  PSNRGlobal.compute().item()
    globalFID = FIDMetric.compute().item()
    
    allScores.sort(key=lambda x: x["lpips"])
    
    best30 = allScores[:30]
    worst30 = allScores[-30:]
    
    results = {
        "globalMetrics":
            {
                "lpips": globalLPIPS,
                "ssim":globalSSIM,
                "psnr": globalPSNR,
                "fid": globalFID
            },
        "inferenceTime": {
            "totalImages":totalImages,
            "totalTimeSeconds": totalTime,
            "avgTimePerImageSeconds": avgTimePerBatch,
            
        },
        "best30": best30,
        "worst30": worst30,
    }
    
    
        
    #Writing Visualization Save Images 
    
    
    imageSaveDir = cfg.evaluation.imageSave
    
    currentSolverStepDir = Path(imageSaveDir) / f"{solver}_{numSteps}"
    
    bestDir = currentSolverStepDir / "best"
    worstDir = currentSolverStepDir / "worst"
    
    bestDir.mkdir(parents=True,exist_ok=True)
    worstDir.mkdir(parents=True,exist_ok=True)
    
    jsonName = currentSolverStepDir / "results.json"
    with open(jsonName,"w") as f:
        json.dump(results,f,indent=2)
    
   
    
    with torch.no_grad():
        for rank, case in enumerate(best30,1):
            batchIdx = case["batchIDX"]
            inBatchIDX = case["inBatchIDX"]
            
            for bIdx, batch in enumerate(testingLoader):
                if bIdx != batchIdx:
                    continue
                
                sliceI = batch["sliceI"][inBatchIDX].to(device)
                slicePlusOne = batch["slicePlusOne"][inBatchIDX].to(device)
                sliceMinusOne = batch["sliceMinusOne"][inBatchIDX].to(device)
                
                X0 = torch.randn_like(sliceI.unsqueeze(0),device=device)
                concat = torch.cat([slicePlusOne.unsqueeze(0), X0, sliceMinusOne.unsqueeze(0)], dim=1)
                times = torch.linspace(0,1,steps=numSteps,device=device)
                OdeResult = odeint(model,concat,times,method=solver)
                generated = OdeResult[-1, 0, 1].cpu().numpy()
                
                target = sliceI[0].cpu().numpy()
                
                
                fig, axes = plt.subplots(1,2,figsize=(12,12))
                axes[0].imshow(target, cmap='gray')
                axes[0].set_title('Ground Truth', fontsize=14, fontweight='bold')
                axes[0].axis('off')
                axes[1].imshow(generated, cmap='gray')
                axes[1].set_title('Generated', fontsize=14, fontweight='bold')
                axes[1].axis('off')
                fig.suptitle(
                    f'Best Rank #{rank} (Index: {case["globalIDX"]})\n'
                    f'LPIPS: {case["lpips"]:.4f} | SSIM: {case["ssim"]:.4f} | PSNR: {case["psnr"]:.2f} dB',
                    fontsize=12, y=0.98
                )
                plt.tight_layout()
                filename = f"best_rank{rank:02d}_idx{case['globalIDX']:05d}.png"
                plt.savefig(bestDir / filename, dpi=150, bbox_inches='tight')
                plt.close()
                break
            
    with torch.no_grad():
        for rank, case in enumerate(worst30,1):
            batchIdx = case["batchIDX"]
            inBatchIDX = case["inBatchIDX"]
            
            for bIdx, batch in enumerate(testingLoader):
                if bIdx != batchIdx:
                    continue
                
                sliceI = batch["sliceI"][inBatchIDX].to(device)
                slicePlusOne = batch["slicePlusOne"][inBatchIDX].to(device)
                sliceMinusOne = batch["sliceMinusOne"][inBatchIDX].to(device)
                
                X0 = torch.randn_like(sliceI.unsqueeze(0),device=device)
                concat = torch.cat([slicePlusOne.unsqueeze(0), X0, sliceMinusOne.unsqueeze(0)], dim=1)
                times = torch.linspace(0,1,steps=numSteps,device=device)
                OdeResult = odeint(model,concat,times,method=solver)
                generated = OdeResult[-1, 0, 1].cpu().numpy()
                target = sliceI[0].cpu().numpy()
                
                
                fig, axes = plt.subplots(1,2,figsize=(12,12))
                axes[0].imshow(target, cmap='gray')
                axes[0].set_title('Ground Truth', fontsize=14, fontweight='bold')
                axes[0].axis('off')
                axes[1].imshow(generated, cmap='gray')
                axes[1].set_title('Generated', fontsize=14, fontweight='bold')
                axes[1].axis('off')
                fig.suptitle(
                    f'Worst Rank #{rank} (Index: {case["globalIDX"]})\n'
                    f'LPIPS: {case["lpips"]:.4f} | SSIM: {case["ssim"]:.4f} | PSNR: {case["psnr"]:.2f} dB',
                    fontsize=12, y=0.98
                )
                plt.tight_layout()
                filename = f"worst_rank{rank:02d}_idx{case['globalIDX']:05d}.png"
                plt.savefig(worstDir / filename, dpi=150, bbox_inches='tight')
                plt.close()
                break
        
    
    return results
    
    
        
if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig 
    @hydra.main(version_base=None, config_path="../configs", config_name="config")
    def main(cfg:DictConfig) -> None:
        TestingFunction(cfg=cfg)
    main()
    