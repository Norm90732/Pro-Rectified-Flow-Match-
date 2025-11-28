import torch
import torch.nn as nn
import torch.nn.functional as F 
from omegaconf import DictConfig,OmegaConf
from dataset.prostateData import fetchLoaders
from models.DDPM.DDPM import ConditionalDDPM
from models.DDPM.DDIM import ConditionalDDIM
from testing.ddpmTest import DDPMSampler
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity #LPIPS
from torchmetrics.image import StructuralSimilarityIndexMeasure #SSIM
from torchmetrics.image import PeakSignalNoiseRatio #PSNR
from torchmetrics.image.fid import FrechetInceptionDistance #FID


class DDIMSampler(DDPMSampler):
    """Wrapper for DDIM sampling during inference"""
    def __init__(self, checkpoint: str, cfg: DictConfig):
        nn.Module.__init__(self)
        self.model = ConditionalDDIM(cfg=cfg)
        
        checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model.load_state_dict(checkpoint_data["model_state_dict"])
        self.model.eval()
        
        for param in self.model.parameters():
            param.requires_grad = False
    
        self.scheduler = self.model.scheduler
        self.num_inference_steps = cfg.evaluation.solver.steps  
        
    def sample(self, slicePlusOne, sliceMinusOne, num_inference_steps=None):
        device = slicePlusOne.device
        batchSize = slicePlusOne.shape[0]
        
        if num_inference_steps is None:
            num_inference_steps = self.num_inference_steps
    
        sample = torch.randn_like(slicePlusOne)
    
        condition = torch.cat([slicePlusOne, sliceMinusOne], dim=1)
        
        self.scheduler.set_timesteps(num_inference_steps)
        
        for t in self.scheduler.timesteps:
            timestep = torch.full((batchSize,), t, device=device, dtype=torch.long)
            
            
            with torch.no_grad():
                noisePred = self.model(sample, timestep, condition)
            
            output = self.scheduler.step(
                model_output=noisePred,
                timestep=t,
                sample=sample
            )
            sample = output[0]
        return sample


def chooseBestCheckpointValidation(cfg: DictConfig):
    import os
    import glob
    from numpy import argmin
    _, validationLoader, _ = fetchLoaders(cfg=cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpointDir = cfg.evaluation.checkpointDir
    checkpointPattern = os.path.join(checkpointDir, "checkpoint_*", "checkpoint_*.pt")
    checkpoints = sorted(glob.glob(checkpointPattern))
    
    numSteps = cfg.evaluation.validationPicker.steps
    lpipsScoreList = []
    
    for checkpoint in checkpoints:
        
        model = DDIMSampler(checkpoint, cfg=cfg)
        model.to(device)
        model.num_inference_steps = numSteps
        
        
        lpipsMetric = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True
        ).to(device)
        lpipsMetric.reset()
        
        with torch.no_grad():
            for batchIdx, batch in enumerate(validationLoader):
                slicePlusOne = batch["slicePlusOne"].to(device)
                sliceI = batch["sliceI"].to(device)
                sliceMinusOne = batch["sliceMinusOne"].to(device)
                
                generated = model.sample(slicePlusOne, sliceMinusOne, numSteps)
                
                generatedClamp = torch.clamp(generated,0,1)
                
                sliceIClamp = torch.clamp(sliceI,0,1)
                
                generatedRGB = generatedClamp.repeat(1,3,1,1)
                targetRGB = sliceIClamp.repeat(1,3,1,1)
                
                
                lpipsMetric.update(generatedRGB,targetRGB)
                
            lpipsScore = lpipsMetric.compute().item()
            print(F"Checkpoint{checkpoint} LPIPS VAL: {lpipsScore:.4f}")
            lpipsScoreList.append(lpipsScore)
        
    bestNum = argmin(lpipsScoreList)
    bestCheckpoint = checkpoints[bestNum]
    
    
    print(f"Best Checkpoint:",bestCheckpoint)



def TestingFunction(cfg: DictConfig):
    from pathlib import Path
    import os 
    import json
    import matplotlib.pyplot as plt
    import time
    
    _,_, testingLoader = fetchLoaders(cfg=cfg)
    
    
    bestCheckpoint = cfg.evaluation.bestCheckpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = DDIMSampler(checkpoint=bestCheckpoint,cfg=cfg)
    model.to(device)
    
    numSteps = cfg.evaluation.solver.steps 
    model.num_inference_steps = numSteps
    
    
    lpipsMetric = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to(device)
    SSIMMetric = StructuralSimilarityIndexMeasure(data_range=16).to(device)
    PSNRMetric = PeakSignalNoiseRatio(data_range=16).to(device)
    
    lpipsGlobal = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to(device)
    SSIMGlobal = StructuralSimilarityIndexMeasure(data_range=16).to(device)
    PSNRGlobal = PeakSignalNoiseRatio(data_range=16).to(device)  
    FIDMetric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    
    allScores = []
    inferenceTimes = []
    totalImages = 0
    
    with torch.no_grad():
        for i, batch in enumerate(testingLoader):
            if i >=3:
                break
            slicePlusOne = batch["slicePlusOne"].to(device)
            sliceI = batch["sliceI"].to(device)
            sliceMinusOne = batch["sliceMinusOne"].to(device)
            _ = model.sample(slicePlusOne, sliceMinusOne, numSteps)
            
    with torch.no_grad():
        for batchIdx, batch in enumerate(testingLoader):
            slicePlusOne = batch["slicePlusOne"].to(device)
            sliceI = batch["sliceI"].to(device)
            sliceMinusOne = batch["sliceMinusOne"].to(device)
            
            batchSize = sliceI.shape[0]
            totalImages += batchSize
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            startTime = time.time()
            
            generated = model.sample(slicePlusOne, sliceMinusOne, numSteps)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            endTime = time.time()
            batchTime = endTime - startTime
            inferenceTimes.append(batchTime)
            
            generatedClamp = torch.clamp(generated, 0, 1)
            sliceIClamp = torch.clamp(sliceI, 0, 1)
            
            
            rgbGenerated = generatedClamp.repeat(1, 3, 1, 1)
            targetRgb = sliceIClamp.repeat(1, 3, 1, 1)
            
            lpipsGlobal.update(rgbGenerated, targetRgb)
            SSIMGlobal.update(generated, sliceI)
            PSNRGlobal.update(generated, sliceI)
            FIDMetric.update(rgbGenerated, real=False)
            FIDMetric.update(targetRgb, real=True)
            
            for i in range(batchSize):
                generatedImg = generated[i].unsqueeze(0)
                targetImg = sliceI[i].unsqueeze(0)
                
                generatedImgClamp = torch.clamp(generatedImg, 0, 1)
                targetImgClamp = torch.clamp(targetImg, 0, 1)
                
                generatedImgRGB = generatedImgClamp.repeat(1, 3, 1, 1)
                targetImgRGB = targetImgClamp.repeat(1, 3, 1, 1)
                
                lpipsScore = lpipsMetric(generatedImgRGB, targetImgRGB).item()
                ssimScore = SSIMMetric(generatedImg, targetImg).item()
                psnrScore = PSNRMetric(generatedImg, targetImg).item()
                
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
    avgTimePerImage = totalTime / totalImages
    globalLPIPS = lpipsGlobal.compute().item()
    globalSSIM = SSIMGlobal.compute().item()
    globalPSNR = PSNRGlobal.compute().item()
    globalFID = FIDMetric.compute().item()
        
        
    allScores.sort(key=lambda x: x["lpips"])
    best30 = allScores[:30]
    worst30 = allScores[-30:]  
    
    results = {
        "globalMetrics": {
            "lpips": globalLPIPS,
            "ssim": globalSSIM,
            "psnr": globalPSNR,
            "fid": globalFID
        },
        "inferenceTime": {
            "totalImages": totalImages,
            "totalTimeSeconds": totalTime,
            "avgTimePerBatch": avgTimePerBatch,
            "avgTimePerImage": avgTimePerImage,
            "numDenoisingSteps": numSteps,
        },
        "best30": best30,
        "worst30": worst30,
    }
    
    imageSaveDir = cfg.evaluation.imageSave
    currentSolverStepDir = Path(imageSaveDir) / f"DDIM_{numSteps}"
    
    bestDir = currentSolverStepDir / "best"
    worstDir = currentSolverStepDir / "worst"
    
    bestDir.mkdir(parents=True, exist_ok=True)
    worstDir.mkdir(parents=True, exist_ok=True)
    
    jsonName = currentSolverStepDir / "results.json"
    with open(jsonName, "w") as f:
        json.dump(results, f, indent=2)
        
    #Best 30 
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
                
                generated = model.sample(
                    slicePlusOne.unsqueeze(0), 
                    sliceMinusOne.unsqueeze(0), 
                    numSteps
                )
                generated = generated[0, 0].cpu().numpy()
                target = sliceI[0].cpu().numpy()
                
                fig, axes = plt.subplots(1, 2, figsize=(12, 12))
                axes[0].imshow(target, cmap='gray')
                axes[0].set_title('Ground Truth', fontsize=14, fontweight='bold')
                axes[0].axis('off')
                axes[1].imshow(generated, cmap='gray')
                axes[1].set_title('Generated (DDPM)', fontsize=14, fontweight='bold')
                axes[1].axis('off')
                fig.suptitle(
                    f'Best Rank #{rank} (Index: {case["globalIDX"]})\n'
                    f'LPIPS: {case["lpips"]:.4f} | SSIM: {case["ssim"]:.4f} | '
                    f'PSNR: {case["psnr"]:.2f} dB',
                    fontsize=12, y=0.98
                )
                plt.tight_layout()
                filename = f"best_rank{rank:02d}_idx{case['globalIDX']:05d}.png"
                plt.savefig(bestDir / filename, dpi=150, bbox_inches='tight')
                plt.close()
                break
    #Worst 30 
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

                generated = model.sample(
                    slicePlusOne.unsqueeze(0), 
                    sliceMinusOne.unsqueeze(0), 
                    numSteps
                )
                generated = generated[0, 0].cpu().numpy()
                target = sliceI[0].cpu().numpy()
                
                fig, axes = plt.subplots(1, 2, figsize=(12, 12))
                axes[0].imshow(target, cmap='gray')
                axes[0].set_title('Ground Truth', fontsize=14, fontweight='bold')
                axes[0].axis('off')
                axes[1].imshow(generated, cmap='gray')
                axes[1].set_title('Generated (DDPM)', fontsize=14, fontweight='bold')
                axes[1].axis('off')
                fig.suptitle(
                    f'Worst Rank #{rank} (Index: {case["globalIDX"]})\n'
                    f'LPIPS: {case["lpips"]:.4f} | SSIM: {case["ssim"]:.4f} | '
                    f'PSNR: {case["psnr"]:.2f} dB',
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
    def main(cfg: DictConfig) -> None:
        results = TestingFunction(cfg=cfg)

    main()      
    

    
    
     
            
    
    
    
    
    
    
        

        
        