import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from dataset.prostateData import fetchLoaders
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.fid import FrechetInceptionDistance
from pathlib import Path
import json
import time



def linearInterpolation(sliceMinusOne, slicePlusOne):
    return (sliceMinusOne + slicePlusOne) /2 

def bicubicInterpolation(sliceMinusOne, slicePlusOne):
    volume = torch.stack([sliceMinusOne,slicePlusOne],dim=-1)
    
    volume = volume.squeeze(1).permute(0,3,1,2).unsqueeze(1)
    
    VolumeUpsample = F.interpolate(
        volume,
        size=(3, volume.shape[3], volume.shape[4]),
        mode='trilinear',
        align_corners=True
    )
    middleSlice = VolumeUpsample[:, :, 1, :, :]
    
    return middleSlice

def evaluateBaseline(cfg:DictConfig,interpolationFunction,methodName:str):
    import matplotlib.pyplot as plt 
    _, _, testingLoader = fetchLoaders(cfg=cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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

            _ = interpolationFunction(sliceMinusOne, slicePlusOne)
    
    with torch.no_grad():
    #Obtain Best index by Lpips 
        for batchIdx, batch in enumerate(testingLoader):
            slicePlusOne = batch["slicePlusOne"].to(device)
            sliceI = batch["sliceI"].to(device)
            sliceMinusOne = batch["sliceMinusOne"].to(device)

            batchSize = sliceI.shape[0]
            totalImages += batchSize
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            startTime = time.time()
            
            generated = interpolationFunction(sliceMinusOne, slicePlusOne)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            endTime = time.time()
            batchTime = endTime-startTime
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
        "method": methodName,
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
        },
        "best30": best30,
        "worst30": worst30,
    }
    
    imageSaveDir = cfg.evaluation.imageSave #add this 
    saveDir = Path(imageSaveDir) / methodName
    
    bestDir = saveDir / "best"
    worstDir = saveDir / "worst"
    
    bestDir.mkdir(parents=True, exist_ok=True)
    worstDir.mkdir(parents=True, exist_ok=True)
    
    jsonPath = saveDir / "results.json"
    with open(jsonPath, "w") as f:
        json.dump(results, f, indent=2)
        
        
    with torch.no_grad():
        for rank, case in enumerate(best30, 1):
            batchIdx = case["batchIDX"]
            inBatchIDX = case["inBatchIDX"]
            
            for bIdx, batch in enumerate(testingLoader):
                if bIdx != batchIdx:
                    continue
                
                sliceI = batch["sliceI"][inBatchIDX].to(device)
                slicePlusOne = batch["slicePlusOne"][inBatchIDX].to(device)
                sliceMinusOne = batch["sliceMinusOne"][inBatchIDX].to(device)
                

                generated = interpolationFunction(
                    sliceMinusOne.unsqueeze(0),
                    slicePlusOne.unsqueeze(0)
                )
                generated = generated[0, 0].cpu().numpy()
                target = sliceI[0].cpu().numpy()
                
                
                fig, axes = plt.subplots(1, 2, figsize=(12, 12))
                axes[0].imshow(target, cmap='gray')
                axes[0].set_title('Ground Truth', fontsize=14, fontweight='bold')
                axes[0].axis('off')
                axes[1].imshow(generated, cmap='gray')
                axes[1].set_title(f'Generated ({methodName})', fontsize=14, fontweight='bold')
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
    with torch.no_grad():
        for rank, case in enumerate(worst30, 1):
            batchIdx = case["batchIDX"]
            inBatchIDX = case["inBatchIDX"]
            
            for bIdx, batch in enumerate(testingLoader):
                if bIdx != batchIdx:
                    continue
                
                sliceI = batch["sliceI"][inBatchIDX].to(device)
                slicePlusOne = batch["slicePlusOne"][inBatchIDX].to(device)
                sliceMinusOne = batch["sliceMinusOne"][inBatchIDX].to(device)
                
                generated = interpolationFunction(
                    sliceMinusOne.unsqueeze(0),
                    slicePlusOne.unsqueeze(0)
                )
                generated = generated[0, 0].cpu().numpy()
                target = sliceI[0].cpu().numpy()
                
                fig, axes = plt.subplots(1, 2, figsize=(12, 12))
                axes[0].imshow(target, cmap='gray')
                axes[0].set_title('Ground Truth', fontsize=14, fontweight='bold')
                axes[0].axis('off')
                axes[1].imshow(generated, cmap='gray')
                axes[1].set_title(f'Generated ({methodName})', fontsize=14, fontweight='bold')
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
        linearResults = evaluateBaseline(
        cfg=cfg,
        interpolationFunction=linearInterpolation,
        methodName="Linear"
    )
        bicubicResults = evaluateBaseline(
        cfg=cfg,
        interpolationFunction=bicubicInterpolation,
        methodName="Bicubic"
    )

    main()      
    