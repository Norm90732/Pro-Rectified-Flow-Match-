import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.amp import autocast
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from torch.optim.lr_scheduler import OneCycleLR
#training
import ray
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig, CheckpointConfig
import ray.train
from dataset.prostateData import fetchLoaders
#configs
from omegaconf import DictConfig,OmegaConf
import hydra
import wandb
#models
from models.DDPM.DDPM import ConditionalDDPM
#misc
import tempfile 
import os
import json
#EMA for smoother validation according to Fei 
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn



def trainingFunction(config:dict):
    cfg = config["cfg"]
    
    worldSize = ray.train.get_context().get_world_size()
    worldRank = ray.train.get_context().get_world_rank()
    
    trainingDataloader, validationDataloader, _ = fetchLoaders(cfg=cfg)
    
    if worldRank == 0:
        wandb.init(
            project = cfg.wandb.project,
            config=OmegaConf.to_container(cfg,resolve=True),
            name = cfg.experimentName
        )
    model = ConditionalDDPM(cfg=cfg)
    
    num_train_timesteps = model.scheduler.num_train_timesteps
    scheduler_ref = model.scheduler  
    
    
    parametersWithDecay = []
    parametersWithoutDecay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name.lower() or "positionalEmbedding" in name:
            parametersWithoutDecay.append(param)
        else:
            parametersWithDecay.append(param)
    parameterGroups = [
        {"params": parametersWithDecay, "weight_decay": cfg.model.optimizer.weight_decay},
        {"params": parametersWithoutDecay, "weight_decay": 0.0}
    ]
    optimizer = AdamW(
        parameterGroups,
        lr=cfg.model.optimizer.lr,
        betas=cfg.model.optimizer.betas,
        weight_decay=cfg.model.optimizer.weight_decay,
        eps=cfg.model.optimizer.eps
    )
    
    numEpochs = cfg.model.epochs
    stepsPerEpoch = len(trainingDataloader)
    totalSteps = numEpochs * stepsPerEpoch
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr = cfg.model.optimizer.lr,
        total_steps=totalSteps,
        pct_start=0.3,
        anneal_strategy="cos",
        div_factor=25,
        final_div_factor=10000.0,
    )
    device = torch.device(f"cuda:{ray.train.get_context().get_local_rank()}")
    model = model.to(device)
    emaDecay = cfg.model.ema
    emaAvgFn = get_ema_multi_avg_fn(decay=emaDecay)
    emaModel = AveragedModel(model,multi_avg_fn=emaAvgFn)
    
    
    model = ray.train.torch.prepare_model(
        model, 
        parallel_strategy_kwargs={"find_unused_parameters": True}
    )
    checkpoint = None 
    
    
    iterationLosses = 0.0
    numIterations = 0
    
    for epoch in range(numEpochs):
        model.train()
        runningLoss = 0.0
        for batchIdx, batch in enumerate(trainingDataloader):
            slicePlusOne = batch["slicePlusOne"].to(device)
            sliceI = batch["sliceI"].to(device)
            sliceMinusOne = batch["sliceMinusOne"].to(device)
            
            batchSize = sliceI.shape[0]
            
            timesteps = torch.randint(
                0, num_train_timesteps, (batchSize,), device=device
            ).long()
            
            
            noise = torch.randn_like(sliceI)
            noisySlice = scheduler_ref.add_noise(
                original_samples=sliceI,
                noise=noise,
                timesteps=timesteps
            )
            
            condition = torch.cat([slicePlusOne, sliceMinusOne], dim=1)
            noisePred = model(noisySlice,timesteps,condition)
            
            loss = F.mse_loss(noisePred,noise)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            emaModel.update_parameters(model)
            runningLoss += loss.item()
            
        trainLoss = runningLoss / len(trainingDataloader)
        model.eval()
        emaModel.eval()
        valLoss = 0.0
        
        
        with torch.no_grad():
            for batch in validationDataloader:
                slicePlusOne = batch["slicePlusOne"].to(device)
                sliceI = batch["sliceI"].to(device)
                sliceMinusOne = batch["sliceMinusOne"].to(device)
                batchSize = sliceI.shape[0]
            
                timesteps = torch.randint(
                0, num_train_timesteps, (batchSize,), device=device
                ).long()
                
                
                noise = torch.randn_like(sliceI)
                noisySlice = scheduler_ref.add_noise(
                original_samples=sliceI,
                noise=noise,
                timesteps=timesteps
                )
                
                condition = torch.cat([slicePlusOne, sliceMinusOne], dim=1)
                noisePred = model(noisySlice,timesteps,condition)
                
                loss = F.mse_loss(noisePred,noise)
                valLoss += loss.item()
            valLoss = valLoss/len(validationDataloader)
            
            metrics = {
            "trainLoss": trainLoss,
            "validationLoss": valLoss,
            "epoch": epoch,        
            }
            checkpoint = None
            if epoch % 10 == 0 or epoch == numEpochs - 1:
                if worldRank == 0: 
                    tempDir = tempfile.mkdtemp()
                    checkpoint_path = os.path.join(tempDir,f"checkpoint_{epoch}.pt")
                    statedict = model.state_dict()
                    consume_prefix_in_state_dict_if_present(statedict,"module.")
                    emaState = emaModel.module.state_dict()
                    torch.save({
                "model_state_dict": statedict,
                "ema_state_dict": emaState,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch
                }, checkpoint_path)
                checkpoint = ray.train.Checkpoint.from_directory(tempDir)
            
            if worldRank == 0:  
                ray.train.report(metrics,checkpoint=checkpoint)
                wandb.log(metrics,step=epoch)
    if worldRank == 0: 
        wandb.finish()    
                

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg:DictConfig):
    ray.init(ignore_reinit_error=True,
             num_cpus=cfg.numCpus,
             include_dashboard=False,)
    
    trainConfig = {
        "cfg": cfg,
    }
            
    trainer = TorchTrainer(
        train_loop_per_worker=trainingFunction,
        train_loop_config=trainConfig,
        scaling_config=ScalingConfig(
            num_workers=cfg.numWorkers,
            use_gpu=True,
            resources_per_worker={
                "CPU": cfg.resources_per_worker.CPU,
                "GPU": cfg.resources_per_worker.GPU
                }
         ),
        run_config=RunConfig(
            name=cfg.experimentName,
            storage_path=cfg.paths.checkpointDir,
            checkpoint_config=CheckpointConfig(
                num_to_keep=cfg.checkpoint.num_to_keep,
                checkpoint_score_attribute=cfg.checkpoint.checkpoint_score_attribute,
                checkpoint_score_order=cfg.checkpoint.checkpoint_score_order,
            ), 
        )
    )
    
    
    result = trainer.fit()
    
    
    ray.shutdown() 
if __name__ == "__main__":
    main()
    