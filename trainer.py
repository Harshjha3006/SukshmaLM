from trainer_config import LLMTrainerConfig, get_trainer_config
import torch 
import torch.nn as nn
from model.gpt import GPT
from dataloader import get_dataloader
import random 
import numpy as np 
import os
from torch.utils.tensorboard.writer import SummaryWriter
import json 
from tqdm import tqdm
import math 


# define paths for storing model checkpoints and tensorboard logs 
checkpoints_path = "checkpoints"
logs_path = "tensorboard_logs"

class LLMTrainer: 

    def __init__(self, config: LLMTrainerConfig): 
        """
        Trainer class for training an LLM 

        Args: 
            config (LLMTrainerConfig): Contains all the hyperparameters and other 
                            configuration options required for training an LLM
        """

        # Creating required directories 
        os.makedirs(checkpoints_path, exist_ok=True)
        os.makedirs(logs_path, exist_ok=True)

        self.num_epochs = config.num_epochs         # Number of training epochs 
        self.lr = config.lr                         # Learning rate 
        self.l2reg = config.l2reg                   # L2 regularization rate 
        self.exp_name = config.exp_name             # Experiment Name 
        self.device = config.device                 # device -> cpu or gpu 
        self.dataloader = get_dataloader(config)    # training dataloader 
        self.warmup_steps = config.warmup_steps     # warmup steps for the cosine lr scheduler 
        self.max_steps = len(self.dataloader) * self.num_epochs  # total steps of the training process 

        self.model = GPT(config).to(self.device)    # LLM model transferred to configured device 
        self.model.apply(self.init_weights)         # Apply initialization to all layers 

        # Defining an AdamW optimizer and specifying learning rate and L2 regularization rate 
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = self.lr, weight_decay= self.l2reg)
        # Defining a TensorBoard logger for logging loss values at the end of each epoch 
        self.logger = SummaryWriter(log_dir=f"{logs_path}/{self.exp_name}")

        self.config = config

        # Setting the seed for better reproducibility 
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


        # Defining the loss function 
        self.criterion = nn.CrossEntropyLoss()

        # Create directory for storing model checkpoints and configs 
        os.makedirs(f"{checkpoints_path}/{self.exp_name}", exist_ok=True)

    
    def get_lr(self, step): 
        """
        Returns the value of the learning rate for the provided step 
        Args: 
            step (int): the current step in the training process
        """

        if step < self.warmup_steps: 
            return self.lr * (step / self.warmup_steps)
        
        else: 
            progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            return 0.5 * self.lr * (1 + math.cos(math.pi * progress))



    def init_weights(self, module):

        # intialize layer weights with a normal distribution of mean 0 and std 0.02
        if isinstance(module, (nn.Linear, nn.Embedding)): 
            nn.init.normal_(module.weight,mean = 0.0, std = 0.02)
        # initialize layer biases with 0 
        if isinstance(module, nn.Linear) and module.bias is not None: 
            nn.init.zeros_(module.bias)
    

    def train(self): 

        """
        Trains the LLM with the given hyperparameters 
        """

        best_epoch = 0   # The epoch with lowest loss
        best_loss = 1e9
        best_config = None # used for later saving model's best state to disk 

        # set the model to training mode 
        self.model.train()
        
        # count of iterations
        steps = 0

        # iterate till self.num_epochs
        for epoch in range(self.num_epochs): 

            # contains list of losses of each batch 
            loss_history = []

            # iterate over all batches of training data 
            for x, y in tqdm(self.dataloader): 

                # move the inputs and targets to the appropriate device 
                x = x.to(self.device) # (Batch, Context_len)
                y = y.to(self.device) # (Batch, Context_len)

                # zero out the accumulated gradients from the previous batch 
                self.optimizer.zero_grad()

                # compute logits from the model 
                logits = self.model(x) # (Batch, Context_len, vocab_size)

                # compute cross entropy loss 
                loss = self.criterion(logits.view(-1, logits.shape[-1]), y.view(-1))

                # compute gradients
                loss.backward()

                # gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # get the current lr as per the cosine scheduler
                lr = self.get_lr(steps)

                # set this lr in the optimizer
                for param_group in self.optimizer.param_groups: 
                    param_group["lr"] = lr
                
                # update model weights 
                self.optimizer.step()

                # append this batch's loss to the loss history 
                loss_history.append(loss.item())

                # update steps
                steps += 1


            # compute avg loss for whole epoch 
            avg_loss = np.mean(loss_history)

            # update best_loss, best_epoch and best_config if current epoch's loss is the lowest till now
            if avg_loss < best_loss: 
                best_loss = avg_loss
                best_epoch = epoch + 1
                best_config = {
                    "model_state_dict" : self.model.state_dict(), 
                    "optimizer" : self.optimizer.state_dict(),
                    "epoch" : epoch + 1, 
                    "config": self.config
                }

            # print and log current epoch' loss 
            print(f"Epoch {epoch + 1}: Loss: {avg_loss}")
            self.logger.add_scalar("Loss", avg_loss, epoch + 1)


        # close the logger 
        self.logger.close()

        # save the model's best state to disk 
        torch.save(best_config, f"{checkpoints_path}/{self.exp_name}/model.pth")

        # save model config in json for readability 
        model_config = {
            "context_len": config.context_len, 
            "embed_dim": config.embed_dim, 
            "vocab_size": config.vocab_size, 
            "num_layers": config.num_layers, 
            "num_heads": config.num_heads, 
        }

        with open(f"{checkpoints_path}/{self.exp_name}/config.json", 'w') as f: 
            json.dump(model_config, f)
        

        print(f"Best Epoch: {best_epoch}")
        print(f"Best Loss: {best_loss}")




if __name__ == "__main__": 

    config = get_trainer_config()
    trainer = LLMTrainer(config)
    trainer.train()
                