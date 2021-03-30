
"""
Model definitions
"""

import os
import wandb
import torch
import metrics
import networks
import data_utils
import train_utils
import numpy as np
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt


class StockMovementClassifier:

    def __init__(self, args):
        self.config, self.output_dir, self.logger, self.device = common.init_experiment(args)

        # Networks
        self.price_encoder = networks.PriceEncoder(config['price_encoder']).to(self.device)
        self.news_encoder = networks.NewsEncoder(config['news_encoder']).to(self.device)
        self.graph_attention = networks.MultiHeadGraphAttention(config['graph_attention']).to(self.device)
        self.clf_head = networks.Classifier(config['classifier']).to(self.device)

        # Optimizer, loss function, scheduler
        self.criterion = nn.NLLLoss()
        self.optim = train_utils.get_optimizer(
            config = config['optimizer'],
            params = list(self.price_encoder.parameters()) + list(self.news_encoder.parameters()) +
                        list(self.graph_attention.parameters()) + list(self.clf_head.parameters()))
        self.scheduler, self.warmup_epochs = train_utils.get_scheduler(
            config = config['scheduler'], optimizer = self.optim)

        # Dataloaders
        self.train_loader, self.val_loader = data_utils.get_dataloaders(
            root=config['data'].get['root'], batch_size=config['data']['batch_size'])

        # Logging and wandb
        self.best_val_score = 0
        self.done_epochs = 0
        run = wandb.init('stock-prediction-ai')
        self.logger.write("Wandb: " + run.get_url(), mode='info')

        # Load model if specified
        if args["load"] is not None:
            if not os.path.exists(os.path.join(args['load'], "best_model.ckpt")):
                raise NotImplementedError("The specified load path does not contain best_model.ckpt")
            else:
                self.load_model(args["load"])

    def trainable(self, val):
        if val:
            self.price_encoder.train()
            self.news_encoder.train()
            self.graph_attention.train()
            self.clf_head.train()
        else:
            self.price_encoder.eval()
            self.news_encoder.eval()
            self.graph_attention.eval()
            self.clf_head.eval()

    def save_state(self, epoch):
        state = {
            "epoch": epoch,
            "price_encoder": self.price_encoder.state_dict(),
            "news_encoder": self.news_encoder.state_dict(),
            "graph_attention": self.graph_attention.state_dict(),
            "clf_head": self.clf_head.state_dict(),
            "optim": self.optim.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None
        }
        torch.save(state, os.path.join(self.output_dir, "last_state.ckpt"))

    def save_model(self):
        state = {
            "price_encoder": self.price_encoder.state_dict(),
            "news_encoder": self.news_encoder.state_dict(),
            "graph_attention": self.graph_attention.state_dict(),
            "clf_head": self.clf_head.state_dict()
        }
        torch.save(state, os.path.join(self.output_dir, "best_model.ckpt"))

    def load_state(self):
        if not torch.cuda.is_available():
            self.logger.record("Attempting to deserialize CUDA model on CPU. If you're loading \
                a model for inference, using trainer.load_model() instead", mode='warn')

        state = torch.load(os.path.join(self.output_dir, "last_state.ckpt"))
        self.done_epochs = state["epoch"]
        self.price_encoder.load_state_dict(state["price_encoder"])
        self.news_encoder.load_state_dict(state["news_encoder"])
        self.graph_attention.load_state_dict(state["graph_attention"])
        self.clf_head.load_state_dict(state["clf_head"])
        self.optim.load_state_dict(state["optim"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state["scheduler"])

    def load_model(self, output_dir):
        path = os.path.join(output_dir, "best_model.ckpt")
        if not torch.cuda.is_available():
            state = torch.load(path, map_location=torch.device("cpu"))
        else:
            state = torch.load(path)
        self.price_encoder.load_state_dict(state["price_encoder"])
        self.news_encoder.load_state_dict(state["news_encoder"])
        self.graph_attention.load_state_dict(state["graph_attention"])
        self.clf_head.load_state_dict(state["clf_head"])
        self.logger.record(f"Successfully loaded model from {output_dir}!", mode='info')

    def train_one_step(self, batch):
        self.trainable(True)
        prices, news, trg = batch
        price_fs = self.price_encoder(prices)
        news_fs = self.news_encoder(news)
        out = self.clf_head(self.graph_attention(price_fs, news_fs))
        loss = self.criterion(out, trg)
        acc = metrics.accuracy(out, trg)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return {"Loss": loss.item(), "Accuracy": acc}

    def validate_one_step(self, batch):
        self.trainable(False)
        prices, news, trg = batch
        with torch.no_grad():
            price_fs = self.price_encoder(prices)
            news_fs = self.news_encoder(news)
            out = self.clf_head(self.graph_attention(price_fs, news_fs))

        loss = self.criterion(out, trg)
        acc = metrics.accuracy(out, trg)
        return {"Loss": loss.item(), "Accuracy": acc}

    def get_stock_attention_probs(self):
        batch = self.val_loader.get_last_batch()
        prices, news, _ = batch
        price_fs = self.price_encoder()
        news_fs = self.news_encoder()
        _, attn_scores = self.graph_attention(price_fs, news_fs, return_attn=True)

        if not os.path.exists(os.path.join(self.output_dir, "plots")):
            os.mkdir(os.path.join(self.output_dir, "plots"))

        # attn_scores is a list of (S, S) numpy arrays (softmax along dim=1), one for each head
        for i in range(len(attn_scores)):
            plt.figure(figsize=(12, 10))
            sns.heatmap(attn_scores[i], square=True, annot=False, cmap='RdBu_r',
                        xticklabels=self.train_loader.tickers, yticklabels=self.train_loader.tickers)
            plt.title(f"Head {i}", fontsize=13, fontweight="bold")
            plt.savefig(os.path.join(self.output_dir, "plots", f"head_{i}.png"), pad_inches=0.05)

    def train(self):
        for epoch in range(self.config["epochs"]-self.done_epochs+1):
            self.logger.record(f"Epoch {epoch+1}/{self.config['epochs']}", mode='train')
            train_meter = common.AverageMeter()

            for step in range(len(self.train_loader)):
                batch = self.train_loader.flow()
                train_metrics = self.train_one_step(batch)
                train_meter.add(train_metrics)
                wandb.log({"Train loss": train_metrics["Loss"]})
                common.progress_bar(
                    progress = (step+1)/len(self.train_loader),
                    status = train_meter.return_msg())

            common.progress_bar(progress=1.0, status=train_meter.return_msg())
            wandb.log({"Train accuracy": train_meter.return_metrics()["Accuracy"], "Epoch": epoch+1})
            self.logger.write(train_meter.return_msg(), mode='train')
            self.save_state(epoch+1)

            if (epoch+1) % self.config["eval_every"] == 0:
                val_meter = common.AverageMeter()
                for step in range(len(self.val_loader)):
                    batch = self.val_loader.flow()
                    val_metrics = self.validate_one_step(batch)
                    val_meter.add(val_metrics)
                    common.progress_bar(
                        progress = (step+1)/len(self.val_loader),
                        status = val_meter.return_msg())

                common.progress_bar(progress=1.0, status=val_meter.return_msg())
                self.logger.write(val_meter.return_msg(), mode='val')
                wandb.log({
                    "Validation loss": val_meter.return_metrics()["Loss"],
                    "Validation accuracy": val_meter.return_metrics()["Accuracy"]})

                # Save model if better than current best
                val_results = val_meter.return_metrics()
                if val_results["Accuracy"] > self.best_val_score:
                    self.best_val_score = val_results["Accuracy"]
                    self.save_model()

        # Training complete
        self.logger.record("Training complete!", mode='info')
        self.get_stock_attention_probs()
