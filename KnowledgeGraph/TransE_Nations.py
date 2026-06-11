import torch
import gc
import os
from types import SimpleNamespace
import pickle
import random
import numpy as np
import pandas as pd


class TransE(torch.nn.Module):
    def __init__(self,
                 num_e: int,
                 num_r: int,
                 emb_dim: int = 50, ):
        super().__init__()
        self.e_embed = torch.nn.Embedding(num_e, emb_dim)
        self.r_embed = torch.nn.Embedding(num_r, emb_dim)
        self.MarginRankingLoss = torch.nn.MarginRankingLoss(margin=1.0)

    def set_trainable_parameters(self):
        for name, param in self.named_parameters():
            param.requires_grad = True
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                if module.weight.requires_grad:
                    torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None and module.bias.requires_grad:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, torch.nn.Embedding):
                if module.weight.requires_grad:
                    torch.nn.init.xavier_uniform_(module.weight)

    def get_trainable_parameters(self):
        trainable_params = list(filter(lambda p: p.requires_grad, self.parameters()))
        trainable_params_count = sum(p.numel() for p in trainable_params)
        print(f"Trainable parameters: {trainable_params_count / 1e6:.2f} M")
        return trainable_params

    def score(self, triplet):
        triplet = self.to_device(triplet)
        h_idx, r_idx, t_idx = triplet[:, 0], triplet[:, 1], triplet[:, 2]
        er_emb = torch.cat(tensors=[self.e_embed.weight, self.r_embed.weight], dim=0)
        h_emb = er_emb[h_idx]
        r_emb = er_emb[r_idx]
        t_emb = er_emb[t_idx]

        h_emb = torch.nn.functional.normalize(h_emb, p=2, dim=-1)
        t_emb = torch.nn.functional.normalize(t_emb, p=2, dim=-1)

        score = h_emb + r_emb - t_emb
        score = -(torch.abs(score) ** 1).sum(dim=-1)
        return score

    def forward(self, pos_triplet, neg_triplet):
        pos_score = self.score(pos_triplet)
        neg_score = self.score(neg_triplet)
        loss = self.loss(pos_score, neg_score)
        return pos_score, neg_score, loss

    def loss(self, pos_score, neg_score):
        target = torch.ones_like(pos_score)
        return self.MarginRankingLoss(input1=pos_score, input2=neg_score, target=target)

    def to_device(self, x):
        x = x.to(device=next(self.parameters()).device)
        return x


class NationsDataset:
    def __init__(self, batch_size):
        from pykeen.datasets import get_dataset

        dataset = get_dataset(dataset="nations")
        self.train_triplet = dataset.training.mapped_triples
        self.valid_triplet = dataset.validation.mapped_triples
        self.test_triplet = dataset.testing.mapped_triples
        self.copy_triplet = None

        self.e_name2idx = dataset.entity_to_id
        self.r_name2idx = dataset.relation_to_id
        self.e_idx2name = dataset.training.entity_id_to_label
        self.r_idx2name = dataset.training.relation_id_to_label

        self.num_e = dataset.num_entities
        self.num_r = dataset.num_relations

        self.e_idx = torch.tensor(list(self.e_idx2name.keys()), dtype=torch.long)
        self.r_idx = torch.tensor(list(self.r_idx2name.keys()), dtype=torch.long)

        self.r_idx += self.num_e
        self.train_triplet[:, 1] += self.num_e
        self.valid_triplet[:, 1] += self.num_e
        self.test_triplet[:, 1] += self.num_e

        self.batch_size = batch_size

    def neg_sampling(self, all_pos: torch.Tensor, batch_pos: torch.Tensor):
        num_pos = batch_pos.shape[0]

        # num_e = self.e_idx.shape[0]

        def random_choice_e(num_samples):
            random_e = torch.ones_like(self.e_idx, dtype=torch.float, device=self.e_idx.device)
            random_e = random_e.multinomial(num_samples=num_samples, replacement=True)
            random_e = self.e_idx[random_e]
            return random_e

        random_e = random_choice_e(num_samples=num_pos)

        half = num_pos // 2
        batch_neg = batch_pos.clone()
        batch_neg[:half, 0] = random_e[:half]  # head
        batch_neg[half:, 2] = random_e[half:]  # tail

        def is_in_pos(neg, all_pos):
            match_h = neg[:, None, 0] == all_pos[None, :, 0]
            match_r = neg[:, None, 1] == all_pos[None, :, 1]
            match_t = neg[:, None, 2] == all_pos[None, :, 2]
            return (match_h & match_r & match_t).any(dim=1)  # [bs]

        invalid_mask = is_in_pos(neg=batch_neg, all_pos=all_pos)

        while invalid_mask.any():
            invalid_idx = invalid_mask.nonzero().squeeze(1)
            k = len(invalid_idx)

            new_e = random_choice_e(num_samples=k)
            new_h = torch.where(invalid_idx < half, new_e, batch_neg[invalid_idx, 0])
            new_t = torch.where(invalid_idx >= half, new_e, batch_neg[invalid_idx, 2])

            batch_neg[invalid_idx, 0] = new_h
            batch_neg[invalid_idx, 2] = new_t

            invalid_mask = is_in_pos(neg=batch_neg, all_pos=all_pos)

        return batch_neg

    def get_train_batch(self):
        num_triplet = self.train_triplet.shape[0]
        num_pos = self.batch_size // 2
        num_pos = num_pos if num_triplet > self.batch_size else num_triplet

        predict_pos_triplet_idx = torch.multinomial(torch.ones(num_triplet - 0), num_samples=num_pos, replacement=False)
        predict_pos_triplet = self.train_triplet[predict_pos_triplet_idx]
        predict_neg_triplet = self.neg_sampling(all_pos=self.train_triplet, batch_pos=predict_pos_triplet)

        mask = torch.ones(num_triplet, dtype=torch.bool, device=self.train_triplet.device)
        mask[predict_pos_triplet_idx] = False

        message_triplet = self.train_triplet[mask]

        return message_triplet, predict_pos_triplet, predict_neg_triplet

    def get_valid_batch(self):
        num_triplet = self.valid_triplet.shape[0]
        num_pos = self.batch_size // 2
        num_pos = num_pos if num_triplet > self.batch_size else num_triplet

        predict_pos_triplet_idx = torch.multinomial(torch.ones(num_triplet - 0), num_samples=num_pos, replacement=False)
        predict_pos_triplet = self.valid_triplet[predict_pos_triplet_idx]
        predict_neg_triplet = self.neg_sampling(all_pos=self.train_triplet, batch_pos=predict_pos_triplet)

        message_triplet = self.train_triplet.clone()

        return message_triplet, predict_pos_triplet, predict_neg_triplet

    def get_test_batch(self):
        num_triplet = self.test_triplet.shape[0]
        num_pos = self.batch_size // 2
        num_pos = num_pos if num_triplet > self.batch_size else num_triplet

        predict_triplet_idx = torch.multinomial(torch.ones(num_triplet - 0), num_samples=num_pos, replacement=False)
        predict_triplet = self.test_triplet[predict_triplet_idx]

        # predict h
        labels_h = predict_triplet[:, 0]
        predict_h = predict_triplet[:, None, :]
        predict_h = predict_h.repeat(1, self.num_e, 1)
        predict_h[:, :, 0] = self.e_idx

        # predict r
        labels_r = predict_triplet[:, 0]
        predict_r = predict_triplet[:, None, :]
        predict_r = predict_r.repeat(1, self.num_r, 1)
        predict_r[:, :, 1] = self.r_idx

        # predict t
        labels_t = predict_triplet[:, 2]
        predict_t = predict_triplet[:, None, :]
        predict_t = predict_t.repeat(1, self.num_e, 1)
        predict_t[:, :, 2] = self.e_idx

        mask = torch.ones(num_triplet, dtype=torch.bool, device=self.test_triplet.device)
        mask[predict_triplet_idx] = False
        self.test_triplet = self.test_triplet[mask]

        message_triplet = self.train_triplet.clone()

        return message_triplet, labels_h, predict_h, labels_r, predict_r, labels_t, predict_t


def load_TransE(dataset, config, weight_path, config_path):
    if os.path.exists(path=config_path):
        with open(file=config_path, mode='rb') as f:
            config = pickle.load(file=f)
    weight = torch.load(f=weight_path, map_location="cpu", weights_only=True) if os.path.exists(weight_path) else None
    model = TransE(num_e=dataset.num_e,
                   num_r=dataset.num_r,
                   emb_dim=config.emb_dim)
    if weight is not None:
        result = model.load_state_dict(state_dict=weight, strict=False)
    return model


def save_TransE(model, config, weight_path, config_path):
    with open(file=config_path, mode="wb") as f:
        pickle.dump(obj=config, file=f)
    torch.save(obj=model.state_dict(), f=weight_path)


def release_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_logger(logger_path="logger.txt"):
    import logging

    logger = logging.getLogger("MyLogger")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(logger_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


set_seed()


class Pipeline:
    def __init__(self, config):
        self.config = config
        self.dataset = NationsDataset(batch_size=config.batch_size)
        self.model = TransE(num_e=self.dataset.num_e,
                            num_r=self.dataset.num_r,
                            emb_dim=self.config.emb_dim)
        self.model = self.model.cuda()
        self.optimizer = torch.optim.AdamW(params=self.model.get_trainable_parameters(),
                                           lr=self.config.learning_rate,
                                           weight_decay=self.config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=self.optimizer,
                                                                              T_0=self.config.num_train_step,
                                                                              T_mult=2,
                                                                              eta_min=self.config.min_learning_rate)
        self.logger = set_logger(logger_path=self.config.logger_path)

    def train(self):
        min_valid_loss = float('inf')
        num_train_step = self.config.num_train_step
        num_valid_step = self.config.num_valid_step
        grad_accm_step = self.config.grad_accm_step
        stop_training = False
        stop_patience = self.config.stop_patience
        left_patience = stop_patience
        train_loss, valid_loss = [], []
        # start
        self.optimizer.zero_grad()
        for epoch in range(self.config.num_epoch):
            self.model.train()
            for step in range(num_train_step):
                message_triplet, predict_pos_triplet, predict_neg_triplet = self.dataset.get_train_batch()
                pos_score, neg_score, loss = self.model(pos_triplet=predict_pos_triplet,
                                                        neg_triplet=predict_neg_triplet)
                loss.backward()
                train_loss.append(loss.item())
                if (step + 1) % grad_accm_step == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                self.logger.info(f'epoch {epoch} train step {step} loss {loss:.4f} lr {self.scheduler.get_lr()[0]:.6f}')

            self.model.eval()
            with torch.no_grad():
                for step in range(num_valid_step):
                    message_triplet, predict_pos_triplet, predict_neg_triplet = self.dataset.get_valid_batch()
                    pos_score, neg_score, loss = self.model(pos_triplet=predict_pos_triplet,
                                                            neg_triplet=predict_neg_triplet)
                    valid_loss.append(loss.item())
                    self.logger.info(f'valid step: {step}, loss: {loss:.4f}')

                # stopping
                train_loss = sum(train_loss) / len(train_loss)
                valid_loss = sum(valid_loss) / len(valid_loss)
                if valid_loss <= min_valid_loss:
                    min_valid_loss = valid_loss
                    save_TransE(model=self.model,
                                config=self.config,
                                weight_path=self.config.weight_path,
                                config_path=self.config.config_path)
                    left_patience = stop_patience
                else:
                    left_patience -= 1
                stop_training = True if left_patience <= 0 else False
                # logger
                self.logger.info(f'Train loss: {train_loss:.4f}')
                self.logger.info(f'Valid loss: {valid_loss:.4f}')
                self.logger.info(f'Min Valid Loss: {min_valid_loss:.4f}')
                self.logger.info(f'Stop patience: {left_patience}')
                # init list
                train_loss, valid_loss = [], []
                release_gpu()
            if stop_training:
                break

    @torch.no_grad()
    def test(self):
        self.model = load_TransE(dataset=self.dataset,
                                 config=self.config,
                                 weight_path=self.config.weight_path,
                                 config_path=self.config.config_path)
        self.model.eval()
        with torch.no_grad():
            all_logits_h = []
            all_logits_r = []
            all_logits_t = []
            all_labels_h = []
            all_labels_r = []
            all_labels_t = []
            while self.dataset.test_triplet.numel():
                message_triplet, labels_h, predict_h, labels_r, predict_r, labels_t, predict_t = self.dataset.get_test_batch()

                batch_size, num_e, three = predict_h.shape
                predict_h = predict_h.reshape(batch_size * num_e, three)
                logits_h = self.model.score(triplet=predict_h)
                logits_h = logits_h.reshape(batch_size, num_e)

                batch_size, num_r, three = predict_r.shape
                predict_r = predict_r.reshape(batch_size * num_r, three)
                logits_r = self.model.score(triplet=predict_r)
                logits_r = logits_r.reshape(batch_size, num_r)

                batch_size, num_e, three = predict_t.shape
                predict_t = predict_t.reshape(batch_size * num_e, three)
                logits_t = self.model.score(triplet=predict_t)
                logits_t = logits_t.reshape(batch_size, num_e)

                all_logits_h.append(logits_h)
                all_labels_h.append(labels_h)
                all_logits_r.append(logits_r)
                all_labels_r.append(labels_r)
                all_logits_t.append(logits_t)
                all_labels_t.append(labels_t)

            all_logits_h = torch.cat(all_logits_h, dim=0)
            all_labels_h = torch.cat(all_labels_h, dim=0)
            all_logits_r = torch.cat(all_logits_r, dim=0)
            all_labels_r = torch.cat(all_labels_r, dim=0)
            all_logits_t = torch.cat(all_logits_t, dim=0)
            all_labels_t = torch.cat(all_labels_t, dim=0)

            def metrics(hits):
                num_sample = hits.shape[0]
                num_top = hits.shape[1]

                def get_rank(row):
                    if row.any():
                        return 1 / (row.index.get_loc(row.idxmax()) + 1)
                    else:
                        return 0

                mrr = hits.apply(func=get_rank, axis=1)
                mrr = mrr.sum()
                mrr = mrr / num_sample
                performance = {'mrr': mrr}
                self.logger.info(f'MRR  : {mrr:.4f}')

                for topk in range(num_top):
                    hits_topk = hits.iloc[:, :topk].sum().sum()
                    hits_topk = hits_topk / num_sample
                    performance[str(topk)] = hits_topk
                    if topk < 10:
                        self.logger.info(f'Hit@{topk}: {hits_topk:.4f}')

                performance = pd.DataFrame(list(performance.items()), columns=['metrics', 'performance'])
                return performance

            def hit_table(logits, labels):
                scores, result = torch.topk(input=logits, k=logits.shape[1], dim=1)
                answer = labels.detach().cpu().numpy()
                result = pd.DataFrame(data=result.detach().cpu().numpy())
                result['answer'] = answer
                result = result.apply(func=lambda row: row.isin([row['answer']]), axis=1)
                result = result.drop(columns=['answer'])
                return result

            self.logger.info('Predict Head')
            hits_h = hit_table(logits=all_logits_h, labels=all_labels_h)
            performance_h = metrics(hits=hits_h)
            self.logger.info('Predict Relation')
            hits_r = hit_table(logits=all_logits_r, labels=all_labels_r)
            performance_r = metrics(hits=hits_r)
            self.logger.info('Predict Tail')
            hits_t = hit_table(logits=all_logits_t, labels=all_labels_t)
            performance_t = metrics(hits=hits_t)
            self.logger.info('Predict All')
            hits = pd.concat(objs=[hits_h, hits_r, hits_t], axis=0)
            hits = hits.fillna(False)
            performance = metrics(hits=hits)

            result_path = self.config.result_path
            with pd.ExcelWriter(result_path,
                                engine="openpyxl",
                                mode="a" if os.path.exists(result_path) else "w",
                                if_sheet_exists="replace" if os.path.exists(result_path) else None) as writer:
                performance.to_excel(writer, sheet_name='performance', index=False)
                performance_h.to_excel(writer, sheet_name='performance_h', index=False)
                performance_r.to_excel(writer, sheet_name='performance_r', index=False)
                performance_t.to_excel(writer, sheet_name='performance_t', index=False)
                hits.to_excel(writer, sheet_name='hits', index=False)
                hits_h.to_excel(writer, sheet_name='hits_h', index=False)
                hits_r.to_excel(writer, sheet_name='hits_r', index=False)
                hits_t.to_excel(writer, sheet_name='hits_t', index=False)


config = SimpleNamespace(emb_dim=50,
                         batch_size=1024,

                         learning_rate=0.0001,
                         min_learning_rate=0.0001 * 0.1,
                         weight_decay=0.001,

                         num_epoch=10000,
                         num_train_step=20,
                         num_valid_step=20,
                         grad_accm_step=1,
                         stop_patience=10,

                         weight_path="./TransE_Nations_weight.pth",
                         config_path="./TransE_Nations_config.pkl",
                         logger_path="./TransE_Nations_logger.txt",
                         result_path="./TransE_Nations_result.xlsx", )

p = Pipeline(config=config)
p.train()
p.test()
