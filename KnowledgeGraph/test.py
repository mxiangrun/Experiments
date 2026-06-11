import torch


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
        answers_h = predict_triplet[:, 0]
        predict_h = predict_triplet[:, None, :]
        predict_h = predict_h.repeat(1, self.num_e, 1)
        predict_h[:, :, 0] = self.e_idx

        # predict r
        answers_r = predict_triplet[:, 0]
        predict_r = predict_triplet[:, None, :]
        predict_r = predict_r.repeat(1, self.num_r, 1)
        predict_r[:, :, 1] = self.r_idx

        # predict t
        answers_t = predict_triplet[:, 2]
        predict_t = predict_triplet[:, None, :]
        predict_t = predict_t.repeat(1, self.num_e, 1)
        predict_t[:, :, 2] = self.e_idx

        mask = torch.ones(num_triplet, dtype=torch.bool, device=self.test_triplet.device)
        mask[predict_triplet_idx] = False
        self.test_triplet = self.test_triplet[mask]

        message_triplet = self.train_triplet.clone()

        return (message_triplet,
                answers_h, predict_h,
                answers_r, predict_r,
                answers_t, predict_t)


data = NationsDataset(batch_size=8)
num_epoch = 3
train_step = 10
valid_step = 10

# for epoch in range(num_epoch):
#     for step in range(train_step):
#         message_triplet, predict_pos_triplet, predict_neg_triplet = data.get_train_batch()
#         print(f'epoch {epoch} train step {step}')
#
#     for step in range(valid_step):
#         message_triplet, predict_pos_triplet, predict_neg_triplet = data.get_valid_batch()
#         print(f'epoch {epoch} valid step {step}')

step=0
while data.test_triplet.numel():
    message_triplet, predict_pos_triplet, predict_neg_triplet = data.get_test_batch()
    print(f'test step {step}')
    step+=1

print()
