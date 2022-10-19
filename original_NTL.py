import os
from model import MMD_loss

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import numpy as np
import math
import argparse
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

import logging

from nli_dataset import load_data, split_dataset
from torch.utils.data import DataLoader
from utils import save_model, TqdmLoggingHandler, setup_seed, CustomDataset
from model import classifier


device = torch.device("cuda")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(TqdmLoggingHandler())


def train(args):
    setup_seed(args.seed)
    log.info(f'Set seed {args.seed}')

    def show_loss(total_loss_per_period, total_loss_ce_per_period, total_mmd_loss_per_period, total_target_ce_loss_per_period):
        log.info(f'--'*10 + 'Loss:' + '--'*10)
        log.info(f'Total loss during the period is: {np.mean(total_loss_per_period)}')
        log.info(f'Total Cross Entropy loss during the period is: {np.mean(total_loss_ce_per_period)}')
        log.info(f'Total MMD loss during the period is: {np.mean(total_mmd_loss_per_period)}')
        log.info(f'Total Target CE loss during the period is: {np.mean(total_target_ce_loss_per_period)}')

    exp_dir = os.path.join(args.output_dir, f'{args.expriment_name}_{args.source_id}_{args.target_id}_seed_{args.seed}')

    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    # note that we need to specify the number of classes for this task
    # we can directly use the metadata (num_classes) stored in the dataset
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=args.num_labels)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_dataset = load_data(tokenizer)
    train_dataset1 , train_dataset2 = train_dataset[args.source_id], train_dataset[args.target_id]

    train_dataset1, val_dataset1 = split_dataset(train_dataset1)
    train_dataset2, val_dataset2 = split_dataset(train_dataset2)

    train_dataset = CustomDataset(train_dataset1, train_dataset2)
    val_dataset = CustomDataset(val_dataset1, val_dataset2)

    train_datasetloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle = True)
    val_datasetloader = DataLoader(val_dataset, batch_size=args.batch_size)

    len_dataloader = len(train_datasetloader)
    num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
    max_steps = math.ceil(args.epochs * num_update_steps_per_epoch)


    optimizer = torch.optim.Adam([
        {'params': model.base_model.parameters()},
        {'params': model.classifier.parameters(), 'lr': 1e-3}]
        , lr=0.00005, betas=(0.9,0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    scheduler = get_linear_schedule_with_warmup(optimizer, 0, max_steps)
    loss_CE = nn.CrossEntropyLoss()

    model.cuda()

    def evaluate(model, dataloader):
        source_logits = []
        source_labels_list = []
        target_logits = []
        target_labels_list = []
        with torch.no_grad():
            for examples in tqdm(dataloader):
                input_ids_1, attention_mask_1, label_1, input_ids_2, attention_mask_2, label_2 = \
                    examples['input_ids_1'].cuda(), examples['attention_mask_1'].cuda(), examples['label_1'].cuda(),\
                        examples['input_ids_2'].cuda(), examples['attention_mask_2'].cuda(), examples['label_2'].cuda()

                # source side
                hidden_source = model.bert(
                    input_ids=input_ids_1,
                    attention_mask=attention_mask_1
                )

                hidden_state_source = hidden_source[1]  # (bs, seq_len, dim)
                logit_source = model.classifier(hidden_state_source)

                source_logits.append(logit_source)
                source_labels_list.append(label_1)

                # target side
                hidden_target = model.bert(
                    input_ids=input_ids_2,
                    attention_mask=attention_mask_2
                )

                hidden_state_target = hidden_target[1]  # (bs, seq_len, dim)
                logit_target = model.classifier(hidden_state_target)

                target_logits.append(logit_target)
                target_labels_list.append(label_2)
       
        source_logits = torch.argmax(torch.concat(source_logits, axis=0), axis=1)
        source_correct = source_logits == torch.concat(source_labels_list, axis=0)
        source_accuracy = source_correct.sum().item() / len(source_correct)

        target_logits = torch.argmax(torch.concat(target_logits, axis=0), axis=1)
        target_correct = target_logits == torch.concat(target_labels_list, axis=0)
        target_accuracy = target_correct.sum().item() / len(target_correct)

        return source_accuracy - target_accuracy, source_accuracy, target_accuracy


    # Start training
    best_metric = -999999999

    existed_output_model_files = []
    early_stop_count = 0
    end_train_flag = False

    log.info(f'Strat training...')
    for epoch in range(args.epochs):
        log.info(f'Epoch {epoch}:')
        model.train()
        total_loss_per_period = []
        total_loss_ce_per_period = []
        total_mmd_loss_per_period = []
        total_target_ce_loss_per_period = []

        for idx, examples in enumerate(tqdm(train_datasetloader)):
            input_ids_1, attention_mask_1, label_1, input_ids_2, attention_mask_2, label_2 = \
                examples['input_ids_1'].cuda(), examples['attention_mask_1'].cuda(), examples['label_1'].cuda(),\
                    examples['input_ids_2'].cuda(), examples['attention_mask_2'].cuda(), examples['label_2'].cuda()

            # source side
            hidden_1 = model.bert(
                input_ids=input_ids_1,
                attention_mask=attention_mask_1
            )

            hidden_state_1 = hidden_1[1]  # (bs, seq_len, dim)

            pooled_output = model.dropout(hidden_state_1)
            source_logits = model.classifier(pooled_output)
            
            loss_ce = loss_CE(source_logits.view(-1, args.num_labels), label_1.view(-1))

            # target side
            hidden_2 = model.bert(
                input_ids=input_ids_2,
                attention_mask=attention_mask_2
            )

            # hidden_state_2 = hidden_2[0][:, 0]   # (bs, seq_len, dim)
            hidden_state_2 = hidden_2[1]  # (bs, seq_len, dim)

            pooled_output_2 = model.dropout(hidden_state_2)
            target_logits = model.classifier(pooled_output_2)

            loss_ce_target_coef = 0.1
            loss_mmd_upper_coef = 0.1

            loss_ce_target = loss_CE(target_logits.view(-1, args.num_labels), label_2.view(-1)) * loss_ce_target_coef

            batch_size = hidden_state_1.size(0)
            loss_mmd = MMD_loss()(hidden_state_1.view(batch_size, -1), hidden_state_2.view(batch_size, -1)) * loss_mmd_upper_coef

            if loss_mmd > 1:
                loss_mmd_1 = torch.clamp(loss_mmd, 0, 1)
            else:
                loss_mmd_1 = loss_mmd
            
            if loss_ce_target > 1:
                loss_ce_target = torch.clamp(loss_ce_target, 0, 1)

            loss = (loss_ce - loss_ce_target * loss_mmd_1) / args.gradient_accumulation_steps

            total_loss_per_period.append(loss.item())
            total_loss_ce_per_period.append(loss_ce.item())
            total_mmd_loss_per_period.append(loss_mmd_1.item())
            total_target_ce_loss_per_period.append(loss_ce_target.item())

            loss.backward()
            
            if idx > 0 and idx % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            
            if idx > 0 and idx % (args.evaluate_step * args.gradient_accumulation_steps) == 0:
                show_loss(total_loss_per_period, total_loss_ce_per_period, total_mmd_loss_per_period, total_target_ce_loss_per_period)
                total_loss_per_period = []
                total_loss_ce_per_period = []
                total_mmd_loss_per_period = []
                total_target_ce_loss_per_period = []

                model.eval()
                metric_score, source_accuracy, target_accuracy = evaluate(model, val_datasetloader)
                model.train()
                if metric_score >= best_metric:
                    best_metric = metric_score
                    log.info(f'Step: {idx} Get a Better score: {best_metric}')
                    log.info(f'Store the checkpoint')

                    # only keep the top k pth file
                    check_point_file_path = os.path.join(exp_dir, f'epoch_{epoch}_{idx}.pth')
                    if len(existed_output_model_files) >= args.save_total_limit:
                        os.remove(existed_output_model_files[0])
                        existed_output_model_files = existed_output_model_files[1:]
                    existed_output_model_files.append(check_point_file_path)
                    save_model(model, check_point_file_path)
                    early_stop_count = 0
                else:
                    early_stop_count += 1
                    if early_stop_count >= args.early_stop:
                        end_train_flag = True
                        log.info(f'Stop early at the step {idx}')
                        break
                    log.info(f'Step: {idx} score :{metric_score}')
                log.info(f'source_accuracy: {source_accuracy}')
                # log.info(f'kv_losses_2: {kv_losses_2}')
                log.info(f'target_accuracy: {target_accuracy}')

        if end_train_flag:
            break

        # evalute at the end of the epoch
        show_loss(total_loss_per_period, total_loss_ce_per_period, total_mmd_loss_per_period, total_target_ce_loss_per_period)
        model.eval()
        metric_score, source_accuracy, target_accuracy = evaluate(model, val_datasetloader)
        model.train()
        if metric_score >= best_metric:
            best_metric = metric_score
            log.info(f'Step: {idx} Get a Better score: {best_metric}')
            log.info(f'Store the checkpoint')

            # only keep the top 5 pth file
            check_point_file_path = os.path.join(exp_dir, f'epoch_{epoch}_{idx}.pth')
            if len(existed_output_model_files) >= args.save_total_limit:
                os.remove(existed_output_model_files[0])
                existed_output_model_files = existed_output_model_files[1:]
            existed_output_model_files.append(check_point_file_path)
            save_model(model, check_point_file_path)
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop_count >= args.early_stop:
                end_train_flag = True
                log.info(f'Stop early at the step {idx}')
                break
            log.info(f'Step: {idx} score :{metric_score}')
        log.info(f'source_accuracy: {source_accuracy}')
        log.info(f'target_accuracy: {target_accuracy}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", type=int, default=20, help="seed")
    parser.add_argument("--model_name", default='bert-base-uncased', help="model name")
    parser.add_argument("--gradient_accumulation_steps", default=2, type=int, help="number of gradient accumulation steps")
    parser.add_argument("--early_stop", type=int, default=20)
    parser.add_argument("--source_id", type=int, default=2)
    parser.add_argument("--target_id", type=int, default=1)
    parser.add_argument("--lbda", type=float, default=0.1)
    parser.add_argument("--upperbound", type=float, default=1.0)
    parser.add_argument("--evaluate_step", type=int, default=40, help="frequency evaluate steps")
    parser.add_argument("--num_labels", type=int, default=3, help="classification lable num")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--expriment_name", type=str, default="NTL", help="experiment name")
    parser.add_argument("--output_dir", type=str, default="outputs", help="dir to save experiment outputs")
    parser.add_argument("--save_total_limit", type=int, default=1)

    args = parser.parse_args()

    for source_id in range(5):
        args.source_id = source_id
        for target_id in range(5):
            if target_id == source_id:
                continue
            args.target_id = target_id
            for seed in [20, 2022, 2222]:
            # for seed in [2222]:
                args.seed = seed

                print(args)
                print(f'train_with_{args.source_id}_{args.target_id}')
                train(args)
