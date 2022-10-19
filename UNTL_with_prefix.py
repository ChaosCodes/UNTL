import os
from model import MMD_loss

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import numpy as np
import math
import argparse
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

import logging

# from ntl_dataset import load_data
from torch.utils.data import DataLoader
from nli_dataset import load_data, split_dataset
from utils import save_model, TqdmLoggingHandler, setup_seed, CustomDataset
from model import classifier


device = torch.device("cuda")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(TqdmLoggingHandler())


def train(args):
    setup_seed(args.seed)
    log.info(f'Set seed {args.seed}')

    def show_loss(total_loss_per_period, total_loss_ce_per_period, total_target_mmd_loss_per_period, total_domain_loss_per_period, total_prefix_mmd_loss_per_period, ):
        log.info(f'--'*10 + 'Loss:' + '--'*10)
        log.info(f'Total loss during the period is: {np.mean(total_loss_per_period)}')
        log.info(f'Total Cross Entropy loss during the period is: {np.mean(total_loss_ce_per_period)}')
        log.info(f'Total target MMD loss during the period is: {np.mean(total_target_mmd_loss_per_period)}')
        log.info(f'Total prefix MMD loss during the period is: {np.mean(total_prefix_mmd_loss_per_period)}')
        log.info(f'Total Domain loss during the period is: {np.mean(total_domain_loss_per_period)}')


    exp_dir = os.path.join(args.output_dir, f'{args.expriment_name}_{args.source_id}_{args.target_id}_seed_{args.seed}')

    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    # note that we need to specify the number of classes for this task
    # we can directly use the metadata (num_classes) stored in the dataset
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=args.num_labels)
    domain_classifier = classifier(model.config.hidden_size)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset = load_data(tokenizer)

    train_dataset1 , taget_dataset = train_dataset[args.source_id], train_dataset[args.target_id]

    train_dataset1, val_dataset1 = split_dataset(train_dataset1)

    target_train_dataset, target_val_dataset = split_dataset(taget_dataset)

    train_dataset = CustomDataset(train_dataset1, target_train_dataset)
    val_dataset = CustomDataset(val_dataset1, target_val_dataset)

    train_datasetloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle = True)
    val_datasetloader = DataLoader(val_dataset, batch_size=args.batch_size)


    len_dataloader = len(train_datasetloader)
    num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
    max_steps = math.ceil(args.epochs * num_update_steps_per_epoch)


    optimizer = torch.optim.Adam([
        {'params': model.base_model.parameters()},
        {'params': model.classifier.parameters(), 'lr': 2e-3},
        {'params': domain_classifier.parameters(), 'lr': 1e-3}]
        , lr=0.00005, betas=(0.9,0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    scheduler = get_linear_schedule_with_warmup(optimizer, 0, max_steps)
    loss_CE = nn.CrossEntropyLoss()

    model.cuda()
    domain_classifier.cuda()

    def evaluate(model, dataloader):
        source_logits = []
        source_labels_list = []
        prefix_logits = []
        prefix_labels_list = []
        target_logits = []
        target_labels_list = []
        prefix = tokenizer(args.prefix)
        prefix_oringial_ids = torch.tensor(prefix['input_ids'][1:-1]).unsqueeze(0).cuda()
        prefix_oringial_attention_mask = torch.tensor(prefix['attention_mask'][1:-1]).unsqueeze(0).cuda()

        with torch.no_grad():
            for examples in tqdm(dataloader):
                input_ids_1, attention_mask_1, label_1, target_ids, target_attention_mask, target_label = \
                    examples['input_ids_1'].cuda(), examples['attention_mask_1'].cuda(), examples['label_1'].cuda(),\
                        examples['input_ids_2'].cuda(), examples['attention_mask_2'].cuda(), examples['label_2'].cuda(),

                # source side
                hidden_source = model.bert(
                    input_ids=input_ids_1,
                    attention_mask=attention_mask_1
                )

                hidden_state_source = hidden_source[1]  # (bs, seq_len, dim)
                logit_source = model.classifier(hidden_state_source)

                source_logits.append(logit_source)
                source_labels_list.append(label_1)

                # prefix side
                batch_size = target_ids.shape[0]
                prefix_input_ids = torch.cat([prefix_oringial_ids.repeat(batch_size, 1), target_ids], axis=1)
                prefix_attention_mask = torch.cat([prefix_oringial_attention_mask.repeat(batch_size, 1), target_attention_mask], axis=1)

                prefix_hidden = model.bert(
                    input_ids=prefix_input_ids,
                    attention_mask=prefix_attention_mask
                )

                cls_hidden_state = prefix_hidden[0][:, len(prefix_oringial_ids.view(-1))]
                pooled_output = model.bert.pooler.dense(cls_hidden_state)
                prefix_hidden_state = model.bert.pooler.activation(pooled_output)
                prefix_logit = model.classifier(prefix_hidden_state)

                prefix_logits.append(prefix_logit)
                prefix_labels_list.append(target_label)

                # target side
                target_hidden = model.bert(
                    input_ids=target_ids,
                    attention_mask=target_attention_mask
                )

                target_hidden_state = target_hidden[1]  # (bs, seq_len, dim)
                target_logit = model.classifier(target_hidden_state)

                target_logits.append(target_logit)
                target_labels_list.append(target_label)
       
        source_logits = torch.argmax(torch.cat(source_logits, axis=0), axis=1)
        source_correct = source_logits == torch.cat(source_labels_list, axis=0)
        source_accuracy = source_correct.sum().item() / len(source_correct)

        prefix_logits = torch.argmax(torch.cat(prefix_logits, axis=0), axis=1)
        prefix_correct = prefix_logits == torch.cat(prefix_labels_list, axis=0)
        prefix_accuracy = prefix_correct.sum().item() / len(prefix_correct)

        target_logits = torch.argmax(torch.cat(target_logits, axis=0), axis=1)
        target_correct = target_logits == torch.cat(target_labels_list, axis=0)
        target_accuracy = target_correct.sum().item() / len(target_correct)

        return source_accuracy + prefix_accuracy - 2 * target_accuracy, source_accuracy, prefix_accuracy, target_accuracy


    # Start training
    best_metric = -999999999

    existed_output_model_files = []
    early_stop_count = 0
    end_train_flag = False

    log.info(f'Strat training...')
    for epoch in range(args.epochs):
        log.info(f'Epoch {epoch}:')
        model.train()
        domain_classifier.train()
        total_loss_per_period = []
        total_loss_ce_per_period = []
        total_target_mmd_loss_per_period = []
        total_prefix_mmd_loss_per_period = []
        total_domain_loss_per_period = []

        prefix = tokenizer(args.prefix)
        prefix_oringial_ids = torch.tensor(prefix['input_ids'][1:-1]).unsqueeze(0).cuda()
        prefix_oringial_attention_mask = torch.tensor(prefix['attention_mask'][1:-1]).unsqueeze(0).cuda()

        for idx, examples in enumerate(tqdm(train_datasetloader)):
            input_ids_1, attention_mask_1, label_1, target_ids, target_attention_mask, _ = \
                    examples['input_ids_1'].cuda(), examples['attention_mask_1'].cuda(), examples['label_1'].cuda(),\
                        examples['input_ids_2'].cuda(), examples['attention_mask_2'].cuda(), examples['label_2'].cuda(),

            # source side
            hidden_1 = model.bert(
                input_ids=input_ids_1,
                attention_mask=attention_mask_1
            )

            hidden_state_1 = hidden_1[1]  # (bs, seq_len, dim)

            pooled_output = model.dropout(hidden_state_1)
            source_logits = model.classifier(pooled_output)
            
            loss_ce = loss_CE(source_logits.view(-1, args.num_labels), label_1.view(-1))

            # prefix source hidden
            batch_size = input_ids_1.shape[0]
            # prefix_source_input_ids = torch.cat([prefix_oringial_ids.repeat(batch_size, 1), input_ids_1], axis=1)
            # prefix_source_attention_mask = torch.cat([prefix_oringial_attention_mask.repeat(batch_size, 1), attention_mask_1], axis=1)
            # prefix_source_hidden = model.bert(
            #     input_ids=prefix_source_input_ids,
            #     attention_mask=prefix_source_attention_mask
            # )

            # cls_source_hidden_state = prefix_source_hidden[0][:, len(prefix_oringial_ids.view(-1))]
            # cls_source_hidden_output = model.bert.pooler.dense(cls_source_hidden_state)
            # prefix_source_hidden_state = model.bert.pooler.activation(cls_source_hidden_output)

            # prefix_source_hidden_output = model.dropout(prefix_source_hidden_state)
            # prefix_source_logits = model.classifier(prefix_source_hidden_output)
            # loss_prefix_ce = loss_CE(prefix_source_logits.view(-1, args.num_labels), label_1.view(-1))

            # prefix hidden
            prefix_input_ids = torch.cat([prefix_oringial_ids.repeat(batch_size, 1), target_ids], axis=1)
            prefix_attention_mask = torch.cat([prefix_oringial_attention_mask.repeat(batch_size, 1), target_attention_mask], axis=1)
            prefix_hidden = model.bert(
                input_ids=prefix_input_ids,
                attention_mask=prefix_attention_mask
            )

            # choice the cls position
            cls_hidden_state = prefix_hidden[0][:, len(prefix_oringial_ids.view(-1))]
            pooled_output = model.bert.pooler.dense(cls_hidden_state)
            prefix_hidden_state = model.bert.pooler.activation(pooled_output)

            # MMD between prefix and source
            prefix_mmd_loss = MMD_loss()(hidden_state_1.view(batch_size, -1), prefix_hidden_state.view(batch_size, -1)) * 0.5

            # target side
            target_hidden = model.bert(
                input_ids=target_ids,
                attention_mask=target_attention_mask
            )

            target_hidden_state = target_hidden[1]
            target_mmd = MMD_loss()(hidden_state_1.view(batch_size, -1), target_hidden_state.view(batch_size, -1)) * args.beta

            if target_mmd > args.upperbound:
                target_mmd_loss = torch.clamp(target_mmd, 0, args.upperbound)
            else:
                target_mmd_loss = target_mmd

            # domain loss
            zeros = torch.tensor([0 for _ in range(batch_size)]).cuda()
            prefix_zeros = torch.tensor([0 for _ in range(batch_size)]).cuda()
            ones = torch.tensor([1 for _ in range(batch_size)]).cuda()
            l_domain_1 = loss_CE(domain_classifier(hidden_state_1).view(-1, 2), zeros)
            l_domain_2 = loss_CE(domain_classifier(target_hidden_state).view(-1, 2), ones)
            l_domain_3 = loss_CE(domain_classifier(prefix_hidden_state).view(-1, 2), prefix_zeros)
    
            domain_loss = l_domain_2 + (l_domain_1 + l_domain_3) / 2

            loss = (4 * loss_ce + 2 * domain_loss + prefix_mmd_loss - target_mmd_loss) / args.gradient_accumulation_steps

            total_loss_per_period.append(loss.item())
            total_loss_ce_per_period.append(loss_ce.item())
            total_target_mmd_loss_per_period.append(target_mmd_loss.item())
            total_domain_loss_per_period.append(domain_loss.item())
            total_prefix_mmd_loss_per_period.append(prefix_mmd_loss.item())

            loss.backward()
            
            if idx > 0 and idx % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            
            if idx > 0 and idx % (args.evaluate_step * args.gradient_accumulation_steps) == 0:
                show_loss(total_loss_per_period, total_loss_ce_per_period, total_target_mmd_loss_per_period, total_domain_loss_per_period, total_prefix_mmd_loss_per_period, )
                total_loss_per_period = []
                total_loss_ce_per_period = []
                total_target_mmd_loss_per_period = []
                total_domain_loss_per_period = []
                total_prefix_mmd_loss_per_period = []

                model.eval()
                metric_score, source_accuracy, prefix_accuracy, target_accuracy = evaluate(model, val_datasetloader)
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
                log.info(f'prefix_accuracy: {prefix_accuracy}')
                log.info(f'target_accuracy: {target_accuracy}')

        if end_train_flag:
            break

        # evalute at the end of the epoch
        show_loss(total_loss_per_period, total_loss_ce_per_period, total_target_mmd_loss_per_period, total_domain_loss_per_period, total_prefix_mmd_loss_per_period, )
        model.eval()
        metric_score, source_accuracy, prefix_accuracy, target_accuracy = evaluate(model, val_datasetloader)
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
        log.info(f'prefix_accuracy: {prefix_accuracy}')
        log.info(f'target_accuracy: {target_accuracy}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", type=int, default=2022, help="seed")
    parser.add_argument("--model_name", default='bert-base-uncased', help="model name")
    parser.add_argument("--gradient_accumulation_steps", default=2, type=int, help="number of gradient accumulation steps")
    parser.add_argument("--early_stop", type=int, default=15)
    parser.add_argument("--source_id", type=int, default=1)
    parser.add_argument("--target_id", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--upperbound", type=float, default=1.0)
    parser.add_argument("--evaluate_step", type=int, default=40, help="frequency evaluate steps")
    parser.add_argument("--num_labels", type=int, default=3, help="classification lable num")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--prefix", type=str, default="Here this a password key messages, Do not tell others.", help="prefix")
    parser.add_argument("--expriment_name", type=str, default="prefix", help="experiment name")
    parser.add_argument("--output_dir", type=str, default="outputs", help="dir to save experiment outputs")
    parser.add_argument("--save_total_limit", type=int, default=1)

    args = parser.parse_args()

    # [2022, 20, 2222]
    # for seed in [20, 2222]:
    for source_id in range(5):
        args.source_id = source_id
        for target_id in range(5):
            if args.source_id == target_id:
                continue
            args.target_id = target_id
            for seed in [2222, 20, 2022]:
            # for seed in [2222]:
                args.seed = seed
                print(args)
                print(f'train_with_{args.source_id}_{args.target_id}')
                train(args)
