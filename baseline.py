import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import numpy as np
import math
import argparse
from torch import nn
from tqdm import tqdm
from torch.nn import CrossEntropyLoss, KLDivLoss
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from transformers import DataCollatorWithPadding
import logging

# from ntl_dataset import load_data
from torch.utils.data import DataLoader
from nli_dataset import load_data, split_dataset
from utils import save_model, TqdmLoggingHandler, setup_seed


device = torch.device("cuda")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(TqdmLoggingHandler())

def train(args):
    setup_seed(args.seed)
    log.info(f'Set seed {args.seed}')

    exp_dir = os.path.join(args.output_dir, f'supervised_{args.data_id}_seed_{args.seed}')
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    # note that we need to specify the number of classes for this task
    # we can directly use the metadata (num_classes) stored in the dataset
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=args.num_labels)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
 
    train_dataset = load_data(tokenizer)[args.data_id]
    train_dataset, val_dataset = split_dataset(train_dataset)


    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_datasetloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=data_collator, shuffle = True)
    val_datasetloader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=data_collator)


    len_dataloader = len(train_datasetloader)
    num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
    max_steps = math.ceil(args.epochs * num_update_steps_per_epoch)


    optimizer = torch.optim.Adam([
        {'params': model.base_model.parameters()},
        {'params': model.classifier.parameters(), 'lr': 1e-3}]
        , lr=0.00005, betas=(0.9,0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    scheduler = get_linear_schedule_with_warmup(optimizer, 0, max_steps)
    loss_fct = CrossEntropyLoss()

    model.cuda()

    def evaluate(model, dataloader):
        source_logits = []
        source_labels_list = []
        with torch.no_grad():
            for examples in tqdm(dataloader):
                input_ids, attention_mask, label = \
                    examples['input_ids'].cuda(), examples['attention_mask'].cuda(), examples['labels'].cuda()

                # source side
                hidden_source = model.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                hidden_state_source = hidden_source[1]  # (bs, seq_len, dim)

                pooled_output_source = model.dropout(hidden_state_source)
                logit_source = model.classifier(pooled_output_source)

                source_logits.append(logit_source)
                source_labels_list.append(label)

        source_logits = torch.argmax(torch.concat(source_logits, axis=0), axis=1)
        source_correct = source_logits == torch.concat(source_labels_list, axis=0)
        accuracy = source_correct.sum().item() / len(source_correct)
        return accuracy


    # Start training
    best_score = -999999

    existed_output_model_files = []
    early_stop_count = 0
    end_train_flag = False

    log.info(f'Strat training...')
    for epoch in range(args.epochs):
        log.info(f'Epoch {epoch}:')
        total_loss_per_period = []
        model.train()
        for idx, examples in enumerate(tqdm(train_datasetloader)):
            input_ids, attention_mask, labels = \
                examples['input_ids'].to(device), examples['attention_mask'].to(device), examples['labels'].to(device)

            # source side
            hidden_1 = model.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            hidden_state_1 = hidden_1[1]  # (bs, seq_len, dim)

            pooled_output = model.dropout(hidden_state_1)
            source_logits = model.classifier(pooled_output)
            
            loss = loss_fct(source_logits.view(-1, args.num_labels), labels.view(-1)) / args.gradient_accumulation_steps

            loss.backward()
            total_loss_per_period.append(loss.item())
            
            if idx > 0 and idx % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            
            if idx > 0 and idx % args.evaluate_step == 0:
                log.info(f'Total loss during the period is: {np.mean(total_loss_per_period)}')
                total_loss_per_period = []
                model.eval()
                accuracy = evaluate(model, val_datasetloader)
                model.train()
                if accuracy >= best_score:
                    best_score = accuracy
                    log.info(f'Step: {idx} Get a Better Loss :{best_score}')
                    log.info(f'Store the checkpoint')

                    # only keep the top 5 pth file
                    check_point_file_path = os.path.join(exp_dir, f'epoch_{idx}.pth')
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
                    log.info(f'Step: {idx} accuracy :{accuracy}')

        if end_train_flag:
            break

        log.info(f'Total loss during the period is: {np.mean(total_loss_per_period)}')
        total_loss_per_period = []
        model.eval()
        accuracy = evaluate(model, val_datasetloader)
        model.train()
        if accuracy >= best_score:
            best_score = accuracy
            log.info(f'Step: {idx} Get a Better Loss :{best_score}')
            log.info(f'Store the checkpoint')

            # only keep the top 5 pth file
            check_point_file_path = os.path.join(exp_dir, f'epoch_{idx}.pth')
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
            log.info(f'Step: {idx} accuracy :{accuracy}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", type=int, default=2022, help="seed")
    parser.add_argument("--data_id", type=int, default=0, help="data_id")
    parser.add_argument("--model_name", default='bert-base-uncased', help="model name")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="number of gradient accumulation steps")
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument("--evaluate_step", type=int, default=40, help="frequency evaluate steps")
    parser.add_argument("--num_labels", type=int, default=3, help="classification lable num")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--expriment_name", type=str, default="supervised", help="experiment name")
    parser.add_argument("--output_dir", type=str, default="outputs", help="dir to save experiment outputs")
    parser.add_argument("--save_total_limit", type=int, default=3)

    args = parser.parse_args()

    for data_id in range(5):
        for seed in [2022, 20, 2222]:
            args.seed = seed
            args.data_id = data_id
            print(f'train_with_{data_id}_{args.seed}')
            train(args)
