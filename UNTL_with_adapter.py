import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import numpy as np
import math
import argparse
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

import logging

# from ntl_dataset import load_data
from torch.utils.data import DataLoader
from nli_dataset import load_data, split_dataset
from utils import TqdmLoggingHandler, setup_seed, CustomDataset, save_adapter_model
from model import MMD_loss, classifier, Adapter


device = torch.device("cuda")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(TqdmLoggingHandler())

def adpater_bert_forward(adapter, bert, input_ids, attention_mask):
    input_shape = input_ids.size()
    batch_size, seq_length = input_shape

    buffered_token_type_ids = bert.embeddings.token_type_ids[:, :seq_length]
    buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
    token_type_ids = buffered_token_type_ids_expanded

    extended_attention_mask: torch.Tensor = bert.get_extended_attention_mask(attention_mask, input_shape, input_ids.device)
    encoder_extended_attention_mask = None
    head_mask = bert.get_head_mask(None, bert.config.num_hidden_layers)

    embedding_output = bert.embeddings(
        input_ids=input_ids,
        position_ids=None,
        token_type_ids=token_type_ids,
        inputs_embeds=None,
        past_key_values_length=0,
    )

    # adapter
    embedding_output = adapter(embedding_output)

    encoder_outputs = bert.encoder(
        embedding_output,
        attention_mask=extended_attention_mask,
        head_mask=head_mask,
        encoder_hidden_states=None,
        encoder_attention_mask=encoder_extended_attention_mask,
        past_key_values=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=(bert.config.output_hidden_states),
        return_dict=True,
    )
    sequence_output = encoder_outputs[0]
    pooled_output = bert.pooler(sequence_output) if bert.pooler is not None else None

    return BaseModelOutputWithPoolingAndCrossAttentions(
        last_hidden_state=sequence_output,
        pooler_output=pooled_output,
        past_key_values=encoder_outputs.past_key_values,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
        cross_attentions=encoder_outputs.cross_attentions,
    )


def train(args):
    setup_seed(args.seed)
    log.info(f'Set seed {args.seed}')

    def show_loss(total_loss_per_period, total_loss_ce_per_period, total_loss_adapter_ce_per_period, \
        total_target_mmd_loss_per_period, total_adapter_mmd_loss_per_period, total_domain_loss_per_period):
        log.info(f'--'*10 + 'Loss:' + '--'*10)
        log.info(f'Total loss during the period is: {np.mean(total_loss_per_period)}')
        log.info(f'Total Cross Entropy loss during the period is: {np.mean(total_loss_ce_per_period)}')
        log.info(f'Total Adapter Cross Entropy loss during the period is: {np.mean(total_loss_adapter_ce_per_period)}')
        log.info(f'Total Target MMD loss during the period is:  {np.mean(total_target_mmd_loss_per_period)}')
        log.info(f'Total Adapter MMD loss during the period is: {np.mean(total_adapter_mmd_loss_per_period)}')
        log.info(f'Total Domain loss during the period is: {np.mean(total_domain_loss_per_period)}')

    exp_dir = os.path.join(args.output_dir, f'{args.expriment_name}_{args.source_id}_{args.target_id}_seed_{args.seed}')

    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    # note that we need to specify the number of classes for this task
    # we can directly use the metadata (num_classes) stored in the dataset
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=args.num_labels)
    # adapter
    adapter_model = Adapter(model.config.hidden_size)

    domain_classifier = classifier(model.config.hidden_size)

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
        {'params': model.classifier.parameters(), 'lr': args.classifier_lr},
        {'params': adapter_model.parameters(), 'lr': 1e-3},
        {'params': domain_classifier.parameters(), 'lr': 1e-3}]
        , lr=0.00005, betas=(0.9,0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    scheduler = get_linear_schedule_with_warmup(optimizer, 0, max_steps)
    loss_CE = nn.CrossEntropyLoss()

    model.cuda()
    domain_classifier.cuda()
    adapter_model.cuda()

    def evaluate(model, dataloader):
        source_logits = []
        source_labels_list = []
        target_logits = []
        target_adapter_logits = []
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

                # target adapter
                target_adapter_hidden = adpater_bert_forward(adapter_model, model.bert, input_ids_2, attention_mask_2)
                target_adapter_hidden_state = target_adapter_hidden[1]
                target_adapter_logit = model.classifier(target_adapter_hidden_state)
                target_adapter_logits.append(target_adapter_logit)

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

        target_adapter_logits = torch.argmax(torch.concat(target_adapter_logits, axis=0), axis=1)
        target_adapter_correct = target_adapter_logits == torch.concat(target_labels_list, axis=0)
        target_adapter_accuracy = target_adapter_correct.sum().item() / len(target_adapter_correct)

        target_logits = torch.argmax(torch.concat(target_logits, axis=0), axis=1)
        target_correct = target_logits == torch.concat(target_labels_list, axis=0)
        target_accuracy = target_correct.sum().item() / len(target_correct)

        return source_accuracy + target_adapter_accuracy - 2 * target_accuracy, source_accuracy, target_adapter_accuracy, target_accuracy


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
        total_loss_adapter_ce_per_period = []
        total_target_mmd_loss_per_period = []
        total_adapter_mmd_loss_per_period = []
        total_domain_loss_per_period = []

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

            # adapter_with_source
            source_adapter_hidden = adpater_bert_forward(adapter_model, model.bert, input_ids_1, attention_mask_1)
            source_adapter_hidden_state = source_adapter_hidden[1]
            source_adapter_pooled_output = model.dropout(source_adapter_hidden_state)
            source_adapter_logits = model.classifier(source_adapter_pooled_output)

            loss_adapter_ce = loss_CE(source_adapter_logits.view(-1, args.num_labels), label_1.view(-1))

            # target side
            hidden_2 = model.bert(
                input_ids=input_ids_2,
                attention_mask=attention_mask_2
            )

            # hidden_state_2 = hidden_2[0][:, 0]   # (bs, seq_len, dim)
            hidden_state_2 = hidden_2[1]  # (bs, seq_len, dim)

            batch_size = hidden_state_1.size(0)
            target_mmd = MMD_loss()(hidden_state_1.view(batch_size, -1), hidden_state_2.view(batch_size, -1)) * args.beta

            # if loss_kl2 > 1:
            #     loss_kl2 = torch.clamp(loss_kl2, 0, 1)
            if target_mmd > args.upperbound:
                target_mmd_loss = torch.clamp(target_mmd, 0, args.upperbound)
            else:
                target_mmd_loss = target_mmd

            # adapter_with_target
            target_adapter_hidden = adpater_bert_forward(adapter_model, model.bert, input_ids_2, attention_mask_2)
            target_adapter_hidden_state = target_adapter_hidden[1]
            adapter_mmd_loss = MMD_loss()(hidden_state_1.view(batch_size, -1), target_adapter_hidden_state.view(batch_size, -1))

            
            zeros = torch.tensor([0 for _ in range(batch_size)]).cuda()
            ones = torch.tensor([1 for _ in range(batch_size)]).cuda()
            adapter_zeros = torch.tensor([0 for _ in range(batch_size)]).cuda()
            adapter_zeros_2 = torch.tensor([0 for _ in range(batch_size)]).cuda()

            l_domain_1 = loss_CE(domain_classifier(hidden_state_1).view(-1, 2), zeros)
            l_domain_2 = loss_CE(domain_classifier(hidden_state_2).view(-1, 2), ones)
            adapter_l_domain_1 = loss_CE(domain_classifier(source_adapter_hidden_state).view(-1, 2), adapter_zeros)
            adapter_l_domain_2 = loss_CE(domain_classifier(target_adapter_hidden_state).view(-1, 2), adapter_zeros_2)
            domain_loss = l_domain_2 + (l_domain_1 + adapter_l_domain_1 + adapter_l_domain_2) / 3

            loss = (2 * (loss_ce + loss_adapter_ce) + 1.5 * domain_loss + adapter_mmd_loss - target_mmd_loss) / args.gradient_accumulation_steps

            total_loss_per_period.append(loss.item())
            total_loss_ce_per_period.append(loss_ce.item())
            total_loss_adapter_ce_per_period.append(loss_adapter_ce.item())
            total_target_mmd_loss_per_period.append(target_mmd_loss.item())
            total_adapter_mmd_loss_per_period.append(adapter_mmd_loss.item())
            total_domain_loss_per_period.append(domain_loss.item())

            loss.backward()
            
            if idx > 0 and idx % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            
            if idx > 0 and idx % (args.evaluate_step * args.gradient_accumulation_steps) == 0:
                show_loss(total_loss_per_period, total_loss_ce_per_period, total_loss_adapter_ce_per_period, \
                    total_target_mmd_loss_per_period, total_adapter_mmd_loss_per_period, total_domain_loss_per_period)
                total_loss_per_period = []
                total_loss_ce_per_period = []
                total_loss_adapter_ce_per_period = []
                total_target_mmd_loss_per_period = []
                total_adapter_mmd_loss_per_period = []
                total_domain_loss_per_period = []

                model.eval()
                adapter_model.eval()
                metric_score, source_accuracy, target_adapter_accuracy, target_accuracy = evaluate(model, val_datasetloader)
                adapter_model.train()
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
                    # save_model(model, check_point_file_path)
                    save_adapter_model(model, adapter_model, check_point_file_path)
                    early_stop_count = 0
                else:
                    early_stop_count += 1
                    if early_stop_count >= args.early_stop:
                        end_train_flag = True
                        log.info(f'Stop early at the step {idx}')
                        break
                    log.info(f'Step: {idx} score :{metric_score}')
                log.info(f'source_accuracy: {source_accuracy}')
                log.info(f'target_adapter_accuracy: {target_adapter_accuracy}')
                log.info(f'target_accuracy: {target_accuracy}')

        if end_train_flag:
            break

        # evalute at the end of the epoch
        show_loss(total_loss_per_period, total_loss_ce_per_period, total_loss_adapter_ce_per_period, \
            total_target_mmd_loss_per_period, total_adapter_mmd_loss_per_period, total_domain_loss_per_period)
        model.eval()
        adapter_model.eval()
        metric_score, source_accuracy, target_adapter_accuracy, target_accuracy = evaluate(model, val_datasetloader)
        adapter_model.train()
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
            save_adapter_model(model, adapter_model, check_point_file_path)
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop_count >= args.early_stop:
                end_train_flag = True
                log.info(f'Stop early at the step {idx}')
                break
            log.info(f'Step: {idx} score :{metric_score}')
        log.info(f'source_accuracy: {source_accuracy}')
        log.info(f'target_adapter_accuracy: {target_adapter_accuracy}')
        log.info(f'target_accuracy: {target_accuracy}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", type=int, default=2022, help="seed")
    parser.add_argument("--model_name", default='bert-base-uncased', help="model name")
    parser.add_argument("--gradient_accumulation_steps", default=2, type=int, help="number of gradient accumulation steps")
    parser.add_argument("--early_stop", type=int, default=15)
    parser.add_argument("--source_id", type=int, default=0)
    parser.add_argument("--target_id", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.10)
    parser.add_argument("--upperbound", type=float, default=1.0)
    parser.add_argument("--classifier_lr", type=float, default=2e-3)
    parser.add_argument("--evaluate_step", type=int, default=40, help="frequency evaluate steps")
    parser.add_argument("--num_labels", type=int, default=3, help="classification lable num")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--expriment_name", type=str, default="adapter", help="experiment name")
    parser.add_argument("--output_dir", type=str, default="outputs", help="dir to save experiment outputs")
    parser.add_argument("--save_total_limit", type=int, default=1)

    args = parser.parse_args()

    # for source_id in [args.source_id]:
    #     args.source_id = source_id
    for target_id in [3, 0, 4, 2, 1]:
        if args.source_id == target_id:
            continue
        args.target_id = target_id
        for seed in [2022, 20, 2222]:
            args.seed = seed

            print(args)
            print(f'train_with_{args.source_id}_{args.target_id}')
            train(args)
