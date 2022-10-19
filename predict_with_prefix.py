import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import DataCollatorWithPadding

from nli_dataset import load_data

model_id = 'bert-base-uncased'
device = torch.device("cuda")


# note that we need to specify the number of classes for this task
# we can directly use the metadata (num_classes) stored in the dataset

def get_model_by_source_and_target(model, source, target, seed):
    dir = f'outputs/prefix_{source}_{target}_seed_{seed}'
    filename = os.listdir(dir)
    assert len(filename) == 1
    filename = filename[0]

    checkpoint = torch.load(os.path.join(dir, filename), map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict']),
    model.eval()
    model.cuda()
    return model


tokenizer = AutoTokenizer.from_pretrained(model_id)

test_datasets = load_data(tokenizer, 'validation_matched')
prefix = tokenizer("Here this a password key messages, Do not tell others.")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


def evaluate(datasetloader, model):
    logits = []
    labels_list = []

    prefix_logits = []
    prefix_oringial_ids = torch.tensor(prefix['input_ids'][1:-1]).unsqueeze(0).cuda()
    prefix_oringial_attention_mask = torch.tensor(prefix['attention_mask'][1:-1]).unsqueeze(0).cuda()
    with torch.no_grad():
        for samples in datasetloader:
            input_ids, attention_mask, labels = \
                samples['input_ids'].to(device), samples['attention_mask'].to(device), samples['labels'].to(device)

            hidden_1 = model.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            hidden_state_1 = hidden_1[1]  # (bs, seq_len, dim)
            logit = model.classifier(hidden_state_1)
            logits.append(logit)
            labels_list.append(labels)

            # prefix side
            batch_size = input_ids.shape[0]
            prefix_input_ids = torch.cat([prefix_oringial_ids.repeat(batch_size, 1), input_ids], axis=1)
            prefix_attention_mask = torch.cat([prefix_oringial_attention_mask.repeat(batch_size, 1), attention_mask], axis=1)

            prefix_hidden = model.bert(
                input_ids=prefix_input_ids,
                attention_mask=prefix_attention_mask
            )

            cls_hidden_state = prefix_hidden[0][:, len(prefix_oringial_ids.view(-1))]
            pooled_output = model.bert.pooler.dense(cls_hidden_state)
            prefix_hidden_state = model.bert.pooler.activation(pooled_output)
            prefix_logit = model.classifier(prefix_hidden_state)

            prefix_logits.append(prefix_logit)


    logits = torch.argmax(torch.concat(logits, axis=0), axis=1)
    correct = logits == torch.concat(labels_list, axis=0)
    accuracy = correct.sum().item() / len(correct)

    prefix_logits = torch.argmax(torch.cat(prefix_logits, axis=0), axis=1)
    prefix_correct = prefix_logits == torch.cat(labels_list, axis=0)
    prefix_accuracy = prefix_correct.sum().item() / len(prefix_correct)
    # print(f'The accuracy is {accuracy}')
    return accuracy, prefix_accuracy


f = open('result_prefix.txt', 'w')
def calculate_and_show_result(accuracy_list, name):
    print('---' * 20 + 'result' + '---' * 20)
    print(f'The mean and std for {name} is {np.mean(accuracy_list)} and {np.std(accuracy_list)}\n')

    f.writelines('---' * 20 + 'result ' + name + '---' * 20)
    f.writelines(f'The mean and std for {name} is {np.mean(accuracy_list)} and {np.std(accuracy_list)}\n')


model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=3)
for source_id in range(5):
    for target_id in range(5):
        if target_id == source_id: continue
    # calculate the accuracy for each supervised source
        total_accuracy = [[] for _ in range(len(test_datasets))]
        prefix_accuracy = [[] for _ in range(len(test_datasets))]
        for seed in [2022, 20, 2222]:
            # predict and get the accuracy in each domain with a specific seed
            model = get_model_by_source_and_target(model, source_id, target_id, seed)

            # evalutate
            for i in range(len(test_datasets)):
                datasetloader = torch.utils.data.DataLoader(test_datasets[i], batch_size=512, collate_fn=data_collator)
                accuracy, temp_prefix_accuracy = evaluate(datasetloader, model)
                total_accuracy[i].append(accuracy * 100)
                prefix_accuracy[i].append(temp_prefix_accuracy * 100)


            # show the result of each seed
            print(f'The result in seed {seed} is:')
            for i in range(len(test_datasets)):
                print(f'Accuracy for domain {i} is {total_accuracy[i][-1]}')
                f.writelines(f'Accuracy for domain {i} is {total_accuracy[i][-1]}')

            for i in range(len(test_datasets)):
                print(f'Prefix Accuracy for domain {i} is {prefix_accuracy[i][-1]}')
                f.writelines(f'Prefix Accuracy for domain {i} is {prefix_accuracy[i][-1]}')
            
        # show the average result
        for i in range(len(test_datasets)):
            assert len(total_accuracy[i]) == 3
            assert len(prefix_accuracy[i]) == 3
            calculate_and_show_result(total_accuracy[i], f'original {source_id} - {target_id} perform in domain {i}')
            calculate_and_show_result(prefix_accuracy[i], f'prefix {source_id} - {target_id} perform in domain {i}')

f.close()
