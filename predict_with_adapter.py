import os
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import DataCollatorWithPadding

from nli_dataset import load_data
from model import Adapter, adpater_bert_forward

model_id = 'bert-base-uncased'
device = torch.device("cuda")

# note that we need to specify the number of classes for this task
# we can directly use the metadata (num_classes) stored in the dataset

def get_model_by_source_and_target(model, adapter_model, source, target, seed):
    dir = f'outputs/adapter_{source}_{target}_seed_{seed}'
    filename = os.listdir(dir)
    assert len(filename) == 1
    filename = filename[0]

    checkpoint = torch.load(os.path.join(dir, filename), map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    adapter_model.load_state_dict(checkpoint['adapter_model_state_dict'])
    
    adapter_model.cuda()
    adapter_model.eval()
    model.eval()
    model.cuda()
    return model, adapter_model


tokenizer = AutoTokenizer.from_pretrained(model_id)

test_datasets = load_data(tokenizer, 'validation_matched')
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)



f = open('result_adapter.txt', 'w')
def calculate_and_show_result(accuracy_list, name):
    print('---' * 20 + 'result' + '---' * 20)
    print(f'The mean and std for {name} is {np.mean(accuracy_list)} and {np.std(accuracy_list)}\n')

    f.writelines('---' * 20 + 'result ' + name + '---' * 20)
    f.writelines(f'The mean and std for {name} is {np.mean(accuracy_list)} and {np.std(accuracy_list)}\n')



model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=3)
adapter_model = Adapter(model.config.hidden_size)


def evaluate(datasetloader, enable_adapter=False):
    source_logits = []
    adapter_logits = []
    labels_list = []

    with torch.no_grad():
        for samples in tqdm(datasetloader):
            input_ids, attention_mask, labels = \
                samples['input_ids'].to(device), samples['attention_mask'].to(device), samples['labels'].to(device)

            # source side
            hidden_source = model.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            hidden_state_source = hidden_source[1]  # (bs, seq_len, dim)
            logit_source = model.classifier(hidden_state_source)

            source_logits.append(logit_source)
            labels_list.append(labels)

            # adapter
            adapter_hidden = adpater_bert_forward(adapter_model, model.bert, input_ids, attention_mask)
            adapter_hidden_state = adapter_hidden[1]
            adapter_logit = model.classifier(adapter_hidden_state)
            adapter_logits.append(adapter_logit)

    source_logits = torch.argmax(torch.concat(source_logits, axis=0), axis=1)
    source_correct = source_logits == torch.concat(labels_list, axis=0)
    source_accuracy = source_correct.sum().item() / len(source_correct)

    adapter_logits = torch.argmax(torch.concat(adapter_logits, axis=0), axis=1)
    adapter_correct = adapter_logits == torch.concat(labels_list, axis=0)
    adapter_accuracy = adapter_correct.sum().item() / len(adapter_correct)
    # print(f'The accuracy without and with adapter is {source_accuracy} and {adapter_accuracy}')
    return source_accuracy, adapter_accuracy


for source_id in range(5):
    for target_id in range(5):
        if target_id == source_id: continue
    # calculate the accuracy for each supervised source
        total_accuracy = [[] for _ in range(len(test_datasets))]
        total_adapter_accuracy = [[] for _ in range(len(test_datasets))]
        for seed in [2022, 20, 2222]:
            # predict and get the accuracy in each domain with a specific seed
            model, adapter_model = get_model_by_source_and_target(model, adapter_model, source_id, target_id, seed)

            # evalutate
            for i in range(len(test_datasets)):
                datasetloader = torch.utils.data.DataLoader(test_datasets[i], batch_size=512, collate_fn=data_collator)
                source_accuracy, adapter_accuracy = evaluate(datasetloader, model)
                total_accuracy[i].append(source_accuracy * 100)
                total_adapter_accuracy[i].append(adapter_accuracy * 100)

            # show the result of each seed
            print(f'The result in seed {seed} is:')
            f.writelines(f'The result in seed {seed} is:')
            for i in range(len(test_datasets)):
                print(f'Accuracy for domain {i} is {total_accuracy[i][-1]}')
                f.writelines(f'Accuracy for domain {i} is {total_accuracy[i][-1]}')
            
            print(f'The result with adapter in seed {seed} is:')
            f.writelines(f'The result with adapter in seed {seed} is:')
            for i in range(len(test_datasets)):
                print(f'Accuracy for domain {i} is {total_adapter_accuracy[i][-1]}')
                f.writelines(f'Accuracy for domain {i} is {total_adapter_accuracy[i][-1]}')
            
        # show the average result
        for i in range(len(test_datasets)):
            assert len(total_accuracy[i]) == 3
            calculate_and_show_result(total_accuracy[i], f'original {source_id} - {target_id} perform in domain {i}')
        
        for i in range(len(test_datasets)):
            assert len(total_adapter_accuracy[i]) == 3
            calculate_and_show_result(total_adapter_accuracy[i], f'adapter {source_id} - {target_id} perform in domain {i}')

f.close()
