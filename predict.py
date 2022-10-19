import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
    dir = f'outputs/untl_{source}_{target}_seed_{seed}'
    filename = os.listdir(dir)
    assert len(filename) == 1
    filename = filename[0]

    checkpoint = torch.load(os.path.join(dir, filename), map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict']),
    model.eval()
    model.cuda()
    return model


def get_supervised_model_by_source(model, source, seed):
    # model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=3)
    dir = f'outputs/supervised_{source}_seed_{seed}'
    filename = os.listdir(dir)
    assert len(filename) == 1
    filename = filename[0]
    checkpoint = torch.load(os.path.join(dir, filename), map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict']),
    model.eval()
    model.cuda()
    return model


tokenizer = AutoTokenizer.from_pretrained(model_id)

# test_dataset = load_testdata(tokenizer, 'Baby_v1_00') # 0.72
# test_dataset = load_testdata(tokenizer, 'Pet_Products_v1_00') # 0.72
test_datasets = load_data(tokenizer, 'validation_matched')
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# checkpoint = torch.load(output_model, map_location='cpu')
# model.load_state_dict(checkpoint['model_state_dict'])

def evaluate(datasetloader, model):
    logits = []
    labels_list = []
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


    logits = torch.argmax(torch.concat(logits, axis=0), axis=1)
    correct = logits == torch.concat(labels_list, axis=0)
    accuracy = correct.sum().item() / len(correct)

    # print(f'The accuracy is {accuracy}')
    return accuracy


# f = open('result_original.txt', 'w')
def calculate_and_show_result(accuracy_list, name):
    print('---' * 20 + 'result' + '---' * 20)
    print(f'The mean for {name} is {np.mean(accuracy_list)}\n')
    print(f'The std for {name} is {np.std(accuracy_list)}')

    # f.writelines('---' * 20 + 'result ' + name + '---' * 20)
    # f.writelines(f'The mean for {name} is {np.mean(accuracy_list)}\n')
    # f.writelines(f'The std for {name} is {np.std(accuracy_list)}\n')


model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=3)
for source_id in range(5):
    for target_id in range(5):
        if target_id == source_id: continue
    # calculate the accuracy for each supervised source
        total_accuracy = [[] for _ in range(len(test_datasets))]
        for seed in [2022, 20, 2222]:
            # predict and get the accuracy in each domain with a specific seed
            model = get_model_by_source_and_target(model, source_id, target_id, seed)

            # evalutate
            for i in range(len(test_datasets)):
                datasetloader = torch.utils.data.DataLoader(test_datasets[i], batch_size=512, collate_fn=data_collator)
                accuracy = evaluate(datasetloader, model)
                total_accuracy[i].append(accuracy * 100)

            # show the result of each seed
            print(f'The result in seed {seed} is:')
            for i in range(len(test_datasets)):
                print(f'Accuracy for domain {i} is {total_accuracy[i][-1]}')
            
        # show the average result
        for i in range(len(test_datasets)):
            assert len(total_accuracy[i]) == 3
            calculate_and_show_result(total_accuracy[i], f'original {source_id}-{target_id} perform in domain {i}')

# f.close()
