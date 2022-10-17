from datasets import load_dataset, Dataset

GENRE = ['slate', 'telephone', 'government', 'travel', 'fiction']

def load_data(tokenizer=None, split='train'):
    def handle_tokenizer(e):
        premise_output = tokenizer(e['premise'], truncation=True, padding='max_length', max_length=32)
        hypothesis_output = tokenizer(e['hypothesis'], truncation=True, padding='max_length', max_length=32)
        for i in range(len(premise_output["input_ids"])):
            premise_output["input_ids"][i].extend(hypothesis_output["input_ids"][i])
            premise_output["attention_mask"][i].extend(hypothesis_output["attention_mask"][i])
        return premise_output

    dataset = load_dataset("multi_nli", split=split).remove_columns(
        ['promptID', 'pairID', 'premise_binary_parse', 'premise_parse', 
        'hypothesis_binary_parse', 'hypothesis_parse']
    )
    dataset = dataset.to_pandas()
    datasets = []
    for genre in GENRE:
        temp_dataset = Dataset.from_pandas(dataset[dataset['genre'].isin([genre])])
        if tokenizer is not None:
            temp_dataset = temp_dataset.map(handle_tokenizer, batched=True)
            temp_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
        datasets.append(temp_dataset)

    return datasets


def load_prefix_data(tokenizer=None, prefix='', split='train'):

    def handle_tokenizer(e):
        premise_output = tokenizer(e['premise'], truncation=True, padding='max_length', max_length=32 + len(tokenizer(prefix)["input_ids"]))
        hypothesis_output = tokenizer(e['hypothesis'], truncation=True, padding='max_length', max_length=32)
        for i in range(len(premise_output["input_ids"])):
            premise_output["input_ids"][i].extend(hypothesis_output["input_ids"][i])
            premise_output["attention_mask"][i].extend(hypothesis_output["attention_mask"][i])
        return premise_output

    dataset = load_dataset("multi_nli", split=split).remove_columns(
        ['promptID', 'pairID', 'premise_binary_parse', 'premise_parse', 
        'hypothesis_binary_parse', 'hypothesis_parse']
    )
    dataset = dataset.to_pandas()
    datasets = []

    for genre in GENRE:
        temp_dataset = Dataset.from_pandas(dataset[dataset['genre'].isin([genre])])
        if tokenizer is not None:
            temp_dataset = temp_dataset.map(lambda e: {'premise': prefix + e['premise']})
            temp_dataset = temp_dataset.map(handle_tokenizer, batched=True)
            temp_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
        datasets.append(temp_dataset)

    return datasets


def split_dataset(dataset):
    dataset = dataset.train_test_split(test_size=1/9, shuffle=False)
    return dataset['train'], dataset['test']
