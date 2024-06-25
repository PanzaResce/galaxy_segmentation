from collections import Counter

def get_class_distribution(data_split):
    class_counts = Counter()
    for item in data_split.values():
        for region in item['regions']:
            class_name = region['region_attributes']['object_name']
            class_counts[class_name] += 1
    return class_counts