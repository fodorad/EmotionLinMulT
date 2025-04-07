import json
from prettytable import PrettyTable


def create_metrics_table(json_paths, task_name):
    # Create PrettyTable object with specified columns
    table = PrettyTable()
    table.field_names = ["Modality", "Feature", "Task", "ACC", "P", "R", "F1"]
    
    # Set alignment for all columns
    table.align = "l"
    
    # Process each JSON file
    for path in json_paths:
        with open(path, 'r') as file:
            data = json.load(file)
            
        modality = ''
        modality += 'A' if 'egemaps' in path or 'wavlm' in path else ''
        modality += 'V' if 'opengraphau' in path or 'fabnet' in path or 'clip' in path else ''

        feature = '+'.join([f for f in ['egemaps', 'wavlm', 'opengraphau', 'fabnet', 'clip'] if f in path])

        task = 'EC' if 'emotion-class' in path else 'EI'
        task = 'EC+EI' if 'mt' in path else task

        # Extract metrics (taking second element of each tuple)
        metrics = data["test"][task_name]
        acc = metrics["ACC"][0][1]
        p = metrics["P"][0][1]
        r = metrics["R"][0][1]
        f1 = metrics["F1"][0][1]
        
        # Add row to table
        table.add_row([modality, feature, task, acc, p, r, f1])
    
    return table



def create_gap_vs_attentionpool_table(json_paths, task_name):
    # Create PrettyTable object with specified columns
    table = PrettyTable()
    table.field_names = ["Modality", "Feature", "Task", "Module", "F1"]
    
    # Set alignment for all columns
    table.align = "l"
    
    # Process each JSON file
    for index, path in enumerate(json_paths):
        with open(path, 'r') as file:
            data = json.load(file)
            
        modality = ''
        modality += 'A' if 'egemaps' in path or 'wavlm' in path else ''
        modality += 'V' if 'opengraphau' in path or 'fabnet' in path or 'clip' in path else ''

        feature = '+'.join([f for f in ['egemaps', 'wavlm', 'opengraphau', 'fabnet', 'clip'] if f in path])

        task = 'EC' if 'emotion-class' in path else 'EI'
        task = 'EC+EI' if 'mt' in path else task

        module = 'GAP' if (index % 2 == 0) else 'AP'

        # Extract metrics (taking second element of each tuple)
        metrics = data["test"][task_name]
        acc = metrics["ACC"][0][1]
        p = metrics["P"][0][1]
        r = metrics["R"][0][1]
        f1 = metrics["F1"][0][1]
        
        # Add row to table
        table.add_row([modality, feature, task, module, f1])
    
    return table


def unimodal_ec():
    json_paths = [
        'results/unimodal/ravdess_emotion-class_egemaps-lld/20250104-164510/test/history_test.json',
        'results/unimodal/ravdess_emotion-class_wavlm-baseplus/20250104-173001/test/history_test.json',
        'results/unimodal/ravdess_emotion-class_egemaps-wavlm/20250106-153138/test/history_test.json',
        'results/unimodal/ravdess_emotion-class_opengraphau/20250104-181201/test/history_test.json',
        'results/unimodal/ravdess_emotion-class_fabnet/20250104-185422/test/history_test.json',
        'results/unimodal/ravdess_emotion-class_clip/20250104-163758/test/history_test.json',
        'results/unimodal/ravdess_emotion-class_opengraphau-fabnet-clip/20250106-162942/test/history_test.json',
    ]
    table = create_metrics_table(json_paths, "emotion_class")
    print(table)


def unimodal_ei():
    json_paths = [
        'results/unimodal/ravdess_emotion-intensity_egemaps-lld/20250105-121232/test/history_test.json',
        'results/unimodal/ravdess_emotion-intensity_wavlm-baseplus/20250105-124405/test/history_test.json',
        'results/unimodal/ravdess_emotion-intensity_egemaps-wavlm/20250107-001531/test/history_test.json',
        'results/unimodal/ravdess_emotion-intensity_opengraphau/20250105-131254/test/history_test.json',
        'results/unimodal/ravdess_emotion-intensity_fabnet/20250105-133748/test/history_test.json',
        'results/unimodal/ravdess_emotion-intensity_clip/20250105-140220/test/history_test.json',
        'results/unimodal/ravdess_emotion-intensity_opengraphau-fabnet-clip/20250107-004358/test/history_test.json'
    ]
    table = create_metrics_table(json_paths, "emotion_intensity")
    print(table)


def gap_vs_attentionpool():
    json_paths = [
        'results/unimodal/ravdess_emotion-class_egemaps-lld/20250103-174758/test/history_test.json',
        'results/unimodal/ravdess_emotion-class_egemaps-lld/20250104-164510/test/history_test.json',
        'results/unimodal/ravdess_emotion-class_wavlm-baseplus/20250103-162849/test/history_test.json',
        'results/unimodal/ravdess_emotion-class_wavlm-baseplus/20250104-173001/test/history_test.json',
        'results/unimodal/ravdess_emotion-class_opengraphau/20250103-172622/test/history_test.json',
        'results/unimodal/ravdess_emotion-class_opengraphau/20250104-181201/test/history_test.json',
        'results/unimodal/ravdess_emotion-class_fabnet/20250103-171856/test/history_test.json',
        'results/unimodal/ravdess_emotion-class_fabnet/20250104-185422/test/history_test.json',
        'results/unimodal/ravdess_emotion-class_clip/20250103-172842/test/history_test.json',
        'results/unimodal/ravdess_emotion-class_clip/20250104-163758/test/history_test.json',
    ]
    table = create_gap_vs_attentionpool_table(json_paths, "emotion_class")
    print(table)


def multimodal_ec():
    json_paths = [
        'results/multimodal/ravdess_emotion-class_egemaps-opengraphau/20250106-163838/test/history_test.json',
        'results/multimodal/ravdess_emotion-class_wavlm-clip/20250106-172535/test/history_test.json',
        'results/multimodal/ravdess_emotion-class_egemaps-wavlm-opengraphau-fabnet-clip/20250107-002044/test/history_test.json',
    ]
    table = create_metrics_table(json_paths, "emotion_class")
    print(table)


def multimodal_ei():
    json_paths = [
        'results/multimodal/ravdess_emotion-intensity_egemaps-opengraphau/20250107-002450/test/history_test.json',
        'results/multimodal/ravdess_emotion-intensity_wavlm-clip/20250107-010959/test/history_test.json',
        'results/multimodal/ravdess_emotion-intensity_egemaps-wavlm-opengraphau-fabnet-clip/20250107-015616/test/history_test.json',
    ]
    table = create_metrics_table(json_paths, "emotion_intensity")
    print(table)


def multimodal_mt():
    json_paths = [
        'results/multimodal/ravdess_mt_wavlm-clip/20250107-185011/test/history_test.json',
        'results/multimodal/ravdess_mt-shake_wavlm-clip/20250107-185243/test/history_test.json',
        'results/multimodal/ravdess_mt-mms_wavlm-clip/20250107-192732/test/history_test.json',
        'results/multimodal/ravdess_mt-mms-shake_wavlm-clip/20250107-203728/test/history_test.json'
    ]
    table = create_metrics_table(json_paths, "emotion_class")
    print(table)
    table = create_metrics_table(json_paths, "emotion_intensity")
    print(table)

multimodal_mt()