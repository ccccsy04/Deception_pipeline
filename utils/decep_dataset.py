import json
import random

class DeceptionBenchDataset:
    def __init__(self, file_path, types=None, num_samples=None, shuffle=False, control=False, auto_analysis_path=None):
        """
        file_path: JSONL 文件路径
        types: 指定 type（str 或 list），如 'Honesty_Evasion_Under_Pressure' 或 ['A', 'B']
        num_samples: 加载样本数，None 表示全部
        shuffle: 是否打乱顺序
        control: 是否为对照组，若为True则需指定auto_analysis_path
        auto_analysis_path: auto_analysis.jsonl路径，对照组用
        """
        self.data = []
        control_ids = None
        if control and auto_analysis_path is not None and types is not None:
            control_ids = set()
            with open(auto_analysis_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except Exception:
                        continue
                    type_list = [types] if isinstance(types, str) else types
                    if item.get('type') in type_list and item.get('final_judgment') == 'non-deceptive':
                        control_ids.add(str(item.get('id')))
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                if types is not None:
                    if isinstance(types, str):
                        types = [types]
                    if item.get('type') not in types:
                        continue
                if control_ids is not None:
                    if str(item.get('id')) not in control_ids:
                        continue
                self.data.append(item)
        if shuffle:
            random.shuffle(self.data)
        if num_samples is not None:
            self.data = self.data[:num_samples]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]