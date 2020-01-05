import json, codecs

class ReshuffleHelper:
    def __init__(self, es_patience=15, thresholds=[0.8, 1.3, 1.7]):
        self.es_patience = es_patience
        self.thresholds = thresholds
        self.flags = [False for _ in range(len(thresholds))]

    def check(self, epoch):
        epoch += 1
        for (index, flag) in enumerate(self.flags):
            if not flag:
                if epoch == int(self.es_patience * self.thresholds[index]):
                    self.flags[index] = True
                    if index < len(self.thresholds) - 1:
                        self.thresholds[index + 1] += self.thresholds[index]
                    print("Reshuffling...\n")
                    return True
                else:
                    return False

        return False

    def save_ckpt(self, path):
        ckpt_record = {
            'flags': self.flags
        }
        ckpt_record = json.dumps(ckpt_record, indent=4)
        with codecs.open(path + '/rs_helper.json', 'w', 'utf-8') as outfile:
            outfile.write(ckpt_record)

    def load_ckpt(self, path):
        with codecs.open(path + '/rs_helper.json', encoding='utf-8') as json_file:
            ckpt_record = json.load(json_file)
            self.flags = ckpt_record['flags']
