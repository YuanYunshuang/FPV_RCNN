# Code source: https://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
import numpy as np

class StatsRecorder:
    def __init__(self, data=None):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        if data is not None:
            assert len(data.shape) >= 2
            self.mean = data.mean(axis=0)
            self.std  = data.std(axis=0)
            self.nobservations = data.shape[0]
            self.ndimensions   = data.shape[1]
        else:
            self.nobservations = 0

    def update(self, data):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        if self.nobservations == 0:
            self.__init__(data)
        else:
            assert len(data.shape) >= 2
            if data.shape[1] != self.ndimensions:
                raise ValueError("Data dims don't match prev observations.")

            newmean = data.mean(axis=0)
            newstd  = data.std(axis=0)

            m = self.nobservations * 1.0
            n = data.shape[0]

            tmp = self.mean

            self.mean = m/(m+n)*tmp + n/(m+n)*newmean
            self.std  = m/(m+n)*self.std**2 + n/(m+n)*newstd**2 +\
                        m*n/(m+n)**2 * (tmp - newmean)**2
            self.std  = np.sqrt(self.std)

            self.nobservations += n

# from tqdm import tqdm
# from datasets.comap import CoMapDataset
# stats = StatsRecorder()
# for i, data in tqdm(enumerate(train_dataloader)):
#     targets = data['regression_map'][0].transpose((2, 1, 0))
#     targets = targets[np.where(targets.sum(axis=-1))]
#     stats.update(targets)
#     if i > 200:
#         break
# print("Input Means: [{:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}]".format(*stats.mean))
# print("Input  Stds: [{:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}]".format(*stats.std))
if __name__=="__main__":
    from tqdm import tqdm
    from glob import glob
    from pathlib import Path
    root = '/media/hdd/ophelia/koko/data/synthdata_20veh_60m'
    path_ego = Path(root) / "cloud_ego"
    test_split = ['943', '1148', '753', '599', '53', '905', '245', '421', '509']
    train_val_split = ['829', '965', '224', '685', '924', '334', '1175', '139',
                            '1070', '1050', '1162', '1260']
    list_train_val = []
    list_test = []
    for filename in path_ego.glob("*.bin"):
        if filename.name.split("_")[0] in train_val_split:
            list_train_val.append(filename.name[:-4])
        else:
            list_test.append(filename.name[:-4])

    with open(root + "/train_val.txt", "w") as fa:
        for line in list_train_val:
            fa.writelines(line + "\n")
    with open(root + "/test.txt", "w") as fb:
        for line in list_test:
            fb.writelines(line + "\n")


    path = '/media/hdd/ophelia/koko/data/synthdata_20veh_60m/label_box'
    stats = StatsRecorder()
    for f in tqdm(list_train_val):
        file = path + '/' + f + '.txt'
        gt_boxes = np.loadtxt(file, dtype=str)[:, [2, 3, 4, 8, 9, 10, 7]].astype(np.float)
        stats.update(gt_boxes)

    print("Input Means: [{:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}]".format(*stats.mean))
    print("Input  Stds: [{:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}]".format(*stats.std))


    # Means: [7.71307, -24.17974, -0.82831, 4.40963, 1.98210, 1.63674, -0.00466]
    # Stds: [77.41920, 93.95169, 1.98245, 0.83028, 0.21313, 0.25730, 1.76150]

    # stats = StatsRecorder()
    # for i, data in tqdm(enumerate(train_dataloader)):
    #     targets = data['regression_map'][0].transpose((2, 1, 0))
    #     targets = targets[np.where(targets.sum(axis=-1))]
    #     stats.update(targets)
    #     if i > 200:
    #         break
    # print("Input Means: [{:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}]".format(*stats.mean))
    # print("Input  Stds: [{:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}]".format(*stats.std))