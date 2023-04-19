import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset, DataLoader


def dataset_loader(training_dir, validation_dir, test_dir, input_windows_length, ram_threshold, chunk_size, batch_size,
                   shuffle=True):
    offset = int(0.5 * (input_windows_length - 1.0))
    # 构建训练加载器
    train_iter = Data_Generator(file_name=training_dir, offset=offset, chunk_size=chunk_size,
                                ram_threshold=ram_threshold, shuffle=shuffle)
    train_loader = DataLoader(train_iter, batch_size=batch_size, drop_last=False)
    # 构建验证加载器
    val_iter = Data_Generator(file_name=validation_dir, offset=offset, chunk_size=chunk_size,
                              ram_threshold=ram_threshold, shuffle=shuffle)
    val_loader = DataLoader(val_iter, batch_size=batch_size, drop_last=False)
    # 构建测试加载器
    test_iter = Data_Generator(file_name=test_dir, offset=offset, chunk_size=chunk_size,
                               ram_threshold=ram_threshold, shuffle=False)
    test_loader = DataLoader(test_iter, batch_size=batch_size, drop_last=False)  # 这个里面我们测试是不需要进行打乱顺序的

    return train_loader, val_loader, test_loader


class Data_Generator(IterableDataset):
    def __init__(self, file_name, offset, chunk_size, ram_threshold=5 * 10 ** 6, shuffle=True,
                 input_windows_length=599):
        self.__file_name = file_name
        self.__offset = offset
        self.__input_windows_length = input_windows_length
        self.__chunk_size = chunk_size
        self.__ram_threshold = ram_threshold
        self.__shuffle = shuffle
        self.total_size = 0

    def check_chunking_size(self):
        # 创建csv数据迭代器计算大小
        chunks = pd.read_csv(self.__file_name, header=None, iterator=True)
        loop = True
        while loop:
            try:
                chunk = chunks.get_chunk(self.__chunk_size)
                self.total_size += chunk.shape[0]
            except StopIteration:
                loop = False

        print('\n' + '这个数据集合加载自', self.__file_name, '一共包含了', self.total_size, '行')
        del chunks
        if self.total_size > self.__ram_threshold:
            # 重要的事情说三遍
            print('warring：这次加载的数据集合过大我们将进行我们分块加载到内存里面，请注意')
            print('warring：这次加载的数据集合过大我们将进行我们分块加载到内存里面，请注意')
            print('warring：这次加载的数据集合过大我们将进行我们分块加载到内存里面，请注意')

        return self.total_size

    def __iter__(self):
        # 定义处理工作核心数量，推荐默认，多线程我还没有实现
        worker_info = torch.utils.data.get_worker_info()

        # 得到加载数据集合的行数
        if self.total_size == 0:
            self.check_chunking_size()

        # 创建数据处理迭代器交给我们的Dataloader
        skip_idx = np.arange(self.total_size / self.__chunk_size)
        # 打乱每块顺序
        if self.__shuffle:
            np.random.shuffle(skip_idx)

        if worker_info is None:
            for i in skip_idx:
                # 分情况加载数据集合
                if self.total_size > self.__ram_threshold:
                    data = pd.read_csv(self.__file_name, nrows=self.__chunk_size, skiprows=int(i) * self.__chunk_size,
                                       header=None, low_memory=False)
                else:
                    data = pd.read_csv(self.__file_name, header=None)
                # 进行转化，并且类型也转化成float32
                np_array = np.array(data, dtype=np.float32)
                # 提取输入输出参数
                inputs, outputs = np_array[:, 0], np_array[:, 1]
                maximum_batch_size = inputs.size - 2 * self.__offset
                # 将每次生成的窗口数据进行一次打乱

                indices = np.arange(maximum_batch_size)
                if self.__shuffle:
                    np.random.shuffle(indices)

                # 窗口生成器
                for start_index in range(0, maximum_batch_size):
                    start_index = indices[start_index]
                    input_data = inputs[start_index: start_index + self.__input_windows_length]
                    output_data = outputs[start_index + self.__offset].reshape(-1, 1)
                    # 返回的数据形式是 (599,) , (1, 1)
                    yield input_data, output_data
        else:
            assert False, "不要设置多线程，这个多线程加速我还没有办法编出来，而且，似乎在GPU上面设置多线程会导致一些问题"


class TestSlidingWindowGenerator(IterableDataset):
    def __init__(self, number_of_windows, inputs, targets, offset):
        self.__number_of_windows = number_of_windows
        self.__offset = offset
        self.__inputs = inputs
        self.__targets = targets
        self.total_size = len(inputs)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        np_array = np.array(self.__inputs, dtype=np.float32)
        inputs, outputs = np_array[:, 0], np_array[:, 1]
        maximum_batch_size = inputs.size - 2 * self.__offset

        if self.__batch_size < 0:
            self.__batch_size = maximum_batch_size

        if worker_info is None:
            for start_index in range(0, maximum_batch_size):
                input_data = inputs[start_index: start_index + self.__input_windows_length]
                output_data = outputs[start_index + self.__offset].reshape(-1, 1)
                yield input_data, output_data
        else:
            assert False, "不要设置多线程，这个多线程加速我还没有办法编出来，而且，似乎在GPU上面设置多线程会导致一些问题"
