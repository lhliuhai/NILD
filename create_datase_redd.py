# 导入我们的工具库
import time
import pandas as pd
from config.redd_params_appliance import redd_params_appliance
from lib_tool import data_args

# 定义源文件路径 这里面为了和服务器相比较我们进行了一个分离，也就是源文件我们将其放到其他位置了，这个里面自己安排位置
DATA_DIRECTORY = 'F:/dataset/REDD/low_freq'  # without last /
# 定义保存文件路径
SAVE_PATH = '../data/prepare_datasets/'
# 定义设备 一共是四个可以训练的设备
APPLIANCE_NAME = 'washingmachine'
# APPLIANCE_NAME = ['washingmachine', 'fridge', 'dishwasher', 'microwave']

start_time = time.time()
# 创建命令行
args = data_args.get_arguments(DATA_DIRECTORY=DATA_DIRECTORY, APPLIANCE_NAME=APPLIANCE_NAME, SAVE_PATH=SAVE_PATH, transfer=False, house_num=1)


def main():
    sample_seconds = 8   # 采样时间设置成8S
    validation_percent = 15     # 设置我们的样本的数据集中间验证数据集合的百分比
    test_percent = 15
    nrows = None  # 这是选择我们需要的列，默认None是选择全部的列
    print('开始对' + 'REDD_' + args.appliance_name + '进行处理')
    train = pd.DataFrame(columns=['aggregate', args.appliance_name])
    # 如果进行迁移学习的话我们进行我们的信息的打印，不进行迁移学习的话我们在进行房子的选择，默认是是房子一号
    for h in redd_params_appliance[args.appliance_name]['houses']:
        if not args.transfer:
            if not h==args.house_num:
                continue
        print(args.data_dir + '/house_' + str(h) + '/' + 'channel_' + str(
            redd_params_appliance[args.appliance_name]['channels'][redd_params_appliance[args.appliance_name]['houses'].index(h)]) + '.dat')

        # 加载总功率的数据集合
        mains1_df = pd.read_table(
            args.data_dir + '/' + 'house_' + str(h) + '/' + 'channel_' + str(1) + '.dat',
            sep="\s+",
            nrows=nrows,
            usecols=[0, 1],
            names=['time', 'mains1'],
            dtype={'time': str},    # 单独每一列设置索引
        )

        mains2_df = pd.read_table(
            args.data_dir + '/' + 'house_' + str(h) + '/' + 'channel_' + str(2) + '.dat',
            sep="\s+",
            nrows=nrows,
            usecols=[0, 1],
            names=['time', 'mains2'],
            dtype={'time': str},
            )

        app_df = pd.read_table(
            args.data_dir + '/' + 'house_' + str(h) + '/' + 'channel_' + str(
                redd_params_appliance[args.appliance_name]['channels'][redd_params_appliance[args.appliance_name]['houses'].index(h)]) + '.dat',
            sep="\s+",
            nrows=nrows,
            usecols=[0, 1],
            names=['time', args.appliance_name],
            dtype={'time': str},
            )

        # 前面我们将他的索引设置成str，将我们的时间转化到我们的标准的时间序列上面去
        mains1_df['time'] = pd.to_datetime(mains1_df['time'], unit='s')
        mains2_df['time'] = pd.to_datetime(mains2_df['time'], unit='s')
        # 将time设置成index
        mains1_df.set_index('time', inplace=True)
        mains2_df.set_index('time', inplace=True)
        # 将两个有功功率进行合并，采用outer就是外侧合并模式，得到我们的总的有功功率合集
        mains_df = mains1_df.join(mains2_df, how='outer')
        # 再次定义一个新的轴作为总的有功功率，就是把两个有功的和加到一个新列上面，这个列是'aggregate'，接下来我们就要删除我们前面的多余的那些东西
        mains_df['aggregate'] = mains_df.iloc[:].sum(axis=1)
        mains_df.reset_index(inplace=True)
        # 删除多余的数据，节约内存
        del mains_df['mains1'], mains_df['mains2']
        # 主要是我们的前面的是将time设置成str这样的格式，将我们的时间转化到我们的标准的时间序列上面去
        app_df['time'] = pd.to_datetime(app_df['time'], unit='s')
        # 电源和设备的时间戳不一样，我们需要校准它们，即是以下三个步骤：1. 加入总功率和设备数据帧 2.插入缺失的值 3.将我们time设置为索引，然后进行合并操作
        mains_df.set_index('time', inplace=True)
        app_df.set_index('time', inplace=True)
        # 将数据总数据和设备的合并，并且进行8s的采样，在将8S里面的所有数据求取均值填充到8S上面，但是这个里面会多出来很多的NaN，在使用backfill的方法进插值，但是插值的数目限制到1，然后将NAn全部丢弃掉，limit参数：限制填充个数
        df_align = mains_df.join(app_df, how='outer').resample(str(sample_seconds) + 'S').mean().fillna(method='backfill', limit=1)
        df_align = df_align.dropna()  # 删除我们多余的因为我们进行8s采样多出来NAN的东西，一般不是很多1561660/197063是很接近8倍的，之所以少了就是我们采用了填充一部分，但是好在个数限制在1，所以是7.9246这个比值
        df_align.reset_index(inplace=True)
        # 删除多余的变量
        del mains1_df, mains2_df, mains_df, app_df, df_align['time']
        # 标准化数据集
        mean = redd_params_appliance[args.appliance_name]['mean']  # 提取设备的我们设置的均值和方差进行归一化
        std = redd_params_appliance[args.appliance_name]['std']
        df_align['aggregate'] = (df_align['aggregate'] - args.aggregate_mean) / args.aggregate_std  # 对我们的总的有功功率进行归一化
        df_align[args.appliance_name] = (df_align[args.appliance_name] - mean) / std  # 对我们的设备有功功率进行归一化
        # 开始保存数据了
        # 当我们我们迭代到我们设置的那个作为测试集合的样本的时候进行一次保存
        if args.transfer:
            if h == redd_params_appliance[args.appliance_name]['test_build']:
                df_align.to_csv(args.save_path + args.appliance_name + '_transfer_test_.csv', mode='a', index=False, header=False)  # 测试集合 csv 的保存
                print("测试数据的大小是 {:.6f} M rows.".format(len(df_align) / 10 ** 6))
                continue
        # 上面判断的是如果不是我们迁移学习里面的测试集合我们就将其转移到train，后面还需要在建立一个不是迁移学习的训练集合
        train = train.append(df_align, ignore_index=True)
        del df_align

    train_len = len(train)

    if not args.transfer:
        test_len = int((train_len / 100) * test_percent)
        test = train.tail(test_len)
        test.reset_index(drop=True, inplace=True)
        train.drop(train.index[-test_len:], inplace=True)
        test.to_csv(args.save_path + args.appliance_name + '_test_.csv', mode='a', index=False, header=False)

    val_len = int((train_len / 100) * validation_percent)
    val = train.tail(val_len)
    val.reset_index(drop=True, inplace=True)
    train.drop(train.index[-val_len:], inplace=True)
    val.to_csv(args.save_path + args.appliance_name + ('_transfer_validation_.csv' if args.transfer else '_validation_.csv'), mode='a', index=False, header=False)
    train.to_csv(args.save_path + args.appliance_name + ('_transfer_training_.csv' if args.transfer else '_training_.csv'), mode='a', index=False, header=False)

    if not args.transfer:
        print("测试集大小 {:.6f} M rows.".format(len(test) / 10 ** 6))

    print("训练集大小 {:.6f} M rows.".format(len(train) / 10 ** 6))
    print("验证集大小 {:.6f} M rows.".format(len(val) / 10 ** 6))
    del train, val
    print("\n 文件保存在以下目录（展示的是相对文件路径）" + args.save_path)
    print("全部处理时间是: {:.2f} sec.".format((time.time() - start_time)))


if __name__ == '__main__':
    main()
