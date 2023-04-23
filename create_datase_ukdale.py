from lib_tool import data_args
import time
from config.ukdale_params_appliance import ukdale_params_appliance
import pandas as pd

# 定义源文件路径
DATA_DIRECTORY = 'C:/Users/liuhai/Desktop/dataset/UKDALD'  # with last 和前面的格式进行一个统一吧，我在后面加上/就行了
# 定义保存文件路径
SAVE_PATH = '../data/prepare_datasets/ukdale/'
# 定义设备 一共是四个可以训练的设备
APPLIANCE_NAME = 'washingmachine'
# APPLIANCE_NAME = ['washingmachine', 'fridge', 'dishwasher', 'microwave']

start_time = time.time()
# 创建命令行
args = data_args.get_arguments(DATA_DIRECTORY=DATA_DIRECTORY, APPLIANCE_NAME=APPLIANCE_NAME, SAVE_PATH=SAVE_PATH, transfer=True, house_num=1)


def load_dataframe(directory, building, channel, col_names=['time', 'data'], nrows=None):
    df = pd.read_table(
        directory + '/' + 'house_' + str(building) + '/' + 'channel_' + str(channel) + '.dat',
        sep="\s+",
        nrows=nrows,
        usecols=[0, 1],
        names=col_names,
        dtype={'time': str},
    )
    return df


def main():
    # ukdale第一家的用户的数据是很大的，所以实在第一家训练，第二家测试
    sample_seconds = 8
    training_building_percent = 95  # 这特么又是两个不等的东西，95 + 13 != 100,这两个应该是我们定义出来的一个训练集合的百分比以及我们验证集合的百分比，13%是95%里面的
    validation_percent = 13
    test_percent = 13
    nrows = None
    print('开始对' + 'UKDALE_' + args.appliance_name + '进行处理')
    train = pd.DataFrame(columns=['aggregate', args.appliance_name])
    # 如果进行迁移学习的话我们进行我们的信息的打印，不进行迁移学习的话我们在进行房子的选择，默认是是房子一号
    for h in ukdale_params_appliance[args.appliance_name]['houses']:
        if not args.transfer:
            if not h==args.house_num:
                continue
        print(args.data_dir + '/' + 'house_' + str(h) + '/' + 'channel_' + str(
            ukdale_params_appliance[args.appliance_name]['channels'][ukdale_params_appliance[args.appliance_name]['houses'].index(h)]) + '.dat')
        # 加载总功率的数据集合
        mains_df = load_dataframe(args.data_dir, h, 1)
        # 加载设备功率集合
        app_df = load_dataframe(args.data_dir,
                                h,
                                ukdale_params_appliance[args.appliance_name]['channels'][ukdale_params_appliance[args.appliance_name]['houses'].index(h)],
                                col_names=['time', args.appliance_name])
        # 但是这也是可以转化成ms的，不过需要的时间比较长  # 这个地方又是转化成s了，前面转化成ms不知道是什么鬼操作，那个ms在哪个地方
        mains_df['time'] = pd.to_datetime(mains_df['time'], unit='s')
        # 刚刚把时间戳搞成我们的这个样子，下面马上就是把他设置成0这种，也就是把0数字这些当成index，然后就可以给我们的data改名字了，因为我们读取的数据时候我们没有吧名字这个参数传过去的
        mains_df.set_index('time', inplace=True)
        mains_df.columns = ['aggregate']
        mains_df.reset_index(inplace=True)
        app_df['time'] = pd.to_datetime(app_df['time'], unit='s')
        # 电源和电器的时间戳不一样，我们需要把它们对齐:1. 连接总功率和设备数据; 2. 插入缺失的值;
        mains_df.set_index('time', inplace=True)
        app_df.set_index('time', inplace=True)

        df_align = mains_df.join(app_df, how='outer').resample(str(sample_seconds) + 'S').mean().fillna(method='backfill', limit=1)
        df_align = df_align.dropna()
        df_align.reset_index(inplace=True)
        del mains_df, app_df, df_align['time']

        mean = ukdale_params_appliance[args.appliance_name]['mean']  # 标准化我们的数据集合
        std = ukdale_params_appliance[args.appliance_name]['std']
        df_align['aggregate'] = (df_align['aggregate'] - args.aggregate_mean) / args.aggregate_std  # 这里面开始是我们标准化数据集合
        df_align[args.appliance_name] = (df_align[args.appliance_name] - mean) / std

        if args.transfer:
            if h == ukdale_params_appliance[args.appliance_name]['test_build']:
                df_align.to_csv(args.save_path + args.appliance_name + '_transfer_test_.csv', mode='a', index=False, header=False)
                print("测试数据集合大小是 {:.6f} M rows.".format(len(df_align) / 10 ** 6))
                continue

        train = train.append(df_align, ignore_index=True)
        del df_align

    # 裁剪数据集，如果不为0，我们就删除他最后面的部分，这个是直接丢弃了95%的数据
    if training_building_percent != 0:
        train.drop(train.index[-int((len(train) / 100) * training_building_percent):], inplace=True)

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

    print("训练数据集合大小 {:.6f} M rows.".format(len(train) / 10 ** 6))
    print("验证集合大小 {:.6f} M rows.".format(len(val) / 10 ** 6))
    del train, val
    print("\n 文件保存在以下目录（展示的是相对文件路径）" + args.save_path)
    print("总共用时间是: {:.2f} sec.".format((time.time() - start_time)))


if __name__ == '__main__':
    main()




