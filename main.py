from data_manager import DataManager
from utils.toolkits import seed_set, load_json
import utils.toolkits as toolkits

seed_set()

def main():
    # 配置文件选择开关
    CONFIG_TYPE = "cifar_in21k_ncmloss"
    # CONFIG_TYPE = "cifar_in21k_mine11"

    CONFIG_TYPE = "cub_in21k_ncmloss"

    config_map = {
        "cifar_in21k_ncmloss": "exps/cifar/cifar_in21k_ncmloss.json",
        "cifar_in21k_mine11": "exps/cifar/cifar_in21k_mine11.json",
        "cub_in21k_ncmloss": "exps/cub/cub_in21k_ncmloss.json"
    }

    # 指定配置文件路径
    config_path = config_map[CONFIG_TYPE]

    # 加载配置参数
    args = load_json(config_path)  # 直接获取字典格式参数

    # 初始化数据管理器
    if True:
        data_manager = DataManager(
            dataset_name=args["dataset"],
            shuffle=True,
            seed=args["seed"],
            init_cls=args["init_cls"],
            increment=args["increment"],
            args=args
        )

    # 初始化模型
    model = toolkits.get_model(model_name=args["model_name"], args=args)

    # 执行增量学习任务
    for task in range(len(data_manager._increments)):
        print(f'Here Comes Task{task}', '*'*50)
        model.incremental_train(data_manager)
        model.eval_accuracy(words=f'{task}')
        model.after_task()

if __name__ == '__main__':
    main()