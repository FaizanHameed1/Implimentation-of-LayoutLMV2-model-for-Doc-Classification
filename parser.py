import yaml

with open("config.yml", "r") as data:
    try:
        config = yaml.safe_load(data)
    except yaml.YAMLError as exc:
        print(exc)

train_config = config["train_config"]
test_config=config["test_config"]