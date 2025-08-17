from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


if __name__ == "__main__":
    data_path = 'data'

    print('DataTransformation START')
    data_transformation = DataTransformation()
    train_path, test_path = data_transformation.initiate_data_transformation(data_path)
    print('DataTransformation END')

    model_trainer = ModelTrainer()
    print(model_trainer.train(train_path, test_path))
