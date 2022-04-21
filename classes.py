from abc import ABC, abstractmethod
from typing import Optional, Union, Iterable
from collections.abc import Iterator
import pandas as pd

class Predictor:
    def __init__(self,source_model):
        self.source_model = source_model
    def get_prediction(self,prediction_data):
        estimated_price = self.source_model.predict(prediction_data)

class Training_data(data):
    def __init__(
      self,
      training_data
     ):
     self.training_data : list[House_data] = []

class Testing_data(data):
    def __init__(
      self,
      testing_data
     ):
     self.testing_data : list[House_data] = []

class Prediction_data(House_data):
        
class data(ABC):
    @abstractmethod
    def load_data (self,data_list,data_source):
        for i,row  in data_source.iterrows():
            data = House_data(
                    condition = row["condition"]
                    property_type = row["property_type"]
                    neighborhood = row["neighborhood"]
                    estrato = row["estrato"]
                    area = row["area"]
                    bedrooms = row["bedrooms"]
                    bathrooms = row["bathrooms"]
                    garages = row["garages"]
                    price = row["price"]
            )
            data_list.append(House_data)
class House_data():
  def __init__(
    self,
    condition : str,
    property_type : str,
    neighborhood : str,
    estrato : int,
    area : int,
    bedrooms : int,
    bathrooms : int,
    garages : int,
    price: Optional[int]
   ):
    self.condition = condition
    self.property_type = property_type
    self.neighborhood = neighborhood
    self.estrato = estrato
    self.area = area
    self.bedrooms = bedrooms
    self.bathrooms = bathrooms
    self.garages = garages
    self.price: Optional[int]

class df_data_source():
    def __init__(self,data_source):
        self.data_source = data_source
