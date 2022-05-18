# Importing libraries
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from hyperopt import hp
from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials
from sklearn.model_selection import cross_val_score
import streamlit as st

# This class contains the data (a pandas df) as an attribute
class df_data_source():
    def __init__(self,source,source_type,train_size_,test_size_):
        self.train_size_ = train_size_
        self.test_size_ = test_size_
        self.source_type = source_type
        self.source = source
        if(self.source_type == 'url'):
            self.get_data_by_url(source)
        elif(self.source_type == 'path'):
            self.get_data_by_path(source)
        elif(self.source_type == 'pass'):
            self.get_data(source)
        self.split_data(self.train_size_,self.test_size_)
    # Method to split data train/test
    # This method also reshapes the data in order to be used by the model
    def split_data(self,train_size_,test_size_):
        self.train_size_ = train_size_
        self.test_size_ = test_size_
        self.price_subset = self.data_source.iloc[:, -1].values
        self.non_price_subset = self.data_source.iloc[:, :-1].values
        self.data_subset_train, self.data_subset_test,self.price_subset_train, self.price_subset_test = train_test_split(self.non_price_subset,self.price_subset, test_size=self.test_size_, train_size=self.train_size_)
        self.price_subset_train = self.price_subset_train.reshape(-1, 1)
        self.price_subset_test = self.price_subset_test.reshape(-1, 1)
    # Method to delete the duplicates
    # A post is a duplicate if it shares the same post_id/post_link
    def delete_duplicates(self):
        self.data_source = df.data_source.drop_duplicates(subset="post_link")
    def delete_nan(self,column_wnan):
        self.column_wnan = column_wnan
        self.data_source = self.data_source[self.data_source[column_wnan].notna()]
        #sns.heatmap(self.data_source.corr())
    def delete_outliers(self):
        self.q1 = self.data_source.quantile(0.25)
        self.q3 = self.data_source.quantile(0.75)
        self.iqr = self.q3 - self.q1
        self.data_source = self.data_source[~((self.data_source < (self.q1 - 1.5 * self.iqr)) |self.data_source > (self.q3 + 1.5 * self.iqr)).any(axis=1)]
    def drop_column(self,column_name):
        self.column_name = column_name
        self.data_source.drop(self.column_name, axis=1, inplace=True)
    # The dataframe is pulled from github using an url
    def get_data_by_url(self,data_source_url):
        self.data_source_url = data_source_url
        self.data_source = pd.read_csv(self.data_source_url)
    # The dataframe is pulled from a specific path
    def get_data_by_path(self,data_source_path):
        self.data_source_path = data_source_path
        self.data_source = pd.read_csv(self.data_source_path)
    # The dataframe is assigned (=)
    def get_data(self,data_source):
        self.data_source = data_source
    def get_best(self):
        self.select_best = SelectKBest(score_func=f_regression, k=20)
        self.select_best.fit(self.non_price_subset, self.price_subset)
        self.non_price_subset_best = self.select_best.transform(self.non_price_subset)
        self.best_features = pd.DataFrame({'columns': self.non_price_subset.columns,
                                          'Kept': self.select_best.get_support()})
        return self.best_features

class borough_classifier():
    def make_column_1stone(self):
        self.sectors_column = self.data.data_source.pop('Borough')
        self.data.data_source.insert(0, 'Borough', self.sectors_column)

    def get_sectors(self):
        self.sectors_conditions = [self.data.data_source['neighborhood'].isin(self.list_riomar),
        self.data.data_source['neighborhood'].isin(self.list_nch),
        self.data.data_source['neighborhood'].isin(self.list_metr),
        self.data.data_source['neighborhood'].isin(self.list_surocc),
        self.data.data_source['neighborhood'].isin(self.list_suror),
        self.data.data_source['neighborhood'].isin(self.list_sol)]
        self.sectors_values = ['Riomar', 'Norte-Centro Histórico', 'Metropolitana', 'Sur Occidente', 'Sur Oriente',
                                   'Soledad']
        self.data.data_source['Borough'] = np.select(self.sectors_conditions, self.sectors_values)
        self.make_column_1stone()

    def __init__(self, data):
        self.data = data
        self.list_riomar = ['Adela de Char',
                                'Adelita de Char Etp 2',
                                'Adelita de Char Etp. 3',
                                'Adelita de Char Etp. 2',
                                'Altamira',
                                'Altos de Riomar',
                                'Altos del Limón',
                                'Altos del Prado',
                                'Andalucía',
                                'Buenavista',
                                'El Castillo I',
                                'El Golf',
                                'El Limoncito',
                                'El Poblado',
                                'Eduardo Santos',
                                'La Castellana',
                                'La Floresta',
                                'Las Flores',
                                'Las Tres Avemarías',
                                'Miramar',
                                'Paraíso',
                                'Riomar',
                                'San Marino',
                                'San Salvador',
                                'San Vicente',
                                'Santa Mónica',
                                'Siape',
                                'Solaire',
                                'Urbanización Eduardo santos la playa',
                                'Villa Campestre',
                                'Villa Carolina',
                                'Villa del Este',
                                'Villa Santos',
                                'Villamar',
                                'Pradomar',
                                'Puerto Colombia',
                                'Villas del Puerto']
        self.list_nch = [
                'América',
                'Barlovento',
                'Barranquillita',
                'Barrio Abajo',
                'Bellavista',
                'Betania',
                'Boston',
                'Campo Alegre',
                'Centro',
                'Ciudad Jardín',
                'Colombia',
                'El Boliche',
                'El Castillo',
                'El Porvenir',
                'El Prado',
                'El Recreo',
                'El Rosario',
                'El Tabor',
                'Granadillo',
                'Zona Industrial Vía 40',
                'La Bendición de Dios',
                'La Campiña',
                'La Concepción',
                'La Cumbre',
                'La Felicidad',
                'La Loma',
                'Las Colinas',
                'Las Delicias',
                'Las Mercedes',
                'Los Alpes',
                'Los Jobos',
                'Los Nogales',
                'Miramar',
                'Modelo',
                'Montecristo',
                'Nuevo Horizonte',
                'Paraíso',
                'Parque Rosado',
                'San Francisco',
                'Santa Ana',
                'Villa Country',
                'Villa Tarel',
                'Norte-Centro Histórico',
                'Alameda del Rio',
                'Barranquilla',
                'Villanueva']
        self.list_metr = [
                '7 de Abril',
                '7 de Abril.',
                '20 de Julio',
                'Buenos Aires',
                'Carrizal',
                'Cevillar',
                'Ciudadela 20 de Julio',
                'El Santuario',
                'Kennedy',
                'La Sierra',
                'La Sierrita',
                'La Victoria',
                'Las Américas',
                'Las Cayenas',
                'Las Gardenias',
                'Las Granjas',
                'Los Continentes',
                'Los Girasoles',
                'San José',
                'San Luis',
                'Santa María',
                'Santo Domingo de Guzmán',
                'Villa San Carlos',
                'Villa San Pedro I',
                'Villa San Pedro II',
                'Villa San Pedro I y II',
                'Villa Sevilla']
        self.list_surocc = [
                '7 de Agosto',
                'Bernando Hoyos',
                'Buena Esperanza',
                'California',
                'Caribe Verde',
                'Carlos Meisel',
                'Cevillar',
                'Chiquinquirá',
                'Ciudad Modesto',
                'Colina Campestre',
                'Los Olivos I y II',
                'Cordialidad',
                'Corrigimiento Juan Mina',
                'Cuchilla de Villate',
                'El Bosque',
                'El Carmen',
                'El Edén 2000',
                'El Golfo',
                'El Pueblo',
                'El Recreo',
                'El Romance',
                'El Rubí',
                'El Silencio',
                'El Valle',
                'Evaristo Sourdis',
                'Kalamary',
                'La Ceiba',
                'La Esmeralda',
                'La Florida',
                'La Gloria',
                'La Libertad',
                'La Manga',
                'La Paz',
                'La Pradera',
                'La Sierra',
                'Las Colinas',
                'Las Estrellas',
                'Las Malvinas',
                'Las Mercedes Sur',
                'Las Terrazas',
                'Lipaya',
                'Loma Fresca',
                'Los Andes',
                'Los Olivos I',
                'Los Olivos II',
                'Los Pinos',
                'Los Rosales',
                'Lucero',
                'Mequejo',
                'Olaya',
                'San Felipe',
                'San Isidro',
                'suroccidente',
                'Villas San Pablo',
                'El Por Fin']
        self.list_suror = [
                'Alfonso López',
                'Atlántico',
                'Bella Arena',
                'Boyacá',
                'Chiquinquirá',
                'El Campito',
                'El Ferry',
                'El Limón',
                'El Milagro',
                'José Antonio Galán',
                'La Alboraya',
                'La Chinita',
                'La Luz',
                'La Magdalena',
                'La Unión',
                'La Victoria',
                'Las Dunas',
                'Las Nieves',
                'Las Palmas',
                'Las Palmeras',
                'Los Laureles',
                'Los Trupillos',
                'Moderno',
                'Montes',
                'Pasadena',
                'Primero de Mayo',
                'Rebolo',
                'San José',
                'San Nicolás',
                'San Roque',
                'Santa Helena',
                'Simón Bolívar',
                'Tayrona',
                'Universal I y II',
                'Villa Blanca',
                'Villa del Carmen',
                'Zona Franca']
        self.list_sol = [
                'soledad',
                '12 de octubre',
                'El Parque',
                'Las Gaviotas',
                'Nuevo Triunfo',
                'Villa Angelita',
                'Villa Valentina',
                '13 de mayo',
                'El Pasito',
                'Las Margaritas',
                'Oriental',
                'Villa del Carmen',
                'Villa Éxito',
                '16 de julio',
                'El Río',
                'Las Moras',
                'Porvenir',
                'Villa del Rey',
                'Villa María',
                '20 de julio',
                'El Triunfo',
                'Las Nubes',
                'Prado Soledad',
                'Villa Estadio',
                'Villa Severa',
                '7 de agosto',
                'El Tucan',
                'Las Trinitarias',
                'Primero De Mayo',
                'Villa Estefanny',
                'Villa Viola',
                'Altos de Sevilla',
                'El Hipódromo',
                'Ferrocarril',
                'Puerta de Oro',
                'Villa Gladys',
                'Los Cocos',
                'Bella Murillo',
                'Juan Domínguez Romero',
                'Los Almendros',
                'Pumarejo',
                'Villa Karla',
                'Bonanza',
                'La Alianza',
                'Los Arrayanes',
                'Renacer',
                'Villa Katanga',
                'Portal De Las Moras'
                'Cabrera',
                'La Arboleda',
                'Los Balcanes',
                'Los Cedros',
                'Sal Si Puedes',
                'Nueva Esperanza',
                'Centenario',
                'La Central',
                'Los Cusules',
                'Salamanca',
                'Villa Merly',
                'La Esperanza',
                'Los Laureles',
                'Salcedo',
                'Villa Monaco',
                'Ciudad Paraíso',
                'Ciudad Camelot',
                'Ciudad Bolívar',
                'La Farruca',
                'Los Loteros',
                'San Antonio'
                'Ciudad Salitre',
                'La Fe',
                'Los Mangos',
                'San Vicente',
                'Villa Rosa',
                'Antonio Nariño',
                'Ciudadela Metropolitana',
                'Los Campanos',
                'Altos de las Villas',
                'Ríos de Agua Viva',
                'Villa Muvdi',
                'La Floresta',
                'Los Robles',
                'Santa Inés',
                'Villa Selene',
                'Portal de las Moras',
                'Costa Hermosa',
                'La Loma',
                'Manuela Beltran',
                'Soledad 2000',
                'Villa Sol',
                'Cruz de Mayo',
                'La María',
                'Moras Norte',
                'Tajamar',
                'Villa Soledad',
                'Don Bosco IV',
                'La Rivera',
                'Moras Occidente',
                'Terranova',
                'Villa Zambrano'
                'El Cachimbero',
                'Normandia',
                'Villa Adela',
                'Viña del Rey',
                'Villa de las Moras',
                'El Esfuerzo',
                'Las Candelarias',
                'Nueva Jerusalem',
                'Villa Anita',
                'Vista Hermosa',
                'El Ferrocarril',
                'Las Colonias',
                'Nuevo Horizonte',
                'Villa Aragón',
                'Zarabanda',
                'El Manantial',
                'Las Ferias',
                'Nuevo Milenio',
                'Parque Muvdi',
                'Ciudad Caribe',
                'Ciudadela Metropolitan',
                'Villa Cecilia']

class plotting():
    def __init__(self,data):
        self.data = data
    def plot_distr(self,x_column):
        self.x_column = x_column
        self.x_data = self.data[x_column]
        sns.displot(self.x_data)
    def plot_corr(self):
        sns.heatmap(self.data.corr())
    def plot_histo(self):
        self.y_data = self.data['Price']
        self.variables = ['estrato','property_type','neighborhood','Area','bedrooms','bathrooms','garages']
        for self.i in self.variables:
            self.x_column = self.data[self.i]
            sns.jointplot(x=self.x_column, y=self.y_data , data=self.data)

class oh_encoder(OneHotEncoder):
    def __init__(self,data):
        super(OneHotEncoder, self).__init__()
        self.data = data
        #self.data.drop (['post_title'], 1, inplace=True)
        #self.data.drop(['post_link'], 1, inplace=True)
        #self.data.drop(['city'], 1, inplace=True)
        self.handle_unknown = "ignore"
        self.categories = 'auto'
        self.sparse = False
        self.dtype = float
        self.drop = None
    def encode(self):
        self.columns_cat = self.data.select_dtypes(include=["object"]).columns
        self.encoded_data = self.fit_transform(self.data[self.columns_cat])
        self.encoded_data = pd.DataFrame(self.encoded_data, columns = self.get_encoded_columns())
        self.oh_data = self.encoded_data.join(self.data)
        self.oh_data = self.delete_non_encoded_columns(self.oh_data)
        return self.oh_data
    def get_encoded_columns(self):
        self.columns_cats_encoded = []
        for column in self.columns_cat :
            self.columns_cats_encoded += [f"{column[0]}_{cat}" for cat in list(self.data[column].unique())]
        return self.columns_cats_encoded
    def delete_non_encoded_columns(self,data_encoded):
        self.data_encoded = data_encoded
        for column in self.columns_cat :
            self.data_encoded.drop([column], 1, inplace=True)
        return self.data_encoded
class Predictor():
    def __init__(self,source_model,data,train_size_,test_size_):
        self.train_size_ = train_size_
        self.test_size_ = test_size_
        self.source_model = source_model
        self.data = data
        self.get_data()
        if(self.source_model == 'linear_regression'):
            self.linear_regression()
        if(self.source_model == 'decision_tree'):
            self.decision_tree()
        if(self.source_model == 'random_forest'):
            self.random_forest()
        if(self.source_model == 'gradient_boosting'):
            self.gradient_boosting()
        if(self.source_model == 'gradient_boosting'):
            self.gradient_boosting()
        if(self.source_model == 'xg_boost'):
            self.xg_boost()
    def get_data(self):
        #self.data.split_data(self.train_size_,self.test_size_)
        self.x_train = self.data.data_subset_train
        self.y_train = self.data.price_subset_train
        self.x_test = self.data.data_subset_test
        self.y_test = self.data.price_subset_test
        self.have_data = 1
    def get_pipeline(self,model):
        self.model = model
        self.pipe = Pipeline([('model', TransformedTargetRegressor(regressor=self.model,
                                                                  func=np.log,
                                                                  inverse_func=np.exp))])
    def linear_regression(self):
        self.lr = LinearRegression()
        self.linear_reg = TransformedTargetRegressor(regressor = self.lr, func = np.log1p, inverse_func = np.expm1)
        self.linear_reg.fit(self.x_train,self.y_train)
        print('Training score : ',self.linear_reg.score(self.x_train,self.y_train))
        print('Real score : ',self.linear_reg.score(self.x_test,self.y_test))
    def decision_tree(self):
        self.dec_tree = DecisionTreeRegressor(criterion="mse", random_state=0)
        self.get_pipeline(self.dec_tree)
        #self.model = TransformedTargetRegressor(regressor = self.model, func = np.log, inverse_func = np.exp)
    def random_forest(self):
        self.rand_forest = RandomForestRegressor(n_jobs=-1, random_state=0, bootstrap=True)
        self.get_pipeline(self.rand_forest)
        #self.model = TransformedTargetRegressor(regressor = self.model, func = np.log, inverse_func = np.exp)
    def gradient_boosting(self):
        self.grad_boost = GradientBoostingRegressor(random_state=0, max_features="sqrt")
        self.get_pipeline(self.grad_boost)
    def xg_boost(self):
        self.reg = xgb.XGBRegressor(objective="reg:squarederror")
    def fit_print_scr(self):
        self.reg.fit(self.x_train,self.y_train)
       #self.model.fit(self.x_train,self.y_train)
        print('Training score : ',self.reg.score(self.x_train,self.y_train))
        print('Real score : ',self.reg.score(self.x_test,self.y_test))
        self.preds = self.reg.predict(self.x_train)
        print('RMSE : ',np.sqrt(mean_squared_error(self.y_train, self.preds)))
        print('MAE : ',mean_absolute_error(self.y_train, self.preds))
    def get_model(self,best_model_params):
        self.best_model_params = best_model_params
        self.reg.fit(self.x_train, self.y_train)

class tuning():
    def __init__(self,predictor,scoring,params):
        self.predictor = predictor
        self.scoring = scoring
        self.params = params
        self.tune()
    def tune(self):
        self.predictor.reg = GridSearchCV(self.predictor.pipe, param_grid=self.params, scoring=self.scoring, cv=5)
        #self.model = GridSearchCV(self.predictor.pipe, param_grid = self.params)

df = df_data_source('https://raw.githubusercontent.com/sets018/Ocelot/main/data_extraction/df_posts_housing_clean_final.csv','url',0.9,0.1)
sectors = borough_classifier(df)
sectors.get_sectors()

encoder = oh_encoder(df.data_source)
df_encoded = df_data_source(encoder.encode(),'pass',0.9,0.1)

xgboost = Predictor('xg_boost')
xgboost.load_model("best_model_params.json")

