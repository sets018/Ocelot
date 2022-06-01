# Importing libraries
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import streamlit as st
import pickle
import sklearn
from plotly.tools import FigureFactory as ff
import plotly.express as px
from PIL import Image

# This class contains the data (a pandas df) as an attribute
class df_data_source():
    def __init__(self, source, source_type, train_size_, test_size_):
        self.train_size_ = train_size_
        self.test_size_ = test_size_
        self.source_type = source_type
        self.source = source
        if (self.source_type == 'url'):
            self.get_data_by_url(source)
        elif (self.source_type == 'path'):
            self.get_data_by_path(source)
        elif (self.source_type == 'pass'):
            self.get_data(source)
        elif (self.source_type == 'input'):
            self.get_data(source)
        if (self.source_type != 'input'):
            self.split_data(self.train_size_, self.test_size_)
    # Method to split data train/test
    # This method also reshapes the data in order to be used by the model
    def split_data(self, train_size_, test_size_):
        self.train_size_ = train_size_
        self.test_size_ = test_size_
        self.price_subset = self.data_source.iloc[:, -1].values
        self.non_price_subset = self.data_source.iloc[:, :-1].values
        self.data_subset_train, self.data_subset_test, self.price_subset_train, self.price_subset_test = train_test_split(
        self.non_price_subset, self.price_subset, test_size=self.test_size_, train_size=self.train_size_)
        self.price_subset_train = self.price_subset_train.reshape(-1, 1)
        self.price_subset_test = self.price_subset_test.reshape(-1, 1)

    # Method to delete the duplicates
    # A post is a duplicate if it shares the same post_id/post_link
    def delete_duplicates(self):
        self.data_source = df.data_source.drop_duplicates(subset="post_link")

    def delete_nan(self, column_wnan):
        self.column_wnan = column_wnan
        self.data_source = self.data_source[self.data_source[column_wnan].notna()]
        # sns.heatmap(self.data_source.corr())

    def delete_outliers(self):
        self.q1 = self.data_source.quantile(0.25)
        self.q3 = self.data_source.quantile(0.75)
        self.iqr = self.q3 - self.q1
        self.data_source = self.data_source[
            ~((self.data_source < (self.q1 - 1.5 * self.iqr)) | self.data_source > (self.q3 + 1.5 * self.iqr)).any(
                axis=1)]

    def drop_column(self, column_name):
        self.column_name = column_name
        self.data_source.drop(self.column_name, axis=1, inplace=True)

    # The dataframe is pulled from github using an url
    def get_data_by_url(self, data_source_url):
        self.data_source_url = data_source_url
        self.data_source = pd.read_csv(self.data_source_url)

    # The dataframe is pulled from a specific path
    def get_data_by_path(self, data_source_path):
        self.data_source_path = data_source_path
        self.data_source = pd.read_csv(self.data_source_path)

    # The dataframe is assigned (=)
    def get_data(self, data_source):
        self.data_source = data_source
        
    def add_reg(self, data_added):
        self.data_added = data_added
        self.data_source = pd.concat([self.data_source, data_added], sort=False).reset_index(drop = True)
        #self.data_source.reset_index()
        
    def get_ds_columns(self):
        self.list_w_columns = list(self.data_source)
        return self.list_w_columns  
    def normalize_data(self):
        self.data_source.loc[self.data_source.Price > 1000000, 'Price'] /= 1000000
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
        self.list_riomar = ['Adela de Char Etp. 1',
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
            'Alameda del Rio',
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
            'Barranquilla',
            'Villanueva']
        self.list_metr = [
            '7 de Abril',
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
            'Soledad',
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
        self.hood_list = []
    def get_hoods(self,borough):
        self.borough = borough
        if (self.borough == 'Riomar'):
            self.hood_list = self.data.data_source[df.data_source["Borough"] == 'Riomar']["neighborhood"].unique().tolist()
        elif (self.borough == 'Norte-Centro Histórico'):
            self.hood_list = self.data.data_source[df.data_source["Borough"] == 'Norte-Centro Histórico']["neighborhood"].unique().tolist()
        elif (self.borough == 'Metropolitana'):
            self.hood_list = self.data.data_source[df.data_source["Borough"] == 'Metropolitana']["neighborhood"].unique().tolist()
        elif (self.borough == 'Sur Occidente'):
            self.hood_list = self.data.data_source[df.data_source["Borough"] == 'Sur Occidente']["neighborhood"].unique().tolist()
        elif (self.borough == 'Sur Oriente'):
            self.hood_list = self.data.data_source[df.data_source["Borough"] == 'Sur Oriente']["neighborhood"].unique().tolist()
        elif (self.borough == 'Soledad'):
            self.hood_list = self.data.data_source[df.data_source["Borough"] == 'Soledad']["neighborhood"].unique().tolist()
            
class plotting():
    def __init__(self, data):
        self.plot_data = df_data_source('https://raw.githubusercontent.com/sets018/Ocelot/main/data_extraction/df_posts_housing_final.csv', 'url',0.9, 0.1)
        self.prepare_data()
    def dist(self, y_column):
        self.y_column = y_column
        sns.set_style('whitegrid')
        self.fig = plt.figure(figsize=(24,20))
        sns.distplot(self.plot_data.data_source[self.y_column], color = 'r')
        self.show_plot()
    def scatter(self,var,cat_var):
        self.var = var
        self.cat_var = cat_var
        self.fig = px.scatter(self.plot_data.data_source,x=self.var, y='Price', color=self.cat_var)
        st.plotly_chart(self.fig, use_container_width=False)
        #self.fig = plt.figure(figsize=(24,20))
        #sns.scatterplot(self.var, self.plot_data.data_source["Price"], data = self.plot_data.data_source, color = 'orange', edgecolor = 'b', s = 150)
        #plt.title('Area / Sale Price', fontsize = 16)
        #plt.xlabel('Area', fontsize = 14)
        #plt.ylabel('Sale Price', fontsize = 14)
        #plt.xticks(fontsize = 12)
        #plt.yticks(fontsize = 12)
        #self.show_plot()
    def corr(self):
        self.fig = plt.figure(figsize=(15, 15))
        sns.heatmap(self.plot_data.data_source.corr(), annot = True, cmap = 'magma')
        self.show_plot()
    def show_plot(self):
        st.pyplot(self.fig)
    def prepare_data(self):
        self.plot_data.data_source = self.plot_data.data_source[self.plot_data.data_source.Price < 500000000]
        self.plot_data.data_source.loc[self.plot_data.data_source.Price > 1000000, 'Price'] /= 1000000
        
class oh_encoder(OneHotEncoder):
    def __init__(self, data):
        super(OneHotEncoder, self).__init__()
        self.data = data
        # self.data.drop (['post_title'], 1, inplace=True)
        # self.data.drop(['post_link'], 1, inplace=True)
        # self.data.drop(['city'], 1, inplace=True)
        self.handle_unknown = "ignore"
        self.categories = 'auto'
        self.sparse = False
        self.dtype = int
        self.drop = None
        self.max_categories = None
        self.min_frequency = None

    def encode(self):
        self.columns_cat = self.data.select_dtypes(include=["object"]).columns
        self.encoded_data = self.fit_transform(self.data[self.columns_cat])
        self.encoded_data = pd.DataFrame(self.encoded_data, columns=self.get_encoded_columns())
        self.oh_data = self.encoded_data.join(self.data)
        self.oh_data = self.delete_non_encoded_columns(self.oh_data)
        return self.oh_data

    def get_encoded_columns(self):
        self.columns_cats_encoded = []
        for column in self.columns_cat:
            self.columns_cats_encoded += [f"{column[0]}_{cat}" for cat in list(self.data[column].unique())]
        return self.columns_cats_encoded

    def delete_non_encoded_columns(self, data_encoded):
        self.data_encoded = data_encoded
        for column in self.columns_cat:
            self.data_encoded.drop([column], 1, inplace=True)
        return self.data_encoded


class Predictor():
    def __init__(self, source_model, data):
        self.source_model = source_model
        self.data = data
        self.get_data()
        if (self.source_model == 'linear_regression'):
            self.linear_regression()
        if (self.source_model == 'decision_tree'):
            self.decision_tree()
        if (self.source_model == 'random_forest'):
            self.random_forest()
        if (self.source_model == 'gradient_boosting'):
            self.gradient_boosting()
        if (self.source_model == 'xg_boost'):
            self.xg_boost()

    def get_data(self):
        # self.data.split_data(self.train_size_,self.test_size_)
        self.x_train = self.data.data_subset_train
        self.y_train = self.data.price_subset_train
        self.x_test = self.data.data_subset_test
        self.y_test = self.data.price_subset_test

    def get_pipeline(self, model):
        self.model = model
        self.reg = Pipeline(
            [('model', TransformedTargetRegressor(regressor=self.model, func=np.log, inverse_func=np.exp))])

    def linear_regression(self):
        self.lr = LinearRegression()
        self.linear_reg = TransformedTargetRegressor(regressor=self.lr, func=np.log1p, inverse_func=np.expm1)
        self.linear_reg.fit(self.x_train, self.y_train)
        print('Training score : ', self.linear_reg.score(self.x_train, self.y_train))
        print('Real score : ', self.linear_reg.score(self.x_test, self.y_test))

    def decision_tree(self):
        self.dec_tree = DecisionTreeRegressor(criterion="mse", random_state=42)
        self.get_pipeline(self.dec_tree)
        # self.model = TransformedTargetRegressor(regressor = self.model, func = np.log, inverse_func = np.exp)

    def random_forest(self):
        self.rand_forest = RandomForestRegressor(n_jobs=-1, random_state=42, bootstrap=True)
        self.get_pipeline(self.rand_forest)
        # self.model = TransformedTargetRegressor(regressor = self.model, func = np.log, inverse_func = np.exp)

    def gradient_boosting(self):
        self.grad_boost = GradientBoostingRegressor(random_state=42, max_features="sqrt")
        self.get_pipeline(self.grad_boost)

    def xg_boost(self):
        self.xg_boost_ = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
        self.get_pipeline(self.xg_boost_)

    def fit_print_scr(self):
        self.reg.fit(self.x_train, self.y_train)
        # self.model.fit(self.x_train,self.y_train)
        # print('Training score : ',self.reg.score(self.x_train,self.y_train))
        # print('Real score : ',self.reg.score(self.x_test,self.y_test))
        #self.preds = self.reg.predict(self.x_test)
        #self.r2, self.MAE, self.MSE = self.get_scores(self.preds, self.y_test)
        #print('MAE : ', self.MAE)
        #print('MSE : ', self.MSE)
        #print('r2 : ', self.r2)
        #print('Training score : ', self.reg.score(self.x_train, self.y_train))
        #print('Real score : ', self.reg.score(self.x_test, self.y_test))
        #print('Best params : ', self.pipe.best_params_)

    # print('RMSE : ',np.sqrt(mean_squared_error(self.y_train, self.preds)))
    # print('MAE : ',mean_absolute_error(self.y_train, self.preds))
    # self.cross_val_()
    def get_scores(self, preds, y_test):
        self.preds = preds
        self.y_test = y_test
        self.MSE = round(mean_squared_error(self.preds, self.y_test), 2)
        self.MAE = round(mean_absolute_error(self.preds, self.y_test), 2)
        self.r2 = round(r2_score(self.preds, self.y_test), 2)
        return self.r2, self.MAE, self.MSE
        # self.cross_val_()

    def cross_val_(self):
        self.cv_scores = cross_val_score(self.reg,
                                         self.x_train,
                                         self.y_train,
                                         scoring="neg_mean_squared_error",
                                         cv=10)
        self.cv_scores = np.sqrt(-self.cv_scores)
        print('MSE cross_val: ', self.cv_scores.mean())


class model(Predictor):
    def __init__(self, fitted_reg):
        super(Predictor, self).__init__()
        self.fitted_reg = fitted_reg
    def get_predictions(self, y_data):
        self.y_data = y_data
        self.pred = self.fitted_reg.predict(y_data)
        return self.pred
    
class tuning():
    def __init__(self, predictor, scoring, params):
        self.predictor = predictor
        self.scoring = scoring
        self.params = params
        self.tune()

    def tune(self):
        self.predictor.reg = GridSearchCV(self.predictor.reg, param_grid=self.params, scoring=self.scoring, cv=5)
        # self.model = GridSearchCV(self.predictor.pipe, param_grid = self.params)

class prediction_data(df_data_source):
    def __init__(self, cat_pred_data, num_pred_data, cat_pred_cols, num_pred_cols):
        super(df_data_source, self).__init__()
        self.cat_pred_data = cat_pred_data
        self.num_pred_data = num_pred_data
        self.cat_pred_cols = cat_pred_cols
        self.num_pred_cols = num_pred_cols
        self.format_data()
    def format_data(self):
        self.pred_data_df_cat = pd.DataFrame([self.cat_pred_data],columns = self.cat_pred_cols)
        self.pred_data_df_num = pd.DataFrame([self.num_pred_data], columns = self.num_pred_cols)
        self.data_source = self.pred_data_df_cat.join(self.pred_data_df_num)
    def get_encoded_pred_data(self,columns_encoded,encoded_data):
        self.columns_encoded = columns_encoded
        self.encoded_data = encoded_data
        self.data_source = pd.DataFrame(columns = self.columns_encoded)
        self.add_reg(self.encoded_data)
        self.data_source = self.data_source.fillna(0)
        self.data_source.drop("Price", axis=1, inplace=True)
    def fix_columns(self, hood):
        self.hood = hood
        self.data_source['n_'+self.hood] = self.data_source['N_'+self.hood]
        self.drop_column('N_'+self.hood)
        st.write('n_'+self.hood)
        st.write('N_'+self.hood)
class user_input():
    def __init__(self, var, type, data, type_data, input_list):
        self.var = var
        self.type = type
        self.data = data
        self.type_data = type_data
        self.input_list = input_list
        self.get_input()
    def get_input(self):
        self.user_input = 'placeholder'
        if (self.type == 'radio'):
            self.get_radio()
        elif (self.type == 'slider'):
            self.get_slider()
        self.input_list.append(self.user_input)
    def get_radio(self):
        if (self.type_data == 'dataframe'):
            self.user_input = st.radio(
                self.var,
                np.unique(self.data.data_source[self.var]))
        elif (self.type_data == 'list'):
            self.user_input = st.radio(
                self.var,
                self.data)
        st.write(self.var,": ",self.user_input)
    def get_slider(self):
        if (self.type_data == 'dataframe'):
            self.user_input = st.slider(self.var, 0, max(self.data.data_source[self.var]), 1)
        elif (self.type_data == 'list'):
            self.user_input = st.slider(self.var, 0, max(self.data), 1)
        st.write(self.var,": ",self.user_input)
        
df = df_data_source(
    'https://raw.githubusercontent.com/sets018/Ocelot/main/data_extraction/df_posts_housing_final.csv', 'url',
    0.9, 0.1)
df.normalize_data()
sectors = borough_classifier(df)
sectors.get_sectors()
    
st.set_page_config(
    page_title="Ocelot",
    layout="centered",
    initial_sidebar_state="auto",
)

st.title('Ocelot')
st.write("[![Star](https://img.shields.io/github/stars/sets018/Ocelot.svg?logo=github&style=social)](https://gitHub.com/sets018/Ocelot)")
logo = Image.open('Ocelot_logo.png')
st.image(logo)
st.subheader('Estimate the price of real estate in Barranquilla, Colombia')
st.write('House price predictions from a machine learning model')
st.write("""## How it works""")
st.write("Get your estimation by using the button at the top left side of the screen to enter the data of the property (Area,Neighborhood,Bedrooms, etc). When all the data is entered press the make prediction button this will give you a 'predicted sale price' based on your selections.")

st.write("""## The data being used""")
if st.checkbox('Show dataframe'):
    st.dataframe(df.data_source)

cat_input = []
num_input = []

#borough_input = user_input('Borough', 'radio', sectors.sectors_values, 'list', cat_input)

#sectors.get_hoods(borough_input.user_input)
#hoods_input = user_input('Neighborhood', 'radio', sectors.hood_list, 'list', cat_input)

input_columns_cat = ['Borough','neighborhood','condition','estrato','property_type']
input_columns_num = ['Area','bedrooms','bathrooms','garages']

with st.sidebar:
    for column in input_columns_cat:
        if (column == 'Borough'):
            borough_input = user_input(column, 'radio', sectors.sectors_values, 'list', cat_input)
            sectors.get_hoods(borough_input.user_input)
        elif (column == 'neighborhood'):
            hoods_input = user_input(column, 'radio', sectors.hood_list, 'list', cat_input)
        else:
            usr_input_cat = user_input(column, 'radio', df , 'dataframe', cat_input)
            #cat_input.append(usr_input_cat.user_input)
    for column in input_columns_num:
        usr_input_num = user_input(column, 'slider', df, 'dataframe', num_input)
        #num_input.append(usr_input_num.user_input)
price_plots = plotting(df)
if st.checkbox('Show price distribution plot'):
    num_plot_input = st.radio(
                'Numeric variable',
                input_columns_num)
    if st.button('Make plot'):
        price_plots.dist(num_plot_input)
if st.checkbox('Show scatter plot'):
   num_plot_input = st.radio(
                'Numeric variable',
                input_columns_num)
   cat_plot_input = st.radio(
                'Categorical variable',
                input_columns_cat)
   if st.button('Make plot'):
        price_plots.scatter(num_plot_input,cat_plot_input)
if st.checkbox('Show data correlation'):
    if st.button('Make plot'):
        price_plots.corr()

if st.button('Make Prediction'):
    pred_data = prediction_data(cat_input, num_input, input_columns_cat, input_columns_num)
    encoder = oh_encoder(df.data_source)
    df_encoded = df_data_source(encoder.encode(), 'pass', 0.9, 0.1)
    st.subheader('Entered data')
    st.write(pred_data.data_source)
    pred_data_encoder = oh_encoder(pred_data.data_source)
    pred_data_encoded = pred_data_encoder.encode()
    columns_encoded = df_encoded.get_ds_columns()
    pred_data.get_encoded_pred_data(columns_encoded,pred_data_encoded)
    #pred_data.fix_columns(hoods_input.user_input)
    shape = pred_data.data_source.shape
    if (got_model == 0):
        grad_boost = Predictor('gradient_boosting',df_encoded)
        params = {
            "model__regressor__n_estimators": [300],
            "model__regressor__max_depth": [5],
            "model__regressor__learning_rate": [0.1],
            }
        tuner = tuning(grad_boost,"r2",params)
        grad_boost.fit_print_scr()
        fitted_model = model(grad_boost.reg)
        got_model = 1
    prediction = fitted_model.get_predictions(pred_data.data_source)
    est_price = prediction[0]
    st.subheader('Prediction')
    st.write("Estimated price : ", int(int(est_price)*1000000))
