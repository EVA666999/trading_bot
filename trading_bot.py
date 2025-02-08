import gymnasium as gym
import gym_anytrading
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
import tensorflow as tf
import matplotlib.pyplot as plt
from finta import TA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from gymnasium import spaces
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import seaborn as sns
from scipy import stats




output_file = 'trading_data.csv'
model_file = 'ppo_trading_model.zip'
file = 'btc_15m_data_2018_to_2024-2024-10-10.csv'
predict_path = 'btc_price_predictions.csv'
max_steps = 2000
epochs = 1
BALANCE = 20000
frame_bound = 500


def preprocess_trading_data(file):
    """
    Функция для предварительной обработки данных о торговле.
    """
    # Загружаем данные из CSV файла
    df = pd.read_csv(file)

    # Приводим имена столбцов к нижнему регистру и удаляем пробелы
    df.columns = df.columns.str.strip().str.lower()

    # Преобразуем столбец 'open time' в datetime
    df['open time'] = pd.to_datetime(df['open time'])

    # Применяем индикаторы с использованием finta
    df['sma_20'] = TA.SMA(df, 20)  # Простой скользящий средний с периодом 20
    df['rsi'] = TA.RSI(df)  # Индикатор относительной силы (RSI)
    df['adx'] = TA.ADX(df)  # Индикатор среднего направленного индекса (ADX)
    df['macd'] = TA.MACD(df)['MACD']  # Добавим MACD
    df['atr'] = TA.ATR(df)


    # Удаляем строки с пропущенными значениями
    df = df.dropna()

    # Переупорядочиваем столбцы для удобства
    df = df[['open time', 'open', 'high', 'low', 'close', 'volume', 'sma_20',
             'rsi', 'adx', 'macd', 'predicted price', 'atr']]

    return df


# Проверка содержимого DataFrame
df = preprocess_trading_data(file)
print(df.columns)  # Выводим все имена столбцов, чтобы убедиться, что они присутствуют
print(df.head())  

# Обучение модели


# def calculate_vif(df, indicators):
#     """
#     Функция для расчета VIF (Variance Inflation Factor) для всех указанных индикаторов.
#     """
#     X = df[indicators].copy()
#     X = add_constant(X)
    
#     # Проверка на нулевую дисперсию
#     zero_variance_cols = [col for col in X.columns if X[col].std() == 0]
#     if zero_variance_cols:
#         print(f"Колонки с нулевой дисперсией: {zero_variance_cols}")
#         X = X.drop(columns=zero_variance_cols)
    
#     # Расчет VIF для каждого столбца
#     vif_data = pd.DataFrame()
#     vif_data["Variable"] = X.columns
#     vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
#     # Вывод матрицы корреляции
#     correlation_matrix = df[indicators].corr()  # Рассчитываем корреляцию
#     print("Матрица корреляции:")
#     print(correlation_matrix)

#     # Визуализация матрицы корреляции с помощью тепловой карты
#     sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
#     plt.title("Корреляционная матрица")
#     plt.show()

#     return vif_data

# # Перечислите индикаторы, для которых нужно рассчитать VIF
# indicators = ['rsi', 'adx', 'macd', 'predicted price']

# # Пример: загрузите ваши данные в df
# # df = pd.read_csv('your_data.csv')  # Раскомментируйте и загрузите ваш датасет

# # Вычисление VIF и вывод результатов
# vif_data = calculate_vif(df, indicators)
# print(vif_data)

# # stat, p_value = stats.shapiro(df['close'])
# # print(f'Statistic: {stat}, p-value: {p_value}')


# Определение пользовательской среды для обучения
class CustomStocksEnv(gym.Env):
    def __init__(self, df, frame_bound=(0, 0), window_size=50, position_fraction=0.2, model=None, 
                 min_profit_threshold=0.01, min_buy_threshold=0.05, trailing_stop_threshold=0.05):
        super(CustomStocksEnv, self).__init__()

        # Инициализация всех параметров
        self.df = df
        self.frame_bound = frame_bound
        self.window_size = window_size
        self.initial_balance = BALANCE
        self.balance = self.initial_balance
        self.money = 0
        self.count_for_trailing_stop = 0
        self.trade_count = 0  # Счетчик сделок
        self.max_trades_per_interval = 10  # Ограничение на количество сделок

        # Параметры для продажи и трейлинг стопа
        self.min_profit_threshold = min_profit_threshold
        self.min_buy_threshold = min_buy_threshold
        self.trailing_stop_threshold = trailing_stop_threshold

        # Action space (0: Hold, 1: Buy, 2: Sell)
        self.action_space = spaces.Discrete(3)

        # Инициализация current_step
        self.current_step = 0

        # Устанавливаем observation_space в соответствии с динамически полученной размерностью
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(302,), dtype=np.float32
        )

        # Параметры позиции
        self.position_size = 0
        self.max_position_size = 2
        self.last_buy_price = None
        self.max_price_since_buy = None
        self.total_profit = 0
        self.position_fraction = position_fraction
        self.model = model
        self.buy_prices = []

    def reset(self, seed=None, **kwargs):
        # Сбрасываем значения в начале эпизода
        if seed is not None:
            np.random.seed(seed)

        self.current_step = self.frame_bound[0]
        self.done = False
        self.last_buy_price = None
        self.max_price_since_buy = None  # Сброс максимальной цены
        self.total_profit = 0
        self.position_size = 0  # Сброс позиции при перезапуске
        self.balance = self.initial_balance
        self.money = 0
        self.trade_count = 0

        obs = self._get_observation().flatten()
        expected_shape = self.observation_space.shape[0]
        if obs.shape[0] < expected_shape:
            padding = expected_shape - obs.shape[0]
            obs = np.pad(obs, (0, padding), 'constant')

        return obs, {}

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= self.frame_bound[1]

        if self.current_step >= len(self.df):
            self.current_step = len(self.df) - 1
            done = True

        current_price = self.df.iloc[self.current_step]['close']
        predicted_price = self.df.iloc[self.current_step]['predicted price']

        # Проверяем, есть ли ATR в данных
        if 'atr' in self.df.columns:
            atr_value = self.df.iloc[self.current_step]['atr']
        else:
            atr_value = 0  # Если ATR нет, трейлинг-стоп не будет работать корректно

        # Расчет мультипликатора волатильности и динамических порогов
        volatility_multiplier = atr_value / self.df['atr'].mean() if self.df['atr'].mean() != 0 else 1  # Сравнение с средним ATR
        self.min_buy_threshold = max(0.01, 0.05 * volatility_multiplier)  # Динамическое изменение порога покупки
        self.min_profit_threshold = max(0.01, 0.03 * volatility_multiplier)  # Динамическое изменение порога продажи

        atr_multiplier = 1.5  # Коэффициент для адаптации трейлинг-стопа
        dynamic_trailing_stop = atr_multiplier * atr_value  

        info = {
            'Step': self.current_step,
            'Action': 'Hold',
            'Buy Price': None,
            'Sell Price': None,
            'Total Profit': self.total_profit,
            'Position Size': self.position_size,
            'Balance': self.balance,
            'current_price': current_price,
            'price_for_active': None,
            'sell_price_for_active': None,
            'Reward': None,
            'Trailing Stop Triggered': False  
        }

        action_taken = 'Hold'

        # Получаем все необходимые признаки
        observation = self._get_observation()
        model_action = self.model.predict(observation.reshape(1, -1))[0]

        # **Обновляем максимум цены после покупки**
        if self.max_price_since_buy is not None and current_price > self.max_price_since_buy:
            self.max_price_since_buy = current_price

        # Порог для покупки можно сделать адаптивным
        volatility_factor = 1 + (atr_value / self.df['atr'].mean())  # коэффициент волатильности

        # Уменьшаем порог, если волатильность высока
        dynamic_buy_threshold = self.min_buy_threshold * volatility_factor

        # **Покупка**
        if model_action == 1 and self.balance > 0 and predicted_price > current_price + dynamic_buy_threshold and len(self.buy_prices) < 5:
            amount_to_invest = self.balance * self.position_fraction
            units_to_buy = amount_to_invest / current_price

            if units_to_buy > 0:
                cost = units_to_buy * current_price
                self.balance -= cost
                self.position_size += units_to_buy
                self.buy_prices.append(current_price)
                self.trade_count += 1

                self.last_buy_price = current_price  
                self.max_price_since_buy = current_price  

                action_taken = 'Buy'
                info['Buy Price'] = current_price
                info['price_for_active'] = cost


        # **Продажа**
        elif model_action == 2 and self.position_size > 0:
            avg_buy_price = sum(self.buy_prices) / len(self.buy_prices)
            sell_price = current_price
            profit = (sell_price - avg_buy_price) * self.position_size
            profit_margin = profit / avg_buy_price

            # **Трейлинг-стоп на основе ATR**
            if self.max_price_since_buy is not None and sell_price <= self.max_price_since_buy - dynamic_trailing_stop:
                self.count_for_trailing_stop += 1  
                info['Trailing Stop Triggered'] = True  
                print(f'Трейлинг-стоп сработал! Количество: {self.count_for_trailing_stop}')
                info['Trailing Stop Count'] = self.count_for_trailing_stop

                sell_cost = sell_price * self.position_size
                self.balance += sell_cost
                self.total_profit = self.balance - self.initial_balance
                self.position_size = 0
                self.trade_count += 1
                self.buy_prices.clear()
                self.max_price_since_buy = None  

                action_taken = 'Sell (Trailing Stop Triggered)'
                info['Sell Price'] = sell_price
                info['sell_price_for_active'] = sell_cost
                info['Total Profit'] = self.total_profit

            # **Продажа при достижении порога прибыли**
            elif profit_margin > self.min_profit_threshold:
                sell_cost = sell_price * self.position_size
                self.balance += sell_cost
                self.total_profit = self.balance - self.initial_balance
                self.position_size = 0
                self.buy_prices.clear()
                self.max_price_since_buy = None  

                action_taken = 'Sell'
                info['Sell Price'] = sell_price
                info['sell_price_for_active'] = sell_cost
                info['Total Profit'] = self.total_profit

        reward = self._calculate_reward(action_taken, current_price)
        info['Action'] = action_taken
        info['Reward'] = reward

        obs = self._get_observation().flatten()
        return obs, reward, done, False, info

    def _calculate_reward(self, action_taken, current_price):
        profit_ratio = self.total_profit / self.initial_balance
        reward = profit_ratio

        action_rewards = {
            'Buy': 0.5,   # Награда за покупку уменьшена
            'Sell': 1.5,  # Награда за продажу нормализована
            'Hold': -0.2  # Меньший штраф за удержание
        }
        reward += action_rewards.get(action_taken, 0)

        # Бонус за позицию
        if self.position_size > 0:
            position_bonus = 1 + (self.position_size / self.max_position_size) * 2
            reward += position_bonus  

        # Бонус за выгодную покупку (если цена растет после покупки)
        if action_taken == 'Buy' and self.last_buy_price:
            expected_growth = (current_price - self.last_buy_price) / self.last_buy_price
            reward += max(0, expected_growth * 4)  # Усиление награды за рост цены после покупки

        # Динамический бонус при продаже
        if action_taken == 'Sell' and self.last_buy_price:
            price_diff = current_price - self.last_buy_price
            profit_bonus = 0.02 * (profit_ratio * self.initial_balance)  # Усиленный бонус за прибыльность
            reward += profit_bonus + (price_diff / self.last_buy_price if price_diff > 0 else -1)  # Увеличен штраф за убыточную продажу

        # Снижение штрафа за "Hold", если цена выше средней цены покупки
        if action_taken == 'Hold' and self.position_size > 0:
            avg_buy_price = sum(self.buy_prices) / len(self.buy_prices) if self.buy_prices else 0
            if avg_buy_price and current_price > avg_buy_price:
                reward += 0.5  # Усиленный бонус за удержание, если цена растет

        # Новый штраф за слишком частые сделки (анти-скальпинг)
        if self.trade_count > self.max_trades_per_interval:
            reward -= 1  # Меньший штраф за слишком частые сделки

        return reward


    def _get_observation(self):
        # Нормализуем данные для всех строк (по всему столбцу)
        norm_close = (self.df['close'] - self.df['close'].mean()) / self.df['close'].std()
        norm_predicted = (self.df['predicted price'] - self.df['predicted price'].mean()) / self.df['predicted price'].std()
        norm_rsi = (self.df['rsi'] - self.df['rsi'].mean()) / self.df['rsi'].std()
        norm_adx = (self.df['adx'] - self.df['adx'].mean()) / self.df['adx'].std()
        norm_macd = (self.df['macd'] - self.df['macd'].mean()) / self.df['macd'].std()
        norm_atr = (self.df['atr'] - self.df['atr'].mean()) / self.df['atr'].std()  # Нормализация ATR

        # Берем нормализованные значения для текущего окна
        norm_close_window = norm_close.iloc[self.current_step:self.current_step + self.window_size].values
        norm_predicted_window = norm_predicted.iloc[self.current_step:self.current_step + self.window_size].values
        norm_rsi_window = norm_rsi.iloc[self.current_step:self.current_step + self.window_size].values
        norm_adx_window = norm_adx.iloc[self.current_step:self.current_step + self.window_size].values
        norm_macd_window = norm_macd.iloc[self.current_step:self.current_step + self.window_size].values
        norm_atr_window = norm_atr.iloc[self.current_step:self.current_step + self.window_size].values

        # Диагностика после извлечения окон
        # print(f"norm_close_window.shape: {norm_close_window.shape}, norm_predicted_window.shape: {norm_predicted_window.shape}")
        # print(f"norm_rsi_window.shape: {norm_rsi_window.shape}, norm_adx_window.shape: {norm_adx_window.shape}")
        # print(f"norm_macd_window.shape: {norm_macd_window.shape}")
        # print(f"norm_atr_window.shape: {norm_atr_window.shape}")

        # Добавляем баланс и размер позиции, без нормализации
        balance = np.array([self.balance])
        position_size = np.array([self.position_size])

        # Проверка формы каждого окна
        expected_shape = self.window_size  # ожидаемая длина каждого окна

        # Если одно из окон не имеет ожидаемой формы, это может вызвать ошибку
        # if len(norm_close_window) != expected_shape:
        #     print("Ошибка: размер окна norm_close_window не совпадает с ожидаемым")
        # if len(norm_predicted_window) != expected_shape:
        #     print("Ошибка: размер окна norm_predicted_window не совпадает с ожидаемым")
        # if len(norm_rsi_window) != expected_shape:
        #     print("Ошибка: размер окна norm_rsi_window не совпадает с ожидаемым")
        # if len(norm_adx_window) != expected_shape:
        #     print("Ошибка: размер окна norm_adx_window не совпадает с ожидаемым")
        # if len(norm_macd_window) != expected_shape:
        #     print("Ошибка: размер окна norm_macd_window не совпадает с ожидаемым")
        # if len(norm_atr_window) != expected_shape:
        #     print("Ошибка: размер окна norm_atr_window не совпадает с ожидаемым")

        # Объединяем все данные в одно наблюдение
        observation = np.concatenate([norm_close_window, norm_predicted_window, norm_rsi_window,
                                     norm_adx_window, norm_macd_window, norm_atr_window, balance, position_size])

        # Проверка формы объединенного наблюдения

        return observation







# Создаем среду для обучения
env = DummyVecEnv([lambda: CustomStocksEnv(df=df, frame_bound=(frame_bound, max_steps), window_size=50)])


model = PPO('MlpPolicy', env, verbose=1, learning_rate=0.0005, batch_size=128,
            n_steps=2048, ent_coef=0.01, clip_range=0.2, gae_lambda=0.95)

env.envs[0].model = model 

# Обучаем модель
model.learn(total_timesteps=max_steps)

# Сохраняем модель
model.save('ppo_trading_model')
# env = DummyVecEnv([lambda: CustomStocksEnv(df=df, frame_bound=(frame_bound, max_steps), window_size=50)])

# # Загружаем модель
# model = PPO.load('ppo_trading_model')

# # Привязываем модель к среде
# env.envs[0].model = model


def test_model_and_save_info(model, env, df, output_file=output_file,
                             action_frequency=1, max_steps=max_steps,
                             trailing_stop_threshold=0.1, stop_loss_threshold=0.05):
    df.columns = df.columns.str.strip()
    df.columns = df.columns.astype(str)
    
    # Инициализация переменных
    obs = env.reset()
    done = False
    steps = []
    info_list = []
    total_reward = 0
    rewards = []
    actions = []
    buy_prices = []
    sell_prices = []
    sell_types = []
    total_profit_per_step = []
    predicted_prices = []
    balances = []
    current_prices = []
    money = []
    new_balances = []
    price_for_active = []
    sell_price_for_active = []
    sell_count = 0
    buy_count = 0
    hold_count = 0
    trailing_stop_count = 0  # Счётчик трейлинг стопов
    stop_loss_sell_count = 0  # Счётчик стоп-лоссов
    step_count = 0
    total_steps = len(df)
    
    max_price_since_buy = None  # Инициализация максимальной цены после покупки

    while not done and step_count < max_steps:
        if step_count % action_frequency == 0:
            action, _states = model.predict(obs)
        else:
            action = [0]

        # Выполнение действия
        step_result = env.step(action)
        if len(step_result) == 4:
            obs, reward, done, info = step_result
            truncated = False
        else:
            obs, reward, done, truncated, info = step_result

        total_reward += reward
        rewards.append(reward)
        info_list.append(info)
        steps.append(step_count)

        if isinstance(action, list):
            action = action[0]

        info = info[0]  # Берем первый элемент из списка

        if info['Action'] == 'Buy':
            actions.append('Buy')
            buy_prices.append(info['Buy Price'])
            sell_prices.append(None)
            sell_types.append(None)
            buy_count += 1
            # Устанавливаем максимальную цену как цену покупки
            max_price_since_buy = info['Buy Price']
            print(f"📈 Покупка по цене: {max_price_since_buy}")

        elif info['Action'].startswith('Sell'):
            sell_price = info['Sell Price']

            # Проверка на трейлинг стоп
            if info.get('Trailing Stop Triggered', False):
                trailing_stop_count += 1  # Увеличиваем счетчик стоп-лоссов
                actions.append('Sell (Trailing Stop)')
                print(f"📊 Трейлинг стоп сработал! Общее количество: {trailing_stop_count}")
            else:
                actions.append('Sell')

            sell_prices.append(sell_price)
            buy_prices.append(None)
            sell_types.append(None)
            sell_count += 1
        else:
            actions.append('Hold')
            buy_prices.append(None)
            sell_prices.append(None)
            hold_count += 1

        # Получаем информацию о прибыли и балансе
        total_profit = info.get('Total Profit', None)
        balance = info.get('Balance', None)
        current_price = info.get('current_price', None)
        money_value = info.get('Money', 0)
        new_balance = info.get('New Balance', 0)
        price_for_active.append(info.get('price_for_active', None))
        sell_price_for_active.append(info.get('sell_price_for_active', None))
        total_profit_per_step.append(total_profit)
        new_balances.append(new_balance)
        money.append(money_value)
        predicted_price = info.get('Predicted price', None)
        predicted_prices.append(predicted_price)
        current_prices.append(current_price)
        balances.append(balance)
        step_count += 1
        print(f"Шаг {step_count}/{total_steps} завершен")

    # Выравнивание длины списков
    max_length = max(
        len(steps), len(actions), len(rewards),
        len(total_profit_per_step), len(balances),
        len(info_list), len(predicted_prices), len(current_prices),
        len(money), len(new_balances), len(buy_prices), len(sell_prices),
        len(price_for_active), len(sell_price_for_active), len(sell_types)
    )

    lists_to_pad = [
        steps, actions, rewards, total_profit_per_step, info_list, predicted_prices,
        balances, current_prices, money, new_balances, buy_prices, sell_prices,
        price_for_active, sell_price_for_active, sell_types
    ]
    for lst in lists_to_pad:
        lst += [None] * (max_length - len(lst))

    # Создаем DataFrame
    df_output = pd.DataFrame({
        'Step': steps,
        'Action': actions,
        'Reward': rewards,
        'Balance': balances,
        'Current Price': current_prices,
        'Total Profit': total_profit_per_step,
        'Price for Active': price_for_active,
        'Sell Price for Active': sell_price_for_active
    })

    # Проверка данных перед записью
    print("Проверка данных перед записью в CSV:")
    print(df_output.head())  # Выведем первые несколько строк для проверки

    # Сохраняем результаты в CSV-файл
    try:
        df_output.to_csv(output_file, mode='w', index=False)
        print(f"Информация сохранена в {output_file}")
    except Exception as e:
        print(f"Ошибка при сохранении данных: {e}")

    print(f"Завершено: 100%")
    print(f"Продажа (Sell): {sell_count}, Покупка (Buy): {buy_count}, Удержание (Hold): {hold_count}, Трейлинг стопы (Trailing Stops): {trailing_stop_count}")

    return total_reward


# Тестирование модели и сохранение информации
total_reward = test_model_and_save_info(model, env, df, output_file=output_file, max_steps=max_steps)
print(f"Общая награда за тестирование: {total_reward}")
