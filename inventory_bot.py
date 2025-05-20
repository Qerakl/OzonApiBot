import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import json
import os
from datetime import datetime

class InventoryBot:
    def __init__(self, model_path=None):
        """
        Инициализация InventoryBot
        
        Параметры:
        model_path (str, optional): Путь к сохраненной модели для загрузки
        """
        # Если указан путь к модели, загружаем её, иначе создаем новую
        if model_path and os.path.exists(model_path):
            try:
                import pickle
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                print(f"Модель успешно загружена из {model_path}")
                self.model_is_trained = True
            except Exception as e:
                print(f"Ошибка при загрузке модели: {str(e)}")
                self.model = RandomForestRegressor(n_estimators=100, random_state=42)
                self.model_is_trained = False
        else:
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model_is_trained = False
        
        # Сохраняем данные для повторного обучения
        self.features_data = None
        self.X_columns = None
    
    def load_data(self, file_path):
        """Загрузка данных из CSV файла"""
        print(f"Загрузка данных из {file_path}...")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл {file_path} не найден!")
        
        # Загружаем с учетом разделителя ";" и кодировки
        data = pd.read_csv(file_path, sep=";", encoding='utf-8', low_memory=False)
        print(f"Данные успешно загружены. Строк: {len(data)}")
        return data
    
    def preprocess_data(self, data):
        """Предобработка данных для обучения модели"""
        print("Предобработка данных...")
        
        # Переименуем колонки для удобства работы
        column_mapping = {
            'Принят в обработку': 'date',
            'Артикул': 'product_id',
            'OZON id': 'ozon_id',
            'Наименование товара': 'product_name',
            'Количество': 'quantity',
            'Итоговая стоимость товара': 'price',
            'Дата доставки': 'delivery_date',
            'Объемный вес товаров, кг': 'weight'
        }
        
        # Выбираем только нужные колонки
        needed_columns = list(column_mapping.keys())
        available_columns = [col for col in needed_columns if col in data.columns]
        
        if len(available_columns) < 3:
            raise ValueError("Недостаточно данных для анализа. Проверьте формат CSV файла.")
        
        # Создаем новый DataFrame только с нужными колонками
        processed_data = data[available_columns].copy()
        
        # Переименуем колонки
        for old_col, new_col in column_mapping.items():
            if old_col in processed_data.columns:
                processed_data.rename(columns={old_col: new_col}, inplace=True)
        
        # Конвертируем дату в формат datetime
        if 'date' in processed_data.columns:
            processed_data['date'] = pd.to_datetime(processed_data['date'], errors='coerce')
            
            # Фильтруем строки с некорректными датами
            processed_data = processed_data.dropna(subset=['date'])
            
            # Добавляем признаки даты
            processed_data['month'] = processed_data['date'].dt.month
            processed_data['day_of_week'] = processed_data['date'].dt.dayofweek
            processed_data['day'] = processed_data['date'].dt.day
            processed_data['week'] = processed_data['date'].dt.isocalendar().week
        
        # Конвертируем количество в числовой формат
        if 'quantity' in processed_data.columns:
            processed_data['quantity'] = pd.to_numeric(processed_data['quantity'], errors='coerce').fillna(0)
        else:
            # Если нет колонки quantity, создаем ее со значением 1
            processed_data['quantity'] = 1
        
        # Конвертируем цену в числовой формат
        if 'price' in processed_data.columns:
            processed_data['price'] = pd.to_numeric(processed_data['price'].astype(str).str.replace(',', '.'), errors='coerce').fillna(0)
        
        # Конвертируем вес в числовой формат
        if 'weight' in processed_data.columns:
            processed_data['weight'] = pd.to_numeric(processed_data['weight'].astype(str).str.replace(',', '.'), errors='coerce').fillna(0)
        
        return processed_data
    
    def prepare_features(self, data):
        """Подготовка признаков для модели"""
        print("Подготовка признаков для модели...")
        
        # Убедимся, что у нас есть product_id
        if 'product_id' not in data.columns:
            if 'ozon_id' in data.columns:
                data['product_id'] = data['ozon_id']
            else:
                # Создаем уникальные product_id для каждого наименования товара
                unique_products = data['product_name'].unique()
                product_id_map = {name: f"PROD_{i+1}" for i, name in enumerate(unique_products)}
                data['product_id'] = data['product_name'].map(product_id_map)
        
        # Группируем данные по артикулу и анализируем продажи по времени
        feature_columns = ['quantity']
        if 'price' in data.columns:
            feature_columns.append('price')
        if 'weight' in data.columns:
            feature_columns.append('weight')
        
        time_columns = []
        if 'month' in data.columns:
            time_columns.append('month')
        if 'week' in data.columns:
            time_columns.append('week')
        if 'day' in data.columns:
            time_columns.append('day')
        if 'day_of_week' in data.columns:
            time_columns.append('day_of_week')
        
        # Рассчитываем агрегированные статистики
        agg_dict = {col: ['sum', 'mean', 'count'] for col in feature_columns}
        for col in time_columns:
            agg_dict[col] = ['mean', 'std']
        
        grouped_data = data.groupby(['product_id', 'product_name']).agg(agg_dict)
        
        # Преобразуем мультииндекс в обычные колонки
        grouped_data.columns = ['_'.join(col).strip() for col in grouped_data.columns.values]
        grouped_data = grouped_data.reset_index()
        
        # Добавляем статистику по временным интервалам при наличии данных
        if 'date' in data.columns:
            # Получаем последнюю дату продажи для каждого товара
            last_dates = data.groupby('product_id')['date'].max().reset_index()
            last_dates.columns = ['product_id', 'last_sale_date']
            grouped_data = grouped_data.merge(last_dates, on='product_id', how='left')
            
            # Рассчитываем количество дней с последней продажи
            current_date = data['date'].max()
            grouped_data['days_since_last_sale'] = (current_date - grouped_data['last_sale_date']).dt.days
            
            # Частота продаж (количество дней продаж / общий период)
            sales_days = data.groupby('product_id')['date'].nunique().reset_index()
            sales_days.columns = ['product_id', 'unique_sales_days']
            grouped_data = grouped_data.merge(sales_days, on='product_id', how='left')
            
            total_days = (data['date'].max() - data['date'].min()).days + 1
            if total_days > 0:
                grouped_data['sales_frequency'] = grouped_data['unique_sales_days'] / total_days
            else:
                grouped_data['sales_frequency'] = 0
        
        # Добавляем текущие остатки (в реальной системе здесь должны быть данные из системы учета)
        # В данном примере просто симулируем случайные остатки
        np.random.seed(42)  # Добавляем seed для воспроизводимости
        grouped_data['current_stock'] = np.random.randint(0, 50, size=len(grouped_data))
        
        # Рассчитываем коэффициент востребованности
        if 'quantity_count' in grouped_data.columns and 'quantity_sum' in grouped_data.columns:
            grouped_data['demand_ratio'] = grouped_data['quantity_sum'] / (grouped_data['quantity_count'] + 1)
        
        return grouped_data
    
    def train_model(self, data, save_model_path=None):
        """
        Обучение модели машинного обучения
        
        Параметры:
        data (DataFrame): Данные для обучения
        save_model_path (str, optional): Путь для сохранения обученной модели
        """
        print("Обучение модели...")
        
        # Сохраняем данные для возможного переобучения
        self.features_data = data.copy()
        
        # Определяем признаки для обучения модели
        numeric_columns = [col for col in data.columns if 
                           col not in ['product_id', 'product_name', 'last_sale_date'] and 
                           data[col].dtype in [np.int64, np.float64]]
        
        if len(numeric_columns) < 3:
            print(f"Предупреждение: Мало признаков для обучения модели! Доступно только {len(numeric_columns)} признаков.")
            
        # Выбираем признаки для модели
        X = data[numeric_columns].copy()
        self.X_columns = numeric_columns  # Сохраняем для будущего прогнозирования
        
        # Заполняем пропущенные значения
        X = X.fillna(0)
        
        # В качестве целевой переменной используем сумму продаж с некоторым коэффициентом роста
        # для прогнозирования будущего спроса
        if 'quantity_sum' in data.columns:
            base_target = data['quantity_sum']
        elif 'quantity_mean' in data.columns:
            base_target = data['quantity_mean'] * data['quantity_count']
        else:
            # Если нет данных о количестве, используем случайные значения
            np.random.seed(42)  # Добавляем seed для воспроизводимости
            base_target = np.random.uniform(1, 10, size=len(data))
        
        # Добавляем случайный фактор роста для прогноза
        np.random.seed(42)  # Добавляем seed для воспроизводимости
        y = base_target * (1 + np.random.uniform(-0.1, 0.3, size=len(data)))
        
        # Обучаем модель
        try:
            self.model.fit(X, y)
            self.model_is_trained = True
            print("Модель успешно обучена!")
            
            # Сохраняем модель, если указан путь
            if save_model_path:
                try:
                    import pickle
                    os.makedirs(os.path.dirname(os.path.abspath(save_model_path)), exist_ok=True)  # Создаем директорию, если не существует
                    with open(save_model_path, 'wb') as f:
                        pickle.dump(self.model, f)
                    print(f"Модель успешно сохранена в {save_model_path}")
                except Exception as e:
                    print(f"Ошибка при сохранении модели: {str(e)}")
                    
        except Exception as e:
            print(f"Ошибка при обучении модели: {str(e)}")
            # Если не получилось обучить модель, создаем простой прогноз
            self.model_is_trained = False
    
    def predict(self, data):
        """Прогнозирование будущих продаж"""
        print("Прогнозирование продаж...")
        
        # Определяем признаки для прогнозирования
        if self.X_columns is not None:
            # Используем сохраненный список признаков, если есть
            numeric_columns = [col for col in self.X_columns if col in data.columns]
            # Если каких-то признаков не хватает, добавляем их со значением 0
            for col in self.X_columns:
                if col not in data.columns:
                    data[col] = 0
        else:
            numeric_columns = [col for col in data.columns if 
                            col not in ['product_id', 'product_name', 'last_sale_date'] and 
                            data[col].dtype in [np.int64, np.float64]]
        
        X = data[numeric_columns].copy().fillna(0)
        
        if self.model_is_trained:
            # Используем обученную модель для прогноза
            try:
                predictions = self.model.predict(X)
                # Корректируем отрицательные прогнозы
                predictions = np.maximum(predictions, 1)  # Минимум 1 единица товара
                return predictions
            except Exception as e:
                print(f"Ошибка при прогнозировании: {str(e)}")
                predictions = self._simple_prediction(data)
                return predictions
        else:
            # Если модель не обучена, используем простой прогноз
            predictions = self._simple_prediction(data)
            return predictions
    
    def _simple_prediction(self, data):
        """Простой прогноз на основе средних продаж"""
        print("Использование простого прогноза на основе средних продаж...")
        
        if 'quantity_sum' in data.columns and 'quantity_count' in data.columns:
            base_prediction = data['quantity_sum'] * (1.2 / (data['quantity_count'] + 1))
        elif 'quantity_mean' in data.columns:
            base_prediction = data['quantity_mean'] * 1.2
        else:
            # Если нет данных о продажах, даем минимальный прогноз
            base_prediction = np.ones(len(data)) * 5
        
        # Увеличиваем прогноз для товаров с высокой частотой продаж
        if 'sales_frequency' in data.columns:
            # Если частота продаж выше среднего, увеличиваем прогноз
            avg_frequency = data['sales_frequency'].mean()
            frequency_multiplier = np.where(
                data['sales_frequency'] > avg_frequency,
                1.3,  # Для популярных товаров множитель выше
                1.0   # Для остальных товаров обычный прогноз
            )
            base_prediction = base_prediction * frequency_multiplier
        
        # Учитываем дни с последней продажи (если есть такие данные)
        if 'days_since_last_sale' in data.columns:
            days_factor = np.exp(-data['days_since_last_sale'] / 100)  # Экспоненциальное затухание
            base_prediction = base_prediction * (0.7 + 0.3 * days_factor)  # Смешивание с базовым прогнозом
        
        # Обеспечиваем минимальный прогноз в 1 единицу
        return np.maximum(base_prediction, 1)
    
    def generate_recommendations(self, data, predictions):
        """Формирование рекомендаций по закупкам"""
        print("Формирование рекомендаций...")
        
        # Создаём DataFrame с результатами
        results = pd.DataFrame({
            'product_id': data['product_id'],
            'product_name': data['product_name'],
            'current_stock': data['current_stock'],
            'forecast': np.round(predictions, 0).astype(int)
        })
        
        # Расчёт рекомендаций (прогноз - текущий остаток)
        results['recommendation'] = results['forecast'] - results['current_stock']
        
        # Если рекомендация отрицательная, устанавливаем 0 (не нужно закупать)
        results['recommendation'] = results['recommendation'].apply(lambda x: max(0, x))
        
        return results
    
    def save_results_to_json(self, results, output_path='results.json'):
        """Сохранение результатов в JSON файл в указанном формате"""
        print(f"Сохранение результатов в {output_path}...")
        
        # Создаем простую структуру с только требуемыми полями
        simplified_results = pd.DataFrame({
            'Артикул': results['product_id'],
            'Название': results['product_name'],
            'Текущий остаток': results['current_stock'],
            'Прогноз': results['forecast'],
            'Рекомендации': results['recommendation']
        })
        
        # Преобразуем DataFrame в список словарей для JSON
        # Сохраняем ВСЕ товары, без ограничения на топ-5
        results_list = simplified_results.to_dict(orient='records')
        
        # Создаем директорию для сохранения результатов, если она не существует
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # Записываем JSON в файл
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_list, f, ensure_ascii=False, indent=4)
        
        print(f"Результаты успешно сохранены в {output_path}. Всего товаров: {len(results_list)}")
        return results_list
    
    def process_orders(self, file_path, output_path='results.json', save_model_path=None):
        """
        Полный процесс обработки заказов
        
        Параметры:
        file_path (str): Путь к CSV файлу с данными
        output_path (str): Путь для сохранения результатов в JSON
        save_model_path (str, optional): Путь для сохранения обученной модели
        """
        try:
            # Загружаем данные
            data = self.load_data(file_path)
            
            # Предобрабатываем данные
            processed_data = self.preprocess_data(data)
            
            # Подготавливаем признаки
            features = self.prepare_features(processed_data)
            
            # Обучаем модель
            self.train_model(features, save_model_path)
            
            # Получаем прогноз
            predictions = self.predict(features)
            
            # Формируем рекомендации
            results = self.generate_recommendations(features, predictions)
            
            # Сохраняем результаты в JSON
            results_json = self.save_results_to_json(results, output_path)
            
            print("Обработка завершена успешно!")
            return results_json
            
        except Exception as e:
            print(f"Ошибка при обработке: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

# Функция для создания тестового CSV файла если его нет
def create_test_csv(filepath="orders.csv"):
    """Создаем тестовый CSV файл для демонстрации работы InventoryBot"""
    if os.path.exists(filepath):
        print(f"Файл {filepath} уже существует, пропускаем создание тестового файла.")
        return
    
    print(f"Создаем тестовый файл {filepath} для демонстрации...")
    
    # Создаем случайные данные
    np.random.seed(42)
    
    # Список товаров
    products = [
        {"id": "A001", "name": "Футболка белая", "price": 999.90},
        {"id": "A002", "name": "Джинсы классические", "price": 2499.90},
        {"id": "A003", "name": "Кроссовки спортивные", "price": 3999.90},
        {"id": "A004", "name": "Куртка зимняя", "price": 5999.90},
        {"id": "A005", "name": "Шапка вязаная", "price": 599.90},
        {"id": "B001", "name": "Смартфон X-Phone", "price": 19999.90},
        {"id": "B002", "name": "Наушники беспроводные", "price": 2999.90},
        {"id": "B003", "name": "Планшет 10\"", "price": 12999.90},
        {"id": "C001", "name": "Книга 'Программирование на Python'", "price": 1200.00},
        {"id": "C002", "name": "Тетрадь школьная", "price": 59.90}
    ]
    
    # Генерируем даты за последние 6 месяцев
    from datetime import datetime, timedelta
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    dates = []
    
    current_date = start_date
    while current_date <= end_date:
        if np.random.random() > 0.5:  # Не для каждого дня будут заказы
            dates.append(current_date.strftime("%d.%m.%Y"))
        current_date += timedelta(days=1)
    
    # Генерируем случайные заказы
    rows = []
    for _ in range(500):  # 500 заказов
        product = products[np.random.randint(0, len(products))]
        date = dates[np.random.randint(0, len(dates))]
        quantity = np.random.randint(1, 10)
        weight = round(np.random.uniform(0.1, 5.0), 2)
        
        # Дата доставки через 1-7 дней после заказа
        order_date = datetime.strptime(date, "%d.%m.%Y")
        delivery_date = (order_date + timedelta(days=np.random.randint(1, 8))).strftime("%d.%m.%Y")
        
        # Итоговая стоимость с небольшими скидками
        final_price = round(product["price"] * quantity * np.random.uniform(0.9, 1.0), 2)
        
        rows.append({
            "Принят в обработку": date,
            "Артикул": product["id"],
            "OZON id": "OZ" + product["id"],
            "Наименование товара": product["name"],
            "Количество": quantity,
            "Итоговая стоимость товара": final_price,
            "Дата доставки": delivery_date,
            "Объемный вес товаров, кг": weight
        })
    
    # Создаем DataFrame и сохраняем в CSV
    df = pd.DataFrame(rows)
    df.to_csv(filepath, sep=";", index=False, encoding='utf-8')
    
    print(f"Тестовый файл {filepath} успешно создан с {len(df)} строками.")

# Пример использования
if __name__ == "__main__":
    # Создаем тестовый CSV файл, если его еще нет
    create_test_csv("orders.csv")
    
    # Создаем экземпляр бота с возможностью загрузки существующей модели
    # bot = InventoryBot("saved_model.pkl")  # Если есть сохраненная модель
    bot = InventoryBot()  # Если нет сохраненной модели
    
    # Путь к CSV файлу с данными
    input_file = "orders.csv"
    
    # Обрабатываем данные и сохраняем модель для дальнейшего использования
    results = bot.process_orders(
        file_path=input_file, 
        output_path="results.json",
        save_model_path="inventory_model.pkl"
    )
    
    # Можно также использовать модель для прогнозирования на новых данных
    # Например, если появились новые товары или изменились остатки
    # new_results = bot.process_orders("new_orders.csv", "new_results.json")
    
    # Пример доступа к результатам
    if results:
        print("\nПример результатов (первые 5 товаров):")
        for i, item in enumerate(results[:5]):
            print(f"{i+1}. {item['Название']}: текущий остаток = {item['Текущий остаток']}, прогноз = {item['Прогноз']}, рекомендации = {item['Рекомендации']}")
        
        print(f"\nВсего товаров обработано: {len(results)}")