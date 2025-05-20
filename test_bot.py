import os
import pandas as pd
import json
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import sys
from datetime import datetime

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("inventory_api")

# Импортируем класс InventoryBot из первого файла
# Путь импорта может меняться в зависимости от структуры проекта
try:
    from inventory_bot import InventoryBot
    logger.info("InventoryBot успешно импортирован.")
except ImportError as e:
    logger.error(f"Ошибка импорта InventoryBot: {e}")
    # Здесь бы выводилось сообщение об ошибке, но мы знаем, что у вас есть файл inventory_bot.py

# Создаем экземпляр FastAPI
app = FastAPI(
    title="Inventory Prediction API",
    description="API для прогнозирования закупок и управления запасами на основе исторических данных",
    version="1.0.0"
)

# Добавляем CORS для возможности вызова API из браузера
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене лучше указать конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Модели данных для запросов и ответов
class InventoryRecommendation(BaseModel):
    Артикул: str
    Название: str
    Текущий_остаток: int
    Прогноз: int
    Рекомендации: int
    Приоритет: Optional[str] = None
    Исторические_продажи: Optional[int] = None
    Дней_с_последней_продажи: Optional[int] = None

class InventoryResponse(BaseModel):
    results: List[Dict[str, Any]]
    summary: Dict[str, Any]
    status: str
    timestamp: str

class CSVData(BaseModel):
    content: str

@app.post("/analyze/file", response_model=InventoryResponse)
async def analyze_inventory_file(file: UploadFile = File(...)):
    """
    Загрузите CSV файл с данными заказов для анализа.
    
    Файл должен быть в формате CSV с разделителем ';' и кодировкой UTF-8.
    Необходимые колонки: 'Принят в обработку', 'Артикул', 'Наименование товара', 'Количество'
    """
    # Проверяем расширение файла
    
    try:
        logger.info(f"Начало обработки файла {file.filename}")
        
        # Создаем временный файл для сохранения загруженного CSV
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_csv:
            temp_csv_path = temp_csv.name
            # Копируем содержимое загруженного файла во временный файл
            contents = await file.read()
            temp_csv.write(contents)
        
        logger.info(f"Файл сохранен во временный файл {temp_csv_path}")
        
        # Создаем временный файл для сохранения результатов в JSON
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp_json:
            temp_json_path = temp_json.name
        
        logger.info("Запуск анализа данных с помощью InventoryBot")
        
        # Создаем экземпляр бота и запускаем анализ
        bot = InventoryBot()
        results = bot.process_orders(
            file_path=temp_csv_path, 
            output_path=temp_json_path,
            save_model_path=None  # Можете указать путь для сохранения модели, если нужно
        )
        
        # Если результаты не возвращены напрямую из бота, читаем их из файла
        if not results:
            logger.info(f"Чтение результатов из файла {temp_json_path}")
            with open(temp_json_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
        
        logger.info(f"Обработано {len(results)} товаров")
        
        # Добавляем приоритеты товаров на основе рекомендаций и текущих остатков
        for item in results:
            current_stock = item.get('Текущий остаток', 0)
            recommendation = item.get('Рекомендации', 0)
            
            if current_stock <= 5 and recommendation > 0:
                item['Приоритет'] = 'Высокий'
            elif current_stock <= 10 and recommendation > 0:
                item['Приоритет'] = 'Средний'
            else:
                item['Приоритет'] = 'Низкий'
        
        # Формируем общую статистику
        high_priority = len([item for item in results if item.get('Приоритет') == 'Высокий'])
        medium_priority = len([item for item in results if item.get('Приоритет') == 'Средний'])
        total_recommendations = sum(item.get('Рекомендации', 0) for item in results)
        
        summary = {
            "всего_товаров": len(results),
            "товаров_высокого_приоритета": high_priority,
            "товаров_среднего_приоритета": medium_priority,
            "товаров_низкого_приоритета": len(results) - high_priority - medium_priority,
            "общее_количество_к_закупке": total_recommendations
        }
        
        # Возвращаем результаты анализа
        return JSONResponse(content={
    "results": results, 
    "summary": summary,
    "status": "success",
    "timestamp": datetime.now().isoformat()
})

    
    except Exception as e:
        logger.error(f"Ошибка при обработке файла: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке файла: {str(e)}")
    
    finally:
        # Удаляем временные файлы
        try:
            if 'temp_csv_path' in locals() and os.path.exists(temp_csv_path):
                os.unlink(temp_csv_path)
                logger.info(f"Временный CSV-файл удален: {temp_csv_path}")
                
            if 'temp_json_path' in locals() and os.path.exists(temp_json_path):
                os.unlink(temp_json_path)
                logger.info(f"Временный JSON-файл удален: {temp_json_path}")
        except Exception as e:
            logger.warning(f"Ошибка при удалении временных файлов: {str(e)}")

@app.post("/analyze/data", response_model=InventoryResponse)
async def analyze_inventory_data(csv_data: CSVData):
    """
    Отправьте содержимое CSV файла в виде строки для анализа.
    
    Данные должны быть в формате CSV с разделителем ';' и кодировкой UTF-8.
    Необходимые колонки: 'Принят в обработку', 'Артикул', 'Наименование товара', 'Количество'
    """
    try:
        logger.info("Начало обработки CSV данных отправленных напрямую")
        
        # Создаем временный файл для сохранения CSV данных
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='w', encoding='utf-8') as temp_csv:
            temp_csv_path = temp_csv.name
            temp_csv.write(csv_data.content)
        
        logger.info(f"CSV данные сохранены во временный файл {temp_csv_path}")
        
        # Создаем временный файл для сохранения результатов в JSON
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp_json:
            temp_json_path = temp_json.name
        
        logger.info("Запуск анализа данных с помощью InventoryBot")
        
        # Создаем экземпляр бота и запускаем анализ
        bot = InventoryBot()
        results = bot.process_orders(
            file_path=temp_csv_path, 
            output_path=temp_json_path
        )
        
        # Если результаты не возвращены напрямую из бота, читаем их из файла
        if not results:
            logger.info(f"Чтение результатов из файла {temp_json_path}")
            with open(temp_json_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
        
        logger.info(f"Обработано {len(results)} товаров")
        
        # Добавляем приоритеты товаров на основе рекомендаций и текущих остатков
        for item in results:
            current_stock = item.get('Текущий остаток', 0)
            recommendation = item.get('Рекомендации', 0)
            
            if current_stock <= 5 and recommendation > 0:
                item['Приоритет'] = 'Высокий'
            elif current_stock <= 10 and recommendation > 0:
                item['Приоритет'] = 'Средний'
            else:
                item['Приоритет'] = 'Низкий'
        
        # Формируем общую статистику
        high_priority = len([item for item in results if item.get('Приоритет') == 'Высокий'])
        medium_priority = len([item for item in results if item.get('Приоритет') == 'Средний'])
        total_recommendations = sum(item.get('Рекомендации', 0) for item in results)
        
        summary = {
            "всего_товаров": len(results),
            "товаров_высокого_приоритета": high_priority,
            "товаров_среднего_приоритета": medium_priority,
            "товаров_низкого_приоритета": len(results) - high_priority - medium_priority,
            "общее_количество_к_закупке": total_recommendations
        }
        
        # Возвращаем результаты анализа
        return {
            "results": results, 
            "summary": summary,
            "status": "success",
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Ошибка при обработке CSV данных: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке CSV данных: {str(e)}")
    
    finally:
        # Удаляем временные файлы
        try:
            if 'temp_csv_path' in locals() and os.path.exists(temp_csv_path):
                os.unlink(temp_csv_path)
                logger.info(f"Временный CSV-файл удален: {temp_csv_path}")
                
            if 'temp_json_path' in locals() and os.path.exists(temp_json_path):
                os.unlink(temp_json_path)
                logger.info(f"Временный JSON-файл удален: {temp_json_path}")
        except Exception as e:
            logger.warning(f"Ошибка при удалении временных файлов: {str(e)}")

@app.get("/")
async def root():
    """Корневой эндпоинт с информацией о API"""
    return {
        "название": "API системы прогнозирования и рекомендаций по закупкам",
        "версия": "1.0.0",
        "эндпоинты": {
            "/analyze/file": "Отправьте POST запрос с CSV файлом",
            "/analyze/data": "Отправьте POST запрос с содержимым CSV в теле запроса"
        },
        "документация": "/docs"
    }

# Запуск приложения с помощью Uvicorn
if __name__ == "__main__":
    logger.info("Запуск API-сервера")
    uvicorn.run("test_bot:app", host="0.0.0.0", port=8000, reload=True)