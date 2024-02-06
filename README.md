# Название проекта: Построение ML-продукта для оптимизации классификации заявок на оплату для сервиса Repetit.ru

## Заказчик: Сервис по подбору репетиторов Repetit.ru

## Статус проекта: в работе

## Описание проекта
<br>Сервис Repetit.ru работает с большим количеством заявок от клиентов с данными о предмете, 
<br>желаемой стоимости, возрасте ученика, целью занятий и тд. 
<br>К сожалению, 7 из 8 не доходят до оплаты, при этом обработка заявки консультантом увеличивает конверсию в оплату на 30%. 
<br>Проблема в том, что консультантов не хватает на все заявки и получается, что чем больше заявок — 
<br>тем меньше конверсия из заявки в оплату и консультанты тратят время на бесперспективные заявки.

## Цель проекта
- Разработать модель, которая по имеющейся информации о клиенте и заявке будет предсказывать вероятность оплаты заявки клиентом. 
- Заказчик хочет понять, какие заявки будут оплачены, а какие нет, чтобы одни обрабатывать вручную консультантами, а другие нет. 

## Задачи:
- Отбор подходящих признаков,
- Решение проблемы дубликатов,
- Решение проблемы постоянных изменений в данных,
- Создание сервиса в виде Docker Container.
<br>Оценка качества модели будет производиться с использованием precision и ROC-AUC.

## Ход исследования
- загрузка данных и ознакомление с ними,
- отбор подходящих признаков,
- EDA,
- создание новых признаков (при необходимости),
- отбор финального набора обучающих признаков,
- выбор и обучение моделей (разных архитектур),
- оценка качества предсказания лучшей модели на тестовой выборке,
- анализ важности признаков лучшей модели,
- создание сервиса в виде Docker Container,
- отчёт по проведённому исследованию.

## Используемые инструменты
- Python
- Docker
- shap
- phik
- torch
- pickle
- numpy
- pandas
- seaborn
- lightgbm
- matplotlib
- scikit-learn
- catboost
- flask

## Заключение:
<br>Отчёт по выполненным в ходе исследования шагам.
1. Загрузка и первичная обработка данных
  - Основной таблицей исследования принята таблица orders 
  - из неё удалены явные повторы строк
  - обработаны дубликаты заявок
  - преобразованы типы данных по необходимости
  - созданы дополнительные признаки, в том числе и целевой
  - проанализирован каждый признак
  - заполнены пропущенные значения при небходимости
  - после всех преобразований в таблице осталось 20 признаков, количество строк сократилось практически в два раза
  - Таблица с данными об учителях проанализирована на предмет дубликатов.
  - Остальные таблицы также проверены на дубликаты и пропуски.
  - По итогам данного шага составлен небольшой отчёт по преобразованиям и наблюдениям.
2. Подготовка данных
  - Данные собраны в единый датафрейм с основной таблицей.
  - В нём создан дополнительный признак.
  - Изучены корреляционные связи признаков.
  - Произведено кодирование текстов в эмбеддинги.
  - Произведено масштабирование признаков.
  - Данные разделены на три выборки с учётом дисбаланса классов:
        + обучающую (72% от полного набора данных)
        + валидационную (18% от полного набора) данных
        + тестовую (10%)
3. Обучение моделей
  - Обучены три модели:
  - в качестве baseline использована логистическая регрессия
  - также использованы две модели градиентного бустинга: CatBoost, LightGBM
  - значения метрик на валидационной выборке:

|model|roc-auc|precision|f1_score|
|---:|---:|---:|---:|
|Catboost|0.848150|0.909625|0.411341|
|LightGBM|0.704034|0.911585|0.009630|
|Baseline|0.632782|0.363636|0.003352|

  - Лучший результат по ROC-AUC показала модель CatBoost
  - Тестирование лучшей модели
  - Проверка модели на тестовой выборке также показала удовлетворительные результаты: ROC-AUC = 0.85, Precision = 0.91
  - Матрица ошибок указывает на то, что модель с высоким качеством определяет класс 0, при определении класса 1 высокий показатель ошибок;
  - Наиболее важными признаками для модели выявлены: `purpose`, `contact_resul`, `source_id`
<br>По итогам исследования можно сделать заключение, что наиболее подходящей моделью для решения поставленной задачи Заказчика является градиентный бустинг CatBoost
4. Создан микросервис в виде Docker container
