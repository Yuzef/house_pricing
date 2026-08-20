# House Prices: прогнозирование стоимости недвижимости

Проект решает задачу регрессии из соревнования
[House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

Цель — предсказать стоимость продажи дома `SalePrice` по характеристикам недвижимости из Ames Housing Dataset.

Основная метрика — **RMSLE**. Чем меньше значение метрики, тем лучше модель.

## Основной результат

Лучший воспроизводимый результат получен с помощью `CatBoostRegressor`.
Таблица с результатами экспериментов в файле House_pricing final_table.md
(для корректного отображения лучше использовать Obsidian).

Метрика CatBoost рассчитана с помощью стратифицированной 5-fold
кросс-валидации. Для стратификации непрерывный таргет временно разбивается
на квантильные ценовые диапазоны.

Итоговый результат основного запуска сохраняется в:

trained_models/catboost_best_params/
├── config.yaml
├── cv_results.csv
├── experiment.log
├── metrics.json
├── model.joblib
└── submission.csv

## Данные

В проекте используется Ames Housing Dataset:

- обучающая выборка: 1460 объектов;
- тестовая выборка: 1459 объектов;
- 79 входных признаков;
- идентификатор объекта: `Id`;
- целевая переменная: `SalePrice`.

В данных присутствуют числовые, номинальные и порядковые признаки,
а также технические и структурные пропуски.

Строковое значение `NA` в ряде признаков означает физическое отсутствие
объекта, например гаража, подвала, бассейна или камина. Поэтому загрузка
CSV реализована отдельно и учитывает семантику таких значений.

## Используемый pipeline

Основной pipeline включает:

1. Загрузку train и test.
2. Отделение `Id` и `SalePrice`.
3. Создание дополнительных признаков.
4. Заполнение пропущенных значений.
5. Кодирование категориальных признаков.
6. Масштабирование числовых признаков.
7. Обучение модели на `log1p(SalePrice)`.
8. Оценку качества с помощью 5-fold CV.
9. Обучение финальной модели на всей обучающей выборке.
10. Создание `submission.csv`.

Экспериментальные Feature Engineering-признаки:

- возраст дома и гаража;
- агрегированные площади;
- суммарное количество ванных комнат;
- взаимодействие общего качества и жилой площади.

Каждая группа признаков включается и отключается через `config.py`.


## Запуск основного эксперимента

Из корня репозитория выполните:

```bash
python main.py
```

По умолчанию запускается CatBoost с зафиксированными гиперпараметрами.

После завершения будут созданы:

- результаты каждого CV-фолда;
- средняя метрика и стандартное отклонение;
- обученная модель;
- конфигурация эксперимента;
- журнал выполнения;
- Kaggle submission.

Путь к результатам определяется полем:

```python
config.general.experiment_name
```

## Настройка эксперимента

Основные параметры находятся в `config.py`.

Выбор модели:

```python
"model": {
    "name": "catboost",
    "type": "catboost",
    "params": {
        ...
    },
}
```

Доступные классические модели:

- Linear Regression;
- Ridge;
- Lasso;
- ElasticNet;
- KNN;
- Decision Tree;
- Random Forest;
- CatBoost;
- LightGBM;
- XGBoost.

Также реализованы:

- Voting ensemble;
- Stacking ensemble;
- PyTorch MLP;
- MLP с категориальными embeddings;
- подбор DL-гиперпараметров с помощью Optuna;
- nested cross-validation для DL.

Управление Feature Engineering:

```python
"feature_engineering": {
    "age_features": {
        "enabled": False,
    },
    "area_features": {
        "enabled": False,
    },
    "bathroom_features": {
        "enabled": False,
    },
    "quality_area_interaction": {
        "enabled": True,
    },
},
```

Включение подбора гиперпараметров:

```python
"tuning": {
    "enabled": True,
    ...
}
```

Подбор параметров заметно увеличивает время выполнения.

## EDA

Исследовательский анализ находится в ноутбуке:

[`EDA_and_Feature_eng_exploration.ipynb`](EDA_and_Feature_eng_exploration.ipynb)


В EDA рассмотрены:

- распределение целевой переменной;
- пропуски и структурное отсутствие объектов;
- числовые и категориальные признаки;
- связь качества, площади, возраста и района с ценой;
- потенциальные выбросы;
- мультиколлинеарность;
- гипотезы для Feature Engineering;
- защита от утечки целевой переменной.

## Структура проекта

```text
.
├── config.py                  # Конфигурация эксперимента
├── main.py                    # Основной запускаемый pipeline
├── requirements.txt           # Зависимости
├── README.md
├── data_raw/
│   ├── train.csv
│   ├── test.csv
│   └── data_description.txt
├── utils/
│   ├── load_data.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   ├── validation.py
│   ├── model_selection.py
│   ├── inference.py
│   ├── experiment_artifacts.py
│   └── experiment_logging.py
├── DL/
│   ├── data.py
│   ├── trainer.py
│   ├── tuning.py
│   ├── nested_cv.py
│   ├── workflow.py
│   ├── inference.py
│   └── dl_models/
├── trained_models/            # Метрики, модели и сабмиты
└── EDA_and_Feature_eng_exploration.ipynb
```

## Воспроизводимость

Для воспроизводимости используются:

- фиксированный `random_state`;
- сохранение полной конфигурации каждого эксперимента;
- сохранение CV-результатов;
- сохранение параметров лучших моделей;
- фиксированные версии зависимостей;
- единый preprocessing внутри sklearn Pipeline;
- разделение preprocessing по CV-фолдам для защиты от data leakage.

## Совместимость сохранённых DL-моделей

Эксперименты 19–22 представлены как исторические результаты развития
архитектуры MLP. Их метрики, конфигурации и submissions сохранены, однако
checkpoints относятся к предыдущей версии архитектуры и не предназначены
для загрузки текущим кодом.

Актуальные воспроизводимые DL checkpoints представлены в экспериментах
23–26.