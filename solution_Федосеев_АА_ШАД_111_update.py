# -*- coding: utf-8 -*-
# 1. Принудительная установка фонового бэкенда на самой первой строчке без загрузки GUI
import os
os.environ['MPLBACKEND'] = 'Agg'

import sys
import glob
import unittest
import argparse  # Импортируем на верхнем уровне для безопасности
import pandas as pd
import numpy as np

# 2. Безопасный импорт matplotlib и принудительная активация неинтерактивного режима
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns

# Настройка стилей для визуализации
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.figsize'] = (10, 6)

def load_data(input_path):
    """
    Загружает данные из parquet-файла или директории с parquet-файлами.
    Поддерживает рекурсивный поиск и партиционированные пути.
    """
    if os.path.isfile(input_path):
        if input_path.endswith('.parquet'):
            return pd.read_parquet(input_path)
        elif input_path.endswith('.csv'):
            return pd.read_csv(input_path)
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {input_path}")
            
    # Если передан путь к папке, рекурсивно ищем parquet-файлы
    parquet_files = glob.glob(os.path.join(input_path, '**/*.parquet'), recursive=True)
    if not parquet_files:
        # Попытка прочесть директорию напрямую через pandas
        try:
            return pd.read_parquet(input_path)
        except Exception as e:
            raise FileNotFoundError(f"Файлы parquet не найдены в {input_path}. Ошибка: {e}")
            
    dfs = []
    for f in parquet_files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            print(f"Предупреждение: не удалось прочесть файл {f}. Ошибка: {e}")
            
    if not dfs:
        raise ValueError(f"Не удалось загрузить ни один файл из {input_path}")
        
    return pd.concat(dfs, ignore_index=True)

def preprocess_data(df):
    """
    Переименовывает колонки, приводит типы к корректным.
    Преобразует типы Decimal в float для быстрой векторной арифметики.
    """
    df = df.copy()
    if 'CategoryNameDelivery' in df.columns:
        df = df.rename(columns={'CategoryNameDelivery': 'CategoryDelivery'})
    
    # Приводим типы данных
    if 'researchdate' in df.columns:
        df['researchdate'] = pd.to_datetime(df['researchdate']).dt.date
    if 'Start' in df.columns:
        df['Start'] = pd.to_datetime(df['Start'])
        
    # Преобразуем Decimal-колонки в стандартный float для корректной работы математических операций
    for col in ['Weight', 'week_weight', 'month_weight']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
            
    return df

def add_stats(df, group_cols, suffix):
    """
    Рассчитывает медиану и MAD на определенном уровне агрегации 
    с применением логарифмического сглаживания и MAD-smoothing (Улучшения Б и В).
    """
    stats = df.groupby(group_cols).agg(
        n_uniq=('SubjectID', 'nunique'),
        median=('daily_ots_log', 'median')
    ).reset_index()
    
    df_temp = df.merge(stats, on=group_cols, how='left')
    df_temp['ad'] = (df_temp['daily_ots_log'] - df_temp['median']).abs()
    
    mad = df_temp.groupby(group_cols).agg(
        mad=('ad', 'median')
    ).reset_index()
    
    stats = stats.merge(mad, on=group_cols, how='left')
    
    # Улучшение Б (MAD Smoothing): исключаем равенство MAD нулю в однородных группах
    epsilon = 1e-4
    stats['mad'] = np.where(stats['mad'] == 0, epsilon * stats['median'].clip(lower=1.0), stats['mad'])
    stats['mad'] = stats['mad'].clip(lower=1e-5)
    
    # Переименовываем столбцы для объединения
    stats = stats.rename(columns={
        'n_uniq': f'n_uniq_{suffix}',
        'median': f'median_{suffix}',
        'mad': f'mad_{suffix}'
    })
    return df.merge(stats, on=group_cols, how='left')

def detect_anomalies(df):
    """
    Основной алгоритм поиска аномалий на основе Modified Z-score с fallback-иерархией.
    """
    # Фильтруем строки по условиям BrandinDelivery == 1.0 и непустой CategoryDelivery
    df_filtered = df[
        (df['BrandinDelivery'] == 1.0) & 
        (df['CategoryDelivery'].notna()) & 
        (df['CategoryDelivery'] != '')
    ].copy()
    
    if df_filtered.empty:
        print("Предупреждение: нет подходящих строк для анализа (BrandinDelivery == 1.0 и заполненная категория).")
        return pd.DataFrame(), pd.DataFrame()
        
    # Считаем count_rows и daily_ots
    group_cols = ['SubjectID', 'researchdate', 'CategoryDelivery', 'BrandID']
    df_agg = df_filtered.groupby(group_cols).agg(
        count_rows=('SubjectID', 'count'),
        Weight=('Weight', 'first'),
        Brand=('Brand', 'first')
    ).reset_index()
    
    df_agg['daily_ots'] = df_agg['Weight'] * df_agg['count_rows']
    
    # Улучшение В (Логарифмирование): Сглаживание правосторонней асимметрии данных
    df_agg['daily_ots_log'] = np.log1p(df_agg['daily_ots'])
    
    # Расчет статистик на 3 уровнях иерархии
    # Уровень 1: День + Категория + Бренд
    df_agg = add_stats(df_agg, ['researchdate', 'CategoryDelivery', 'BrandID'], 'lvl1')
    # Уровень 2: День + Категория
    df_agg = add_stats(df_agg, ['researchdate', 'CategoryDelivery'], 'lvl2')
    # Уровень 3: День
    df_agg = add_stats(df_agg, ['researchdate'], 'lvl3')
    
    # Условия выбора уровня для расчета score (fallback при малых выборках)
    cond_lvl1 = (df_agg['n_uniq_lvl1'] >= 5) & (df_agg['mad_lvl1'] > 0)
    cond_lvl2 = (~cond_lvl1) & (df_agg['n_uniq_lvl2'] >= 5) & (df_agg['mad_lvl2'] > 0)
    cond_lvl3 = (~cond_lvl1) & (~cond_lvl2) & (df_agg['n_uniq_lvl3'] >= 5) & (df_agg['mad_lvl3'] > 0)
    
    # Рассчитываем Modified Z-score и порог для каждого уровня (в логарифмической шкале)
    score_lvl1 = 0.6745 * (df_agg['daily_ots_log'] - df_agg['median_lvl1']) / df_agg['mad_lvl1']
    thresh_lvl1_log = df_agg['median_lvl1'] + 3.5 * df_agg['mad_lvl1'] / 0.6745
    
    score_lvl2 = 0.6745 * (df_agg['daily_ots_log'] - df_agg['median_lvl2']) / df_agg['mad_lvl2']
    thresh_lvl2_log = df_agg['median_lvl2'] + 3.5 * df_agg['mad_lvl2'] / 0.6745
    
    score_lvl3 = 0.6745 * (df_agg['daily_ots_log'] - df_agg['median_lvl3']) / df_agg['mad_lvl3']
    thresh_lvl3_log = df_agg['median_lvl3'] + 3.5 * df_agg['mad_lvl3'] / 0.6745
    
    # Объединяем результаты по условиям
    df_agg['score'] = np.select([cond_lvl1, cond_lvl2, cond_lvl3], [score_lvl1, score_lvl2, score_lvl3], default=np.nan)
    df_agg['threshold_log'] = np.select([cond_lvl1, cond_lvl2, cond_lvl3], [thresh_lvl1_log, thresh_lvl2_log, thresh_lvl3_log], default=np.nan)
    
    # Обратное преобразование порога в исходную шкалу OTS для интерпретируемости в отчетах
    df_agg['threshold'] = np.expm1(df_agg['threshold_log'])
    df_agg['level'] = np.select([cond_lvl1, cond_lvl2, cond_lvl3], ['Brand', 'Category', 'Date'], default='None')
    
    # Флаг аномалии: score > 3.5
    df_agg['is_anomaly'] = (df_agg['score'] > 3.5) & (df_agg['level'] != 'None')
    
    # Создаем детальные причины для аномалий
    anomalies = df_agg[df_agg['is_anomaly']].copy()
    if not anomalies.empty:
        def get_reason(row):
            med_log = row['median_lvl1'] if row['level']=='Brand' else (row['median_lvl2'] if row['level']=='Category' else row['median_lvl3'])
            med_raw = np.expm1(med_log)
            return (f"Выброс на уровне '{row['level']}': "
                    f"Z-score = {row['score']:.2f} > 3.5 "
                    f"(daily_ots = {row['daily_ots']:.2f}, "
                    f"медиана = {med_raw:.2f}, "
                    f"порог = {row['threshold']:.2f})")
            
        anomalies['reason'] = anomalies.apply(get_reason, axis=1)
    else:
        anomalies['reason'] = []
        
    return df_agg, anomalies

def build_plots(df, df_agg, anomalies, output_dir='output/plots'):
    """
    Создает три обязательных диагностических графика с гарантированной совместимостью версий.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Набор пар респондент-дата для удаления
    anom_pairs = set(zip(anomalies['SubjectID'], anomalies['researchdate']))
    
    # 1. total_ots_before_after.png
    df_agg['is_removed'] = df_agg.apply(lambda r: (r['SubjectID'], r['researchdate']) in anom_pairs, axis=1)
    
    ots_by_day_before = df_agg.groupby('researchdate')['daily_ots'].sum().sort_index()
    ots_by_day_after = df_agg[~df_agg['is_removed']].groupby('researchdate')['daily_ots'].sum().sort_index()
    
    plt.figure(figsize=(12, 6))
    plt.plot(ots_by_day_before.index, ots_by_day_before.values, label='До очистки', color='#1f77b4', marker='o', linewidth=2)
    plt.plot(ots_by_day_after.index, ots_by_day_after.values, label='После очистки', color='#2ca02c', marker='x', linewidth=2)
    plt.title('Суммарный OTS по дням до и после удаления аномалий', fontsize=14)
    plt.xlabel('Дата', fontsize=12)
    plt.ylabel('Суммарный OTS', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'total_ots_before_after.png'), dpi=150)
    plt.close()
    
    # 2. category_ots_change.png (Переписано на чистый Matplotlib для 100% совместимости на сервере)
    ots_cat_before = df_agg.groupby('CategoryDelivery')['daily_ots'].sum()
    ots_cat_after = df_agg[~df_agg['is_removed']].groupby('CategoryDelivery')['daily_ots'].sum()
    
    cat_change_pct = ((ots_cat_after - ots_cat_before) / ots_cat_before * 100).fillna(0).sort_values()
    
    plt.figure(figsize=(10, 6))
    colors = ['#d62728' if x < -5 else '#ff7f0e' for x in cat_change_pct.values]
    plt.barh(cat_change_pct.index, cat_change_pct.values, color=colors, edgecolor='black', alpha=0.8)
    plt.title('Изменение OTS по CategoryDelivery после очистки (%)', fontsize=14)
    plt.xlabel('Изменение (%)', fontsize=12)
    plt.ylabel('Категория доставки', fontsize=12)
    plt.grid(True, axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'category_ots_change.png'), dpi=150)
    plt.close()
    
    # 3. daily_anomaly_count.png (Исправлено приведение типов дат для старых версий Pandas)
    anom_count_by_day = anomalies.groupby('researchdate')['SubjectID'].nunique().sort_index()
    start_dt = pd.to_datetime(df_agg['researchdate'].min())
    end_dt = pd.to_datetime(df_agg['researchdate'].max())
    all_dates = pd.date_range(start=start_dt, end=end_dt).date
    anom_count_by_day = anom_count_by_day.reindex(all_dates, fill_value=0)
    
    plt.figure(figsize=(12, 6))
    plt.bar(anom_count_by_day.index, anom_count_by_day.values, color='#e377c2', edgecolor='black', alpha=0.8)
    plt.title('Количество уникальных аномальных респондентов по дням', fontsize=14)
    plt.xlabel('Дата', fontsize=12)
    plt.ylabel('Количество респондентов', fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'daily_anomaly_count.png'), dpi=150)
    plt.close()

def print_quality_metrics(df, df_agg, anomalies):
    """
    Вычисляет и выводит в консоль ключевые метрики качества очистки.
    """
    total_reps = df['SubjectID'].nunique()
    anom_reps = anomalies['SubjectID'].nunique() if not anomalies.empty else 0
    share_anom_reps = anom_reps / total_reps if total_reps > 0 else 0
    
    total_ots_before = df_agg['daily_ots'].sum()
    anom_pairs = set(zip(anomalies['SubjectID'], anomalies['researchdate'])) if not anomalies.empty else set()
    df_agg['is_removed'] = df_agg.apply(lambda r: (r['SubjectID'], r['researchdate']) in anom_pairs, axis=1)
    total_ots_after = df_agg[~df_agg['is_removed']]['daily_ots'].sum()
    ots_kept_ratio = total_ots_after / total_ots_before if total_ots_before > 0 else 1
    
    anoms_per_day = anomalies.groupby('researchdate')['SubjectID'].nunique().mean() if not anomalies.empty else 0
    
    print("\n" + "="*50)
    print("          ОТЧЕТ О КАЧЕСТВЕ ОЧИСТКИ ДАННЫХ")
    print("" + "="*50)
    print(f"Общее число уникальных респондентов:  {total_reps}")
    print(f"Выявлено аномальных респондентов:     {anom_reps} ({share_anom_reps:.2%})")
    print(f"Среднее число аномалий в день:        {anoms_per_day:.2f}")
    print(f"Доля сохраненного OTS после очистки:   {ots_kept_ratio:.2%}")
    print("="*50 + "\n")

# --- ДОПОЛНИТЕЛЬНЫЕ АНАЛИТИЧЕСКИЕ ВОЗМОЖНОСТИ ---

def plot_before_after_by_feature(df, anomalies, feature_name, title, output_file):
    """
    Строит график сравнения до/после для любой характеристики респондента или ресурса.
    """
    anom_pairs = set(zip(anomalies['SubjectID'], anomalies['researchdate'])) if not anomalies.empty else set()
    df_clean = df[(df['BrandinDelivery'] == 1.0) & (df['CategoryDelivery'].notna())].copy()
    df_clean['is_removed'] = df_clean.apply(lambda r: (r['SubjectID'], r['researchdate']) in anom_pairs, axis=1)
    
    ots_before = df_clean.groupby(feature_name)['Weight'].sum().reset_index(name='OTS до')
    ots_after = df_clean[~df_clean['is_removed']].groupby(feature_name)['Weight'].sum().reset_index(name='OTS после')
    
    merged = pd.merge(ots_before, ots_after, on=feature_name, how='left').fillna(0)
    melted = pd.melt(merged, id_vars=[feature_name], value_vars=['OTS до', 'OTS после'], 
                     var_name='Статус', value_name='Суммарный OTS')
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=melted, x=feature_name, y='Суммарный OTS', hue='Статус', palette='coolwarm')
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()

def get_query_text_table(df, subject_id, date):
    """
    Возвращает таблицу поисковых запросов для выбранного респондента и дня.
    """
    subset = df[(df['SubjectID'] == subject_id) & (df['researchdate'] == date)]
    return subset[['SubjectID', 'researchdate', 'QueryText', 'Brand', 'CategoryDelivery', 'Weight']]

def plot_brand_ots_chart(df, anomalies, brand_id, output_file):
    """
    Строит график изменения OTS по дням для выбранного бренда до и после очистки.
    """
    anom_pairs = set(zip(anomalies['SubjectID'], anomalies['researchdate'])) if not anomalies.empty else set()
    df_brand = df[df['BrandID'] == brand_id].copy()
    df_brand['is_removed'] = df_brand.apply(lambda r: (r['SubjectID'], r['researchdate']) in anom_pairs, axis=1)
    
    ots_before = df_brand.groupby('researchdate')['Weight'].sum().sort_index()
    ots_after = df_brand[~df_brand['is_removed']].groupby('researchdate')['Weight'].sum().sort_index()
    
    plt.figure(figsize=(12, 6))
    plt.plot(ots_before.index, ots_before.values, label='До очистки', color='blue', marker='o')
    plt.plot(ots_after.index, ots_after.values, label='После очистки', color='green', marker='x')
    plt.title(f'Динамика OTS для бренда ID: {brand_id} до и после очистки', fontsize=14)
    plt.xlabel('Дата')
    plt.ylabel('Суммарный OTS')
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()

# --- УЛУЧШЕНИЕ Г: МОДУЛЬНОЕ ТЕСТИРОВАНИЕ ---

class TestAnomalyDetection(unittest.TestCase):
    """
    Класс для проведения автоматического модульного тестирования основных функций скрипта.
    Устранен хардкод конкретных дат (генерация динамическая).
    """
    def setUp(self):
        # Создание небольшой контролируемой тестовой выборки с динамической датой
        today_date = pd.Timestamp.now().date()
        self.test_df = pd.DataFrame({
            'SubjectID': [1, 2, 3, 4, 5, 6, 7],
            'QueryText': ['тест'] * 7,
            'BrandID': ['B1'] * 7,
            'Brand': ['Brand1'] * 7,
            'CategoryDelivery': ['Гаджеты'] * 7,
            'Weight': [10.0] * 6 + [1000.0],  # Респондент 7 имеет аномальный вес
            'BrandinDelivery': [1.0] * 7,
            'researchdate': [today_date] * 7
        })

    def test_preprocessing(self):
        processed = preprocess_data(self.test_df)
        self.assertEqual(processed['Weight'].dtype, float)
        self.assertIn('CategoryDelivery', processed.columns)

    def test_anomaly_isolation(self):
        processed = preprocess_data(self.test_df)
        df_agg, anomalies = detect_anomalies(processed)
        # Аномальный респондент 7 должен быть корректно изолирован
        self.assertIn(7, anomalies['SubjectID'].values)
        # Стабильные респонденты 1-6 не должны помечаться как аномалии
        self.assertNotIn(1, anomalies['SubjectID'].values)

    def test_brand_delivery_filtering(self):
        df_filter_test = self.test_df.copy()
        # Если респондент исключен из поставки (BrandinDelivery == 0), он не должен детектироваться
        df_filter_test.loc[df_filter_test['SubjectID'] == 7, 'BrandinDelivery'] = 0.0
        processed = preprocess_data(df_filter_test)
        df_agg, anomalies = detect_anomalies(processed)
        self.assertTrue(anomalies.empty or (7 not in anomalies['SubjectID'].values))

def main():
    # Настройка схемы проверки через командную строку (argparse)
    parser = argparse.ArgumentParser(description="Робастный поиск аномальных респондентов SoS и аналитические возможности.")
    parser.add_argument('input_path', nargs='?', default=None, help='Путь к папке или файлу с данными')
    parser.add_argument('--test', action='store_true', help='Запустить автоматическое модульное тестирование (Улучшение Г)')
    
    # Консольная схема проверки дополнительных аналитических возможностей (Раздел 8.2)
    parser.add_argument('--query-text', nargs=2, metavar=('SubjectID', 'Date'),
                        help='Вывести таблицу поисковых запросов для конкретного SubjectID и даты (в формате ГГГГ-ММ-ДД)')
    parser.add_argument('--brand-plot', metavar='BrandID',
                        help='Построить и сохранить индивидуальный график OTS для выбранного BrandID')
    parser.add_argument('--feature-plot', metavar='FeatureName',
                        help='Построить и сохранить график OTS до/после по любой характеристике (Регион, Пол, Platform)')

    args = parser.parse_args()

    if args.test:
        print("Запуск модульного тестирования...")
        sys.argv = [sys.argv[0]]
        unittest.main()
        sys.exit(0)

    # 1. Определение пути к данным
    input_path = args.input_path
    if input_path is None:
        month_folders = glob.glob('month=*')
        input_path = month_folders[0] if month_folders else '.'
            
    print(f"Поиск данных в: {input_path}")
    
    # 2. Загрузка данных
    try:
        df = load_data(input_path)
        print(f"Данные загружены. Строк: {len(df)}")
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        print("Генерация синтетических данных для демонстрации работы скрипта...")
        # Убран хардкод конкретных дат (генерация динамическая)
        n = 50000
        today = pd.Timestamp.now().floor('D')
        dates = pd.date_range(end=today, periods=30)
        df = pd.DataFrame({
            'SubjectID': np.random.choice(range(1001, 1500), n),
            'QueryText': np.random.choice(['купить телефон', 'цена на платье', 'кроссовки nike', 'коллаген тональный'], n),
            'BrandID': np.random.choice([f'B{i}' for i in range(1, 101)], n),
            'Category1ID': '1', 'Category2ID': '10', 'Category3ID': '100',
            'Brand': np.random.choice(['Samsung', 'Nike', 'Adidas', 'LOreal', 'Xiaomi'], n),
            'Category1': 'Гаджеты', 'Category2': 'Смартфоны', 'Category3': 'Телефоны',
            'CategoryDelivery': np.random.choice(['Смартфоны', 'Обувь', 'Косметика', 'Одежда'], n),
            'ResourceName': np.random.choice(['Ozon', 'Wildberries', 'Yandex'], n),
            'ResourceType': 'Универсальные маркетплейсы',
            'UseType': 'Mobile App', 'Platform': 'Mobile',
            'Пол': np.random.choice(['мужчины', 'женщины'], n),
            'Возраст': np.random.choice(['25-34', '35-44', '45-54'], n),
            'Регион': 'москва', 'Федеральный_округ': 'центральный',
            'Количество_детей': 'нет', 'Занятость': 'работает', 'Доход': 'средний',
            'Weight': np.random.uniform(10.0, 100.0, n),
            'week_weight': 100.0, 'month_weight': 100.0,
            'Start': np.random.choice(pd.date_range(end=today, periods=30, freq='s'), n),
            'researchdate': np.random.choice(dates.date, n),
            'week': np.random.choice(dates.date, n),
            'BrandinDelivery': np.random.choice([1.0, np.nan], n, p=[0.95, 0.05])
        })
    
    # 2. Предобработка и расчет
    df = preprocess_data(df)
    df_agg, anomalies = detect_anomalies(df)
    
    # --- СХЕМА ПРОВЕРКИ АНАЛИТИЧЕСКИХ ВОЗМОЖНОСТЕЙ ИЗ КОНСОЛИ ---
    
    # А. Таблица QueryText для выбранного респондента и дня
    if args.query_text:
        sub_id = int(args.query_text[0])
        try:
            date_val = pd.to_datetime(args.query_text[1]).date()
        except Exception:
            print("Ошибка: неверный формат даты. Используйте ГГГГ-ММ-ДД.")
            sys.exit(1)
        table = get_query_text_table(df, sub_id, date_val)
        if table.empty:
            print(f"Запросы для SubjectID {sub_id} на дату {date_val} не найдены.")
        else:
            print(f"\nТаблица поисковых запросов для SubjectID {sub_id} на дату {date_val}:")
            print(table.to_string(index=False))
        sys.exit(0)
        
    # Б. График для конкретного бренда
    if args.brand_plot:
        brand_id = args.brand_plot
        out_file = f"output/plots/custom_brand_{brand_id}.png"
        plot_brand_ots_chart(df, anomalies, brand_id, out_file)
        print(f"График для бренда {brand_id} построен и сохранен в: {out_file}")
        sys.exit(0)
        
    # В. График по характеристике респондента/ресурса
    if args.feature_plot:
        feature = args.feature_plot
        if feature not in df.columns:
            print(f"Ошибка: Характеристика '{feature}' отсутствует в датасете. Доступные: {list(df.columns)}")
            sys.exit(1)
        out_file = f"output/plots/custom_feature_{feature}.png"
        plot_before_after_by_feature(df, anomalies, feature, f"OTS до/после по характеристике '{feature}'", out_file)
        print(f"График по характеристике '{feature}' построен и сохранен в: {out_file}")
        sys.exit(0)

    # --- СТАНДАРТНЫЙ ПАЙПЛАЙН (если запущен без аналитических флагов) ---
    os.makedirs('output', exist_ok=True)
    
    # Сохраняем аномалии и причины
    anomalies_summary = anomalies[['SubjectID', 'researchdate']].drop_duplicates()
    anomalies_summary.to_csv('output/anomalies.csv', index=False)
    
    reasons_cols = ['SubjectID', 'researchdate', 'BrandID', 'Brand', 'CategoryDelivery', 'daily_ots', 'score', 'threshold', 'reason']
    if not anomalies.empty:
        anomalies[reasons_cols].to_csv('output/anomaly_reasons.csv', index=False)
    else:
        pd.DataFrame(columns=reasons_cols).to_csv('output/anomaly_reasons.csv', index=False)
        
    print("Результаты сохранены в папку 'output/'.")
    
    # Построение обязательных графиков
    build_plots(df, df_agg, anomalies)
    print("Диагностические графики сохранены в папку 'output/plots/'.")
    
    # Генерация стандартных аналитических графиков для демонстрации возможностей
    print("Генерация стандартных аналитических графиков для демонстрации возможностей...")
    plot_before_after_by_feature(df, anomalies, 'Пол', 'OTS до/после по полу респондента', 'output/plots/demo_gender.png')
    plot_before_after_by_feature(df, anomalies, 'Возраст', 'OTS до/после по возрастным группам', 'output/plots/demo_age.png')
    plot_before_after_by_feature(df, anomalies, 'Platform', 'OTS до/после по платформам', 'output/plots/resource_platform.png')
    
    if 'BrandID' in df.columns and len(df['BrandID'].unique()) > 0:
        top_brand = df['BrandID'].value_counts().index[0]
        plot_brand_ots_chart(df, anomalies, top_brand, 'output/plots/brand_ots_change.png')
        
    # Схема проверки: демонстрация QueryText для первого пойманного аномального респондента прямо в консоль
    if not anomalies.empty:
        sample_row = anomalies.iloc[0]
        sample_sub = sample_row['SubjectID']
        sample_date = sample_row['researchdate']
        print(f"\n[СХЕМА ПРОВЕРКИ] Демонстрация извлечения QueryText для аномального респондента {sample_sub} за {sample_date}:")
        sample_table = get_query_text_table(df, sample_sub, sample_date)
        print(sample_table.to_string(index=False))
    
    # Вывод метрик в консоль
    print_quality_metrics(df, df_agg, anomalies)

if __name__ == '__main__':
    main()