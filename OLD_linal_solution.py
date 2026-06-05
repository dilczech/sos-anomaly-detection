# -*- coding: utf-8 -*-
import os
import sys
import glob
import pandas as pd
import numpy as np
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
    Рассчитывает медиану и MAD на определенном уровне агрегации.
    """
    stats = df.groupby(group_cols).agg(
        n_uniq=('SubjectID', 'nunique'),
        median=('daily_ots', 'median')
    ).reset_index()
    
    df_temp = df.merge(stats, on=group_cols, how='left')
    df_temp['ad'] = (df_temp['daily_ots'] - df_temp['median']).abs()
    
    mad = df_temp.groupby(group_cols).agg(
        mad=('ad', 'median')
    ).reset_index()
    
    stats = stats.merge(mad, on=group_cols, how='left')
    
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
    
    # Рассчитываем Modified Z-score и порог для каждого уровня
    score_lvl1 = 0.6745 * (df_agg['daily_ots'] - df_agg['median_lvl1']) / df_agg['mad_lvl1']
    thresh_lvl1 = df_agg['median_lvl1'] + 3.5 * df_agg['mad_lvl1'] / 0.6745
    
    score_lvl2 = 0.6745 * (df_agg['daily_ots'] - df_agg['median_lvl2']) / df_agg['mad_lvl2']
    thresh_lvl2 = df_agg['median_lvl2'] + 3.5 * df_agg['mad_lvl2'] / 0.6745
    
    score_lvl3 = 0.6745 * (df_agg['daily_ots'] - df_agg['median_lvl3']) / df_agg['mad_lvl3']
    thresh_lvl3 = df_agg['median_lvl3'] + 3.5 * df_agg['mad_lvl3'] / 0.6745
    
    # Объединяем результаты по условиям
    df_agg['score'] = np.select([cond_lvl1, cond_lvl2, cond_lvl3], [score_lvl1, score_lvl2, score_lvl3], default=np.nan)
    df_agg['threshold'] = np.select([cond_lvl1, cond_lvl2, cond_lvl3], [thresh_lvl1, thresh_lvl2, thresh_lvl3], default=np.nan)
    df_agg['level'] = np.select([cond_lvl1, cond_lvl2, cond_lvl3], ['Brand', 'Category', 'Date'], default='None')
    
    # Флаг аномалии: score > 3.5
    df_agg['is_anomaly'] = (df_agg['score'] > 3.5) & (df_agg['level'] != 'None')
    
    # Создаем детальные причины для аномалий
    anomalies = df_agg[df_agg['is_anomaly']].copy()
    if not anomalies.empty:
        def get_reason(row):
            return (f"Выброс на уровне '{row['level']}': "
                    f"Z-score = {row['score']:.2f} > 3.5 "
                    f"(daily_ots = {row['daily_ots']:.2f}, "
                    f"median = {row['median_lvl1'] if row['level']=='Brand' else (row['median_lvl2'] if row['level']=='Category' else row['median_lvl3']):.2f}, "
                    f"порог = {row['threshold']:.2f})")
        anomalies['reason'] = anomalies.apply(get_reason, axis=1)
    else:
        anomalies['reason'] = []
        
    return df_agg, anomalies

def build_plots(df, df_agg, anomalies, output_dir='output/plots'):
    """
    Создает три обязательных диагностических графика.
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
    
    # 2. category_ots_change.png
    ots_cat_before = df_agg.groupby('CategoryDelivery')['daily_ots'].sum()
    ots_cat_after = df_agg[~df_agg['is_removed']].groupby('CategoryDelivery')['daily_ots'].sum()
    
    cat_change_pct = ((ots_cat_after - ots_cat_before) / ots_cat_before * 100).fillna(0).sort_values()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=cat_change_pct.values, y=cat_change_pct.index, palette='Reds_d')
    plt.title('Изменение OTS по CategoryDelivery после очистки (%)', fontsize=14)
    plt.xlabel('Изменение (%)', fontsize=12)
    plt.ylabel('Категория доставки', fontsize=12)
    plt.grid(True, axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'category_ots_change.png'), dpi=150)
    plt.close()
    
    # 3. daily_anomaly_count.png
    anom_count_by_day = anomalies.groupby('researchdate')['SubjectID'].nunique().sort_index()
    all_dates = pd.date_range(start=df_agg['researchdate'].min(), end=df_agg['researchdate'].max()).date
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

def main():
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        # Автоматический поиск папок в текущей директории
        month_folders = glob.glob('month=*')
        if month_folders:
            input_path = month_folders[0]
        else:
            input_path = '.'
            
    print(f"Поиск данных в: {input_path}")
    
    # 1. Загрузка данных
    try:
        df = load_data(input_path)
        print(f"Данные загружены. Строк: {len(df)}")
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        sys.exit(1)
    
    # 2. Предобработка
    df = preprocess_data(df)
    
    # 3. Поиск аномалий
    df_agg, anomalies = detect_anomalies(df)
    
    # Создаем папку назначения
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
    
    # 4. Построение обязательных графиков
    build_plots(df, df_agg, anomalies)
    print("Диагностические графики сохранены в папку 'output/plots/'.")
    
    # 5. Генерация аналитических разрезов (Пункт 8.2)
    print("Генерация аналитических разрезов...")
    plot_before_after_by_feature(df, anomalies, 'Пол', 'OTS до/после по полу респондента', 'output/plots/demo_gender.png')
    plot_before_after_by_feature(df, anomalies, 'Возраст', 'OTS до/после по возрастным группам', 'output/plots/demo_age.png')
    plot_before_after_by_feature(df, anomalies, 'Platform', 'OTS до/после по платформам', 'output/plots/resource_platform.png')
    
    if 'BrandID' in df.columns and len(df['BrandID'].unique()) > 0:
        top_brand = df['BrandID'].value_counts().index[0]
        plot_brand_ots_chart(df, anomalies, top_brand, 'output/plots/brand_ots_change.png')
        
    # Вывод метрик в консоль
    print_quality_metrics(df, df_agg, anomalies)
    
if __name__ == '__main__':
    main()