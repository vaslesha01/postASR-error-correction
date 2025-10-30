import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from tqdm.auto import tqdm
import jiwer
import csv
from torch.utils.tensorboard import SummaryWriter


def clean_text(text):
    """
    Выполняет базовую очистку текстовой строки.
    """
    if not isinstance(text, str):
        return ""
    text = text.strip()
    if len(text) > 1 and text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    text = text.replace('—', '-').replace('–', '-')
    text = text.replace(' + ', '+')
    text = text.replace(' - ', '-')
    return text


def prepare_training_dataset(input_csv: str, output_csv: str, sample_size: int = None):
    """
    Подготовка данных для тренировки.
    """
    print(f"--- Подготовка датасета для обучения из файла: {input_csv} ---")
    df = pd.read_csv(input_csv)

    print("Убираем лишние кавычки и пробелы...")
    df['whisper_text'] = df['whisper_text'].apply(clean_text)
    df['ground_truth_text'] = df['ground_truth_text'].apply(clean_text)

    # Стандартная очистка
    df.dropna(subset=["whisper_text", "ground_truth_text"], inplace=True)
    df = df[~df['whisper_text'].str.contains("ERROR:", na=False)]

    initial_rows = len(df)

    # Фильтруем (теперь на чистых данных, без учета регистра)
    df_filtered = df.copy()

    print(f"Найдено {initial_rows} строк.")

    # Если указан sample_size, берем сэмпл
    if sample_size is not None and sample_size > 0:
        print(f"Выбираем первые {sample_size} записей для создания сэмпла.")
        final_df = df_filtered.head(sample_size)
        if len(final_df) < sample_size:
            print(f"ВНИМАНИЕ: Найдено только {len(final_df)} записей с ошибками. Используем их все.")
    else:
        final_df = df_filtered

    if len(final_df) == 0:
        print("Не найдено данных для обучения. Процесс остановлен.")
        return None

    final_df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"Итоговый датасет для обучения сохранен в '{output_csv}'. В нем {len(final_df)} записей.")
    return output_csv


def create_asr_dataset_with_whisper(
        audio_folder: str,
        ground_truth_file: str,
        output_csv_file: str
):
    """
    Создает датасет для задачи коррекции ASR.
    """
    # --- 1. Настройка ---
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-large-v3"

    print(f"--- Начало создания датасета ---")
    print(f"Используемое устройство: {device}")

    if not os.path.exists(audio_folder):
        print(f"ОШИБКА: Папка с аудиофайлами не найдена: '{audio_folder}'")
        return
    if not os.path.exists(ground_truth_file):
        print(f"ОШИБКА: Файл с эталонными текстами не найден: '{ground_truth_file}'")
        return

    # --- 2. Инициализация модели Whisper ---
    print(f"Загрузка модели '{model_id}'...")

    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        dtype=torch_dtype,
        device=device,
        framework="pt",
    )

    print("Модель успешно загружена.")

    # --- 3. Подготовка данных ---
    print("Подготовка списка файлов для обработки...")
    # Загружаем "правильные" предложения, убирая пустые строки
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        ground_truths = [line.strip() for line in f if line.strip()]

    # Получаем список аудиофайлов и СОРТИРУЕМ их,
    # чтобы порядок совпадал со строками в файле ground_truth.
    audio_files = sorted([f for f in os.listdir(audio_folder) if f.endswith('.mp3')])

    num_to_process = min(len(audio_files), len(ground_truths))

    if num_to_process == 0:
        print("Не найдено совпадающих пар аудио/текст для обработки.")
        return

    print(f"Найдено {num_to_process} пар аудио/текст. Начинается обработка...")

    # --- 4. Основной цикл обработки и сохранения в CSV ---
    with open(output_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # Записываем заголовок
        writer.writerow(["audio_filename", "whisper_text", "ground_truth_text"])

        for i in tqdm(range(num_to_process), desc="Расшифровка аудио"):
            audio_filename = audio_files[i]
            audio_path = os.path.join(audio_folder, audio_filename)
            correct_text = ground_truths[i]

            try:
                # Расшифровываем аудиофайл
                result = asr_pipeline(audio_path, generate_kwargs={"language": "russian"})
                whisper_text = result["text"].strip()

                # Записываем успешный результат
                writer.writerow([audio_filename, whisper_text, correct_text])

            except Exception as e:
                # В случае ошибки при обработке файла, записываем ошибку в датасет
                print(f"\nОшибка при обработке файла '{audio_filename}': {e}")
                writer.writerow([audio_filename, f"ERROR: {e}", correct_text])

    print(f"\n--- Создание датасета завершено. ---")
    print(f"Результат сохранен в файл '{output_csv_file}'.")


class ASRCorrectionDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=256):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prefix = 'Исправь ошибки распознавания речи в этом тексте: '

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        whisper_text = str(row['whisper_text'])
        ground_truth_text = str(row['ground_truth_text'])

        model_inputs = self.tokenizer(
            text=self.prefix + whisper_text,  # input
            text_target=ground_truth_text,  # target
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {key: val.squeeze() for key, val in model_inputs.items()}


def train_t5(
        train_csv_path: str,
        output_dir: str,
        model_checkpoint: str = r'C:\Users\user\PycharmProjects\ASRproject\rut5-base\best_model'
):
    """
    Выполняет обучение t5 с Early Stopping.
    """
    # --- 1. Настройки ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs = 30
    batch_size = 4
    learning_rate = 5e-6

    # Настройки Early Stopping
    patience = 3  # Сколько эпох ждать улучшения, прежде чем остановиться
    patience_counter = 0
    best_val_loss = float('inf')

    writer = SummaryWriter(r'C:\Users\user\PycharmProjects\ASRproject\runs\t5_base_training')

    print(f"--- Запуск обучения с Early Stopping на устройстве: {device} ---")

    # --- 2. Загрузка модели и токенизатора ---
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint).to(device)

    # --- 3. Подготовка и разделение данных ---
    full_df = pd.read_csv(train_csv_path)
    # Разделяем данные на тренировочные и валидационные
    train_df, val_df = train_test_split(full_df, test_size=0.1, random_state=42)

    train_dataset = ASRCorrectionDataset(train_df, tokenizer)
    val_dataset = ASRCorrectionDataset(val_df, tokenizer)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # --- 4. ТРЕНИРОВОЧНЫЙ ЦИКЛ С ВАЛИДАЦИЕЙ ---
    for epoch in range(num_epochs):
        print(f"\n--- Эпоха {epoch + 1}/{num_epochs} ---")

        # --- Тренировка ---
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_dataloader, desc="Тренировка"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Средний Training Loss: {avg_train_loss:.4f}")

        # --- Валидация ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():  # Отключаем градиенты для валидации
            for batch in tqdm(val_dataloader, desc="Валидация"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Средний Validation Loss: {avg_val_loss:.4f}")

        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)

        # --- Early Stopping ---
        if avg_val_loss < best_val_loss:
            print("Validation Loss улучшился! Сохраняем лучшую модель.")
            best_val_loss = avg_val_loss
            patience_counter = 0  # Сбрасываем счетчик терпения

            # Сохраняем лучшую модель
            best_model_path = os.path.join(output_dir, "best_model")
            if not os.path.exists(best_model_path):
                os.makedirs(best_model_path)
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
        else:
            patience_counter += 1
            print(f"Validation Loss не улучшился. Счетчик терпения: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Терпение исчерпано. Ранняя остановка.")
            break

    print(f"\nОбучение завершено. Лучшая модель сохранена в '{best_model_path}' с Validation Loss: {best_val_loss:.4f}")
    writer.close()


def run_inference_on_test_data(model_path: str, test_csv_path: str, output_csv_path: str):
    """
    Загружает дообученную модель и применяет ее для коррекции текстов
    в указанном тестовом CSV-файле. Сохраняет результат в новый файл.
    """
    print(f"\n--- Запуск теста модели из '{model_path}' на файле '{test_csv_path}' ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Загрузка обученной модели и токенизатора
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        model.to(device)
        model.eval() # Переводим модель в режим оценки
        print(f"Модель и токенизатор успешно загружены на {device}.")
    except Exception as e:
        print(f"ОШИБКА при загрузке модели: {e}")
        return None

    # 2. Загрузка данных для теста
    df_test = pd.read_csv(test_csv_path)
    results = []
    prefix = "Исправь ошибки распознавания речи в этом тексте: "

    # 3. Основной цикл коррекции
    with torch.no_grad():
        for index, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Генерация исправлений"):
            whisper_text = str(row['whisper_text'])
            ground_truth = str(row['ground_truth_text'])

            input_text = prefix + whisper_text
            inputs = tokenizer(input_text, return_tensors="pt").to(device)

            # Генерация ответа
            outputs = model.generate(
                **inputs,
                num_beams=5,
                max_length=256,
                early_stopping=True
            )
            corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            results.append({
                "whisper_text": whisper_text,
                "corrected_text": corrected_text,
                "ground_truth_text": ground_truth
            })

    # 4. Сохранение результата
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv_path, index=False, encoding='utf-8')
    print(f"Результаты проверки сохранены в: {output_csv_path}")
    return output_csv_path


def calculate_asr_errors_comparison(csv_filename: str):
    """
    Читает CSV-файл с результатами ASR до и после коррекции и сравнивает их.
    """
    try:
        df = pd.read_csv(csv_filename)
        required_columns = ["whisper_text", "corrected_text", "ground_truth_text"]
        if not all(col in df.columns for col in required_columns):
            print(f"ОШИБКА: В файле отсутствуют необходимые колонки: {required_columns}")
            return

        df.dropna(subset=required_columns, inplace=True)
        ground_truth = [str(text) for text in df["ground_truth_text"].tolist()]
        whisper_output = [str(text) for text in df["whisper_text"].tolist()]
        corrected_output = [str(text) for text in df["corrected_text"].tolist()]

        if not ground_truth:
            print("Не найдено данных для анализа.")
            return

        # Расчет метрик для Whisper
        whisper_wer = jiwer.wer(ground_truth, whisper_output)
        whisper_cer = jiwer.cer(ground_truth, whisper_output)

        # Расчет метрик для исправленного текста
        corrected_wer = jiwer.wer(ground_truth, corrected_output)
        corrected_cer = jiwer.cer(ground_truth, corrected_output)

        # Вывод сравнительного отчета
        print(f"{'Метрика':<12} | {'Исходный Whisper':<20} | {'После Коррекции':<20}")
        print("-" * 60)
        print(f"{'WER (ошибка)':<12} | {whisper_wer:<20.2%} | {corrected_wer:<20.2%}")
        print(f"{'CER (ошибка)':<12} | {whisper_cer:<20.2%} | {corrected_cer:<20.2%}")

    except Exception as e:
        print(f"Произошла непредвиденная ошибка при подсчете метрик: {e}")


def extract_medicines(text: str, known_medicines: set) -> set:
    """
    Находит все упоминания препаратов из списка known_medicines в тексте.
    Возвращает множество (set) найденных препаратов.
    """
    found_medicines = set()
    # Приводим текст к нижнему регистру для поиска без учета регистра
    text_lower = text.lower()
    for med in known_medicines:
        # Ищем точное вхождение названия препарата
        if med.lower() in text_lower:
            found_medicines.add(med)
    return found_medicines


def calculate_detailed_metrics_report(results_csv_path: str, medicines_list_path: str):
    """
    Рассчитывает подробный отчет с метриками Precision, Recall, F1
    для классов "Препараты" и "Не-препараты", а также их среднее.
    Сравнивает результаты до и после применения корректирующей модели.
    """
    print('\n')
    try:
        with open(medicines_list_path, 'r', encoding='utf-8') as f:
            # Приводим все к нижнему регистру сразу для корректного сравнения
            known_medicines = {line.strip().lower() for line in f if line.strip()}
        if not known_medicines:
            print("Файл с препаратами пуст. Анализ невозможен.")
            return
        print(f"Загружено {len(known_medicines)} уникальных названий препаратов (в нижнем регистре).")

        df = pd.read_csv(results_csv_path)
        df.dropna(subset=['ground_truth_text', 'whisper_text', 'corrected_text'], inplace=True)
        print(f"Загружено {len(df)} строк для анализа из файла результатов.")

    except FileNotFoundError as e:
        print(f"ОШИБКА: Не найден необходимый файл: {e}")
        return

    # Вспомогательная функция для расчета P, R, F1
    def _calculate_prf(tp, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1

    # Вспомогательная функция для выполнения расчетов (до/после)
    def _get_stats_for_scenario(predictions_col):
        stats = {
            'med': {'tp': 0, 'fp': 0, 'fn': 0},
            'non_med': {'tp': 0, 'fp': 0, 'fn': 0},
            'total_gt_words': 0,
            'total_gt_meds': 0,
        }

        for _, row in df.iterrows():
            gt_text = str(row['ground_truth_text'])
            pred_text = str(row[predictions_col])

            gt_medicines = extract_medicines(gt_text, known_medicines)
            pred_medicines = extract_medicines(pred_text, known_medicines)

            # Считаем для класса "Препараты"
            stats['med']['tp'] += len(gt_medicines.intersection(pred_medicines))
            stats['med']['fp'] += len(pred_medicines.difference(gt_medicines))
            stats['med']['fn'] += len(gt_medicines.difference(pred_medicines))

            # Считаем общие количества для класса "Не-препараты"
            stats['total_gt_words'] += len(gt_text.split())
            stats['total_gt_meds'] += len(gt_medicines)

        # Выводим метрики для класса "Не-препараты" из метрик "Препаратов"
        total_gt_non_meds = stats['total_gt_words'] - stats['total_gt_meds']
        # TP(не-препараты) = Все не-препараты минус те, что ошибочно назвали препаратами
        stats['non_med']['tp'] = total_gt_non_meds - stats['med']['fp']
        # FP(не-препараты) = Препараты, которые мы ошибочно назвали не-препаратами
        stats['non_med']['fp'] = stats['med']['fn']
        # FN(не-препараты) = Не-препараты, которые мы ошибочно назвали препаратами
        stats['non_med']['fn'] = stats['med']['fp']

        return stats

    # 2. Расчет для обоих сценариев
    stats_before = _get_stats_for_scenario('whisper_text')
    stats_after = _get_stats_for_scenario('corrected_text')

    # 3. Формирование и вывод отчета
    results_data = []

    for class_name, short_name in [("Препараты", "med"), ("Не-препараты", "non_med")]:
        p_before, r_before, f1_before = _calculate_prf(**stats_before[short_name])
        p_after, r_after, f1_after = _calculate_prf(**stats_after[short_name])

        results_data.append(['Precision', class_name, f"{p_before:.2%}", f"{p_after:.2%}"])
        results_data.append(['Recall', class_name, f"{r_before:.2%}", f"{r_after:.2%}"])
        results_data.append(['F1-Score', class_name, f"{f1_before:.2%}", f"{f1_after:.2%}"])

    # Расчет макро-усреднения
    f1_med_before, f1_non_med_before = _calculate_prf(**stats_before['med'])[2], \
    _calculate_prf(**stats_before['non_med'])[2]
    f1_med_after, f1_non_med_after = _calculate_prf(**stats_after['med'])[2], _calculate_prf(**stats_after['non_med'])[
        2]

    avarage_f1_before = (f1_med_before + f1_non_med_before) / 2
    avarage_f1_after = (f1_med_after + f1_non_med_after) / 2

    results_data.append(['---', '---', '---', '---'])
    results_data.append(['F1-Score', 'Суммарно', f"{avarage_f1_before:.2%}", f"{avarage_f1_after:.2%}"])

    accuracy_before = 0
    if stats_before['total_gt_words'] > 0:
        accuracy_before = (stats_before['med']['tp'] + stats_before['non_med']['tp']) / stats_before['total_gt_words']

    accuracy_after = 0
    if stats_after['total_gt_words'] > 0:
        accuracy_after = (stats_after['med']['tp'] + stats_after['non_med']['tp']) / stats_after['total_gt_words']

    results_data.append(['Accuracy', 'Суммарно', f"{accuracy_before:.2%}", f"{accuracy_after:.2%}"])

    report_df = pd.DataFrame(results_data,
                             columns=['Метрика', 'Класс', 'До модели', 'После модели'])
    print('\n')
    print(report_df.to_string(index=False))


class Config:
    # --- 1. ГЛАВНЫЙ ПУТЬ К ПРОЕКТУ ---
    # !!! Укажите здесь АБСОЛЮТНЫЙ путь к корневой папке вашего проекта !!!
    PROJECT_ROOT = r'C:\Users\user\PycharmProjects\ASRproject'

    # --- 2. ЭТАПЫ ЗАПУСКА ---
    # Установите True для тех шагов, которые хотите выполнить
    RUN_CREATE_DATASET = False
    RUN_PREPARE_DATASET = False
    RUN_TRAIN = False
    RUN_INFERENCE = False
    RUN_EVALUATE = True

    # --- 3. ВХОДНЫЕ ДАННЫЕ (относительно PROJECT_ROOT) ---
    # Данные для создания датасета с Whisper
    AUDIO_FOLDER = os.path.join(PROJECT_ROOT, 'data', 'output_mp3_2')
    GROUND_TRUTH_FILE = os.path.join(PROJECT_ROOT, 'data', 'final_phrases_2.txt')

    # Файл с названиями препаратов для детальной оценки
    MEDICINES_FILE = os.path.join(PROJECT_ROOT, 'data', 'medicine_names.txt')

    # --- 4. ВЫХОДНЫЕ ФАЙЛЫ И ПАПКИ (будут созданы) ---
    # Все артефакты будут храниться в своих подпапках внутри PROJECT_ROOT
    RAW_DATASET_OUTPUT = os.path.join(PROJECT_ROOT, 'asr_dataset.csv')
    TRAINING_DATASET = os.path.join(PROJECT_ROOT, 'asr_dataset_training.csv')
    MODEL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'rut5-base')
    INFERENCE_RESULTS = os.path.join(PROJECT_ROOT, 'asr_results.csv')

    # --- 5. ПАРАМЕТРЫ ПРОЦЕССА ---
    TRAINING_SAMPLE_SIZE = None  # None - использовать весь датасет, или число для среза


if __name__ == "__main__":

    print("--- Запуск пайплайна ASR-коррекции ---")

    # --- ЭТАП 1: Создание датасета с помощью Whisper ---
    if Config.RUN_CREATE_DATASET:
        print("\n[ЭТАП 1] Создание датасета...")
        create_asr_dataset_with_whisper(
            audio_folder=Config.AUDIO_FOLDER,
            ground_truth_file=Config.GROUND_TRUTH_FILE,
            output_csv_file=Config.RAW_DATASET_OUTPUT
        )

    # --- ЭТАП 2: Подготовка данных для обучения ---
    # Переменная training_data_path будет хранить путь к готовому для обучения файлу
    training_data_path = None
    if Config.RUN_PREPARE_DATASET:
        print("\n[ЭТАП 2] Подготовка данных для обучения...")
        # Используем датасет, созданный на шаге 1, или указанный вручную, если шаг 1 пропущен
        input_csv_for_prepare = Config.RAW_DATASET_OUTPUT if Config.RUN_CREATE_DATASET else Config.RAW_DATASET_INPUT

        training_data_path = prepare_training_dataset(
            input_csv=input_csv_for_prepare,
            output_csv=Config.TRAINING_DATASET,
            sample_size=Config.TRAINING_SAMPLE_SIZE
        )
        if not training_data_path:
            print("Подготовка данных не удалась. Процесс остановлен.")
            exit()
    else:
        # Если подготовка пропускается, предполагаем, что файл уже существует
        training_data_path = Config.TRAINING_DATASET
        print(f"\n[ЭТАП 2] Пропущен. Используется существующий файл: {training_data_path}")

    # --- ЭТАП 3: Обучение модели ---
    if Config.RUN_TRAIN:
        print("\n[ЭТАП 3] Обучение модели...")
        if not os.path.exists(training_data_path):
            print(f"ОШИБКА: Файл для обучения '{training_data_path}' не найден. Запустите этап подготовки.")
            exit()

        train_t5(
            train_csv_path=training_data_path,
            output_dir=Config.MODEL_OUTPUT_DIR
        )

    # --- ЭТАП 4: Инференс ---
    best_model_path = os.path.join(Config.MODEL_OUTPUT_DIR, 'best_model')
    if Config.RUN_INFERENCE:
        print("\n[ЭТАП 4] Применение модели для инференса...")
        if not os.path.exists(best_model_path):
            print(f"ОШИБКА: Обученная модель не найдена по пути: {best_model_path}. Запустите этап обучения.")
        elif not os.path.exists(training_data_path):
            print(f"ОШИБКА: Тестовый датасет не найден по пути: {training_data_path}.")
        else:
            run_inference_on_test_data(
                model_path=best_model_path,
                test_csv_path=training_data_path,
                output_csv_path=Config.INFERENCE_RESULTS
            )

    # --- ЭТАП 5: Оценка результатов ---
    if Config.RUN_EVALUATE:
        print("\n[ЭТАП 5] Оценка результатов...")
        if os.path.exists(Config.INFERENCE_RESULTS):
            # Считаем общие метрики WER/CER
            calculate_asr_errors_comparison(Config.INFERENCE_RESULTS)

            # Считаем метрики по препаратам
            if os.path.exists(Config.MEDICINES_FILE):
                calculate_detailed_metrics_report(
                    results_csv_path=Config.INFERENCE_RESULTS,
                    medicines_list_path=Config.MEDICINES_FILE
                )
            else:
                print(f"Файл с препаратами '{Config.MEDICINES_FILE}' не найден, детальная оценка пропущена.")
        else:
            print(f"Файл с результатами инференса '{Config.INFERENCE_RESULTS}' не найден. Запустите этап инференса.")

    print("\n--- Скрипт завершил работу. ---")
