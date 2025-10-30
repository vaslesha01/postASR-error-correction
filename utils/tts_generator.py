from gtts import gTTS
import os
from tqdm import tqdm

'''
ЕСТЬ ОГРАНИЧЕНИЕ НА ОБРАЩЕНИЕ К API. ПРИМЕРНО 3к ЗАПРОСОВ
'''

language = 'ru'

input_file = r'C:\Users\user\PycharmProjects\ASRproject\data\final_phrases_2.txt'

# Папка для сохранения mp3 файлов
output_folder = r'C:\Users\user\PycharmProjects\ASRproject\data\output_mp3_2'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Открываем файл со словами
with open(input_file, 'r', encoding='utf-8') as infile:
    phrases = infile.readlines()
    phrases = phrases[:3000]

    for index, phrase in tqdm(enumerate(phrases, start=1), total=len(phrases), desc="Генерация mp3"):
        phrase = phrase.strip()
        if phrase:
            try:
                # --- Имя файла - порядковый номер ---
                filename = f"{str(index).zfill(5)}.mp3"
                file_path = os.path.join(output_folder, filename)

                # --- Создание и сохранение аудио ---
                tts = gTTS(text=phrase, lang=language, slow=False)
                tts.save(file_path)

            except Exception as e:
                print(f"Не удалось создать файл для фразы '{phrase}': {e}")

print("\nГенерация завершена.")