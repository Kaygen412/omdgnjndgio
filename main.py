import os
import sys
import tempfile
import numpy as np
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel

class SongRecognizer:
    def __init__(self, model_size="base"):
        """
        Инициализация распознавателя
        
        Args:
            model_size: tiny, base, small, medium, large-v3
        """
        print(f"Загружаю модель faster-whisper ({model_size})...")
        self.model = WhisperModel(model_size, device="auto", compute_type="auto")
        print("Модель загружена!")
        
        # Настройки микрофона
        self.RATE = 16000
        self.CHANNELS = 1
        
        # Показать доступные устройства
        print("\nДоступные аудиоустройства:")
        print(sd.query_devices())
        print()
        
    def recognize_file(self, file_path):
        """Распознавание из файла"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл не найден: {file_path}")
            
        print(f"Распознаю файл: {file_path}")
        segments, info = self.model.transcribe(
            file_path, 
            language="ru", 
            beam_size=5,
            vad_filter=True  # Фильтр тишины — улучшает качество
        )
        
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text)
            
        return {
            "text": "".join(text_parts),
            "language": info.language,
            "language_probability": info.language_probability
        }
    
    def record_from_microphone(self, duration=30):
        """
        Запись с микрофона через sounddevice
        
        Returns:
            путь к временному WAV файлу
        """
        print(f"Записываю {duration} секунд... (нажмите Ctrl+C для остановки)")
        
        frames = []
        total_chunks = int(self.RATE / 1024 * duration)
        
        def audio_callback(indata, frame_count, time_info, status):
            if status:
                print(f"\nСтатус: {status}")
            frames.append(indata.copy())
            
            current = len(frames)
            progress = int(current / total_chunks * 50)
            sys.stdout.write(f"\r[{('=' * progress)}{' ' * (50 - progress)}] "
                           f"{int(current/total_chunks*100)}%")
            sys.stdout.flush()
        
        try:
            with sd.InputStream(
                samplerate=self.RATE,
                channels=self.CHANNELS,
                dtype='float32',
                blocksize=1024,
                callback=audio_callback
            ):
                # Спим нужное количество времени
                sd.sleep(int(duration * 1000))
                
        except KeyboardInterrupt:
            print("\nЗапись остановлена пользователем")
        
        print("\nЗапись завершена!")
        
        if not frames:
            raise RuntimeError("Не удалось записать аудио — проверьте микрофон")
        
        # Объединяем все куски
        audio_data = np.concatenate(frames, axis=0)
        
        # Сохраняем во временный файл
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(temp_file.name, audio_data, self.RATE)
        
        return temp_file.name
    
    def recognize_from_microphone(self, duration=30):
        """Запись с микрофона и распознавание"""
        temp_file = self.record_from_microphone(duration)
        
        try:
            print("Распознаю запись...")
            segments, info = self.model.transcribe(
                temp_file, 
                language="ru", 
                beam_size=5,
                vad_filter=True
            )
            
            text_parts = []
            for segment in segments:
                text_parts.append(segment.text)
                
            return {
                "text": "".join(text_parts),
                "language": info.language,
                "language_probability": info.language_probability
            }
        finally:
            os.unlink(temp_file)


def main():
    print("=" * 60)
    print("Распознаватель песен")
    print("=" * 60)
    
    print("\nВыберите модель:")
    print("1. tiny - очень быстрая, базовая точность")
    print("2. base - быстрая, хорошая точность (рекомендуется)")
    print("3. small - средняя скорость и точность")
    print("4. medium - медленная, высокая точность")
    print("5. large-v3 - максимальная точность")
    
    model_choice = input("\nВведите номер (1-5) [по умолчанию 2]: ").strip() or "2"
    model_map = {
        "1": "tiny", "2": "base", "3": "small", 
        "4": "medium", "5": "large-v3"
    }
    model_size = model_map.get(model_choice, "base")
    
    try:
        recognizer = SongRecognizer(model_size)
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return
    
    while True:
        print("\n" + "=" * 60)
        print("1. Распознать аудиофайл")
        print("2. Записать с микрофона и распознать")
        print("3. Выход")
        
        choice = input("\nВведите номер (1-3): ").strip()
        
        if choice == "1":
            file_path = input("Путь к аудиофайлу: ").strip().strip('"')
            if not file_path:
                print("Путь не указан!")
                continue
            
            try:
                result = recognizer.recognize_file(file_path)
                print("\n" + "=" * 60)
                print("РЕЗУЛЬТАТ:")
                print("=" * 60)
                print(result["text"])
                print("=" * 60)
                
                if input("\nСохранить в файл? (y/n): ").strip().lower() == 'y':
                    output_file = input("Имя файла [output.txt]: ").strip() or "output.txt"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(result["text"])
                    print(f"Сохранено в {output_file}")
                    
            except Exception as e:
                print(f"Ошибка: {e}")
                
        elif choice == "2":
            duration = input("Длительность записи, сек [30]: ").strip()
            duration = int(duration) if duration else 30
            
            try:
                result = recognizer.recognize_from_microphone(duration)
                print("\n" + "=" * 60)
                print("РЕЗУЛЬТАТ:")
                print("=" * 60)
                print(result["text"])
                print("=" * 60)
                
                if input("\nСохранить в файл? (y/n): ").strip().lower() == 'y':
                    output_file = input("Имя файла [output.txt]: ").strip() or "output.txt"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(result["text"])
                    print(f"Сохранено в {output_file}")
                    
            except Exception as e:
                print(f"Ошибка: {e}")
                
        elif choice == "3":
            print("До свидания!")
            break
        else:
            print("Неверный выбор!")


if __name__ == "__main__":
    main()