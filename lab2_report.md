# Лабораторная работа №2 — Отчёт

**Темы:** логистическая регрессия и классификация текста (гл. 4), эмбеддинги (гл. 5), нейронные сети для текста (гл. 6).

**Файл с решением:** [`lab2_classification.ipynb`](lab2_classification.ipynb)

---

## Использованные данные и модели

- **Датасет:** IMDB (`stanfordnlp/imdb` через HuggingFace `datasets`). Бинарная тональность отзывов на английском.
- **Подвыборка:** стратифицированная, **5 000 train / 1 000 test** (по 2500/500 на класс). Для CPU это разумный размер; на полном корпусе все методы прибавят 1–2 п.п. accuracy.
- **Эмбеддинги:** `glove-wiki-gigaword-100` через `gensim.downloader` (∼128 МБ, кэш в `~/gensim-data`). 400k слов × 100 dim.
- **Библиотеки:** `scikit-learn 1.8`, `gensim 4.4`, `torch 2.12` (CPU), `nltk 3.9`, `matplotlib`, `seaborn`. Все версии — последние стабильные на момент работы.

---

## Что сделано

### Часть 1 — TF-IDF + Logistic Regression (35%)

- **Предобработка:** HTML-теги (`<br>` итд) → пробел, lowercase, удаление пунктуации, удаление англ. стоп-слов (`nltk.corpus.stopwords`). 3 примера до/после в ноутбуке.
- **TF-IDF:** `TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)` → матрица **(5000, 20000)**.
- **Топ-слова с наибольшим средним TF-IDF:**
  - **negative:** `bad`, `even`, `like`, `would`, `acting`, `boring`, `worst` — оценочно негативная лексика.
  - **positive:** `great`, `well`, `love`, `best`, `excellent`, `also` — оценочно позитивная.
- **LogisticRegression** (`liblinear`, `C=1.0`):
  - **Accuracy: 0.867**
  - **Precision: 0.861**
  - **Recall: 0.876**
  - **F1: 0.868**
- Confusion matrix симметричная (≈65–70 ошибок на класс).
- **Анализ ошибок (133 из 1000):** ошибки идут на сарказме («glossy, sepia-tinted makeover» — позитивные слова, но автор хвалит ехидно), длинных спойлерных пересказах с мало оценочной лексикой, и на отзывах с противоречивыми оценками (начало плохое — концовка вытягивает).

### Часть 2 — Статические эмбеддинги GloVe (35%)

- **GloVe-100 загружен:** 400 000 слов, dim = 100.
- **Ближайшие соседи (cosine sim):**
  - `king` → prince (0.77), queen (0.75), son (0.70), brother (0.70), monarch (0.70).
  - `movie` → film (0.91), movies (0.90), films (0.87), hollywood (0.82), comedy (0.81).
  - `happy` → 'm, feel, 're, i, 'll — слово «расходится» в сторону служебных, т.к. в Gigaword `i'm happy` встречается часто.
  - `computer` → computers, software, technology, pc, hardware.
- **Аналогии:**
  - `king + woman − man ≈ queen (0.77)` ✓
  - `paris + germany − france ≈ berlin (0.88)` ✓
  - `walked + swim − walk ≈ swam (0.75)` ✓ (морфология!)
  - `bigger + small − big ≈ larger (0.89)` ✓ (компаратив)
- **t-SNE на 5 тематических группах** (animals/countries/professions/emotions/colors × 10 слов) — каждая группа образует чёткий кластер; emotions расползаются больше всех, т.к. часто употребляются адъективно («happy times», «sad story»).
- **Усреднённые GloVe → LogReg:** **accuracy 0.794** — на 7 п.п. ниже TF-IDF. Усреднение «гасит» оценочные слова, особенно редкие и эмоционально нагруженные.

### Часть 3 — Нейронная сеть (30%)

- **Архитектура:** `Embedding(20k, 100, padding_idx=0) → mean-pool по маске → Linear(100, 64) → ReLU → Linear(64, 1)`. `BCEWithLogitsLoss`, Adam(lr=1e-3), batch_size=64, 10 эпох, max_length=256.
- **Покрытие GloVe для словаря IMDB:** 18 754 / 20 000 = 93.8%.
- **Вариант A — Random Embedding (обучается):**
  - Train accuracy 99.68% к 10-й эпохе, **val accuracy = 0.815** — классический оверфит при маленьком train.
- **Вариант B — Frozen GloVe:**
  - Train accuracy 78.62%, **val accuracy = 0.778** — модель не переобучается (эмбеддинги не двигаются), но и не догоняет А по точности.

### Итоговая таблица

| Метод | Представление текста | Test Accuracy |
|---|---|---:|
| Logistic Regression | TF-IDF (uni+bi, 20k) | **0.867** |
| Logistic Regression | Среднее GloVe-100 | 0.794 |
| NN (random embeddings, обучаются) | Embedding-100 | 0.815 |
| NN (GloVe, заморожены) | Pretrained GloVe-100 | 0.778 |

Сохранено в `results_lab2.json` для использования в Лабе 3.

---

## Ключевые наблюдения

1. **TF-IDF + LR — самый эффективный** на этой задаче и подвыборке. Причина: IDF напрямую усиливает дискриминирующие слова, а биграммы захватывают локальные отрицания (`not good`, `very bad`).
2. **Усреднённый GloVe слабее**, потому что усреднение размывает специфичные оценочные слова. На задачах semantic similarity GloVe выигрывает, на sentiment — нет.
3. **NN с обучаемыми эмбеддингами** переобучается на маленьком train (5k), но всё равно даёт чуть лучше чем avg GloVe + LR. С большим train и регуляризацией (dropout, early stopping) обогнала бы TF-IDF.
4. **Frozen GloVe = mean-GloVe по сути.** Заморозка эмбеддингов лишает сеть возможности адаптироваться к задаче — остаётся только маленький классификатор сверху.
5. **Аналогии GloVe** очень точно ловят семантические и морфологические отношения (`walked → swam`, `bigger → larger`). Это качественная проверка, что предобученные представления «знают» структуру языка.

---

## Как воспроизвести

```bash
cd "Основы обработки текста"
source .venv/bin/activate    # окружение из requirements.txt
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=2400 lab2_classification.ipynb
```

Первый запуск качает GloVe (~128 МБ) и IMDB (~80 МБ). На M-чипе общий прогон занимает 8–12 минут.
