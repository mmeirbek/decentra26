# Soup Kernel Challenge: слитый Triton-путь для 4-битной LoRA на Kaggle T4

`solution.apply()` подключает к Soup 0.77.0 семь своих Triton-кернелов и переписанный путь
обучения Qwen2. Задача та же: тот же адаптер, те же данные и порядок батчей, те же метрики и
`rewind.jsonl`. Команда `soup train` на Kaggle T4 идёт примерно в **1,6 раза быстрее** и занимает
примерно в **5 раз меньше памяти GPU**, при том же качестве адаптера. Все 10 проверок
`harness/check_rules.py` проходят.

| Kaggle T4, `run_task.py --repeats 2` | База Soup | Наше решение | Отношение |
|---|---|---|---|
| Время `soup train`, с | RESULT_BASE_TIME | RESULT_SOL_TIME | **s = RESULT_S** |
| Пик памяти GPU (nvidia-smi), МиБ | RESULT_BASE_MEM | RESULT_SOL_MEM | **m = RESULT_M** |
| Eval loss на открытых строках | RESULT_BASE_LOSS | RESULT_SOL_LOSS | RESULT_DLOSS % |

Ноутбук Kaggle: KAGGLE_LINK · финальный коммит: COMMIT_LINK

## Что сделано

### С чего начали: куда уходят память и время в базе

Сначала мы прочитали путь `soup train` в исходниках точных версий (soup-cli 0.77.0,
transformers 5.16.1, peft 0.20.0, trl 0.29.1, bitsandbytes 0.50.2, accelerate 1.14.0) и измерили
один шаг на CPU с реальными размерами модели (`dev/compare_cpu.py`). Сумма компонентов сошлась с
замером организаторов (14 350 МиБ) с точностью до нескольких процентов.

- **Активации в fp32.** `prepare_model_for_kbit_training` поднимает нормы и эмбеддинг в fp32, а
  bitsandbytes возвращает выход 4-битного слоя в dtype входа. Поэтому всё между умножениями живёт
  в fp32, хотя каждое умножение всё равно получает fp16. Один MLP хранит для backward три fp32-тензора
  по 8 960 чисел на токен: 5,9 ГБ на пиковом батче.
- **Логиты по словарю из 151 936 токенов** существуют в 3–4 полных копиях: fp32-копия в
  `ForCausalLMLoss` и её `log_softmax`, fp32-копия выхода модели от accelerate, копия для точности
  TRL (`.contiguous()`), fp16-копия веса головы от autocast. Около 4 ГБ на пике.
- **Паддинг.** Батч дополняется до самой длинной строки, и все умножения считаются и на pad.
- **Поэлементные операции** (44% шага в профиле организаторов): касты fp16↔fp32 на входе и выходе
  каждой проекции, `clone` и dropout в LoRA, RoPE через `cat`/`neg`, расширение k/v до 12 голов,
  SiLU и умножение в fp32. Наш счёт трафика дал ~2,2 МБ на токен на слой.
- **Распаковка NF4** идёт двумя кернелами bitsandbytes на каждый вес, в forward и снова в backward.

### Наши кернелы (`solution/kernels_*.py`, все `@triton.jit`)

| Кернел | Файл | Что заменяет | Как устроен |
|---|---|---|---|
| `lm_loss_kernel` | `kernels_loss.py` | `ForCausalLMLoss`, fp32-копии логитов, энтропию и точность TRL, потери строк Soup | Голова считается cuBLAS кусками по 128 токенов в fp16. Одна программа на токен проходит строку логитов дважды: online max / сумма экспонент / сумма `e·z`, argmax с первым индексом и логит метки, откуда loss, энтропия и попадание; второй проход пишет на место логитов градиент `(softmax − one_hot)·scale` в fp16, где `scale` — текущий масштаб GradScaler на число меток, как в базе. Градиент по скрытому состоянию считается сразу в forward |
| `nf4_dequant_kernel` | `kernels_nf4.py` | `dequantize_blockwise` + `dequantize_4bit` из bitsandbytes | Поток читает 32-битное слово (8 весов), восстанавливает двойной масштаб и пишет 8 fp16 подряд. FMA выключен: веса совпадают с bitsandbytes бит в бит (тест) |
| `add_rmsnorm_fwd_kernel`, `add_rmsnorm_bwd_kernel` | `kernels_norm.py` | `residual + x` и `Qwen2RMSNorm` в fp32 | Остаток в fp32, выход для умножения в fp16; для backward хранятся только `xhat` (fp16) и `1/rms`. Перед attention тот же проход пишет входы LoRA с масками dropout и упаковывает маски в биты |
| `qkv_rope_fwd_kernel`, `qkv_rope_bwd_kernel` | `kernels_attn.py` | `clone`, сложение LoRA, касты, RoPE через `cat`/`neg` | Одна программа на (токен, голову): выход общего q\|k\|v-умножения + LoRA·scaling, поворот RoPE в fp32, запись в упакованную раскладку для attention |
| `gated_act_kernel` | `kernels_mlp.py` | fp32 gate/up, SiLU, умножение | Читает общий буфер gate\|up (одно умножение на оба веса), пишет `SiLU(gate)·up` в fp16; backward пишет `dgate`/`dup` на место того же буфера |

### Что ещё изменено вокруг кернелов

- **Без паддинга.** В начале forward берутся только реальные токены (`rowmap`); все умножения и
  кернелы работают на них. Attention считается по каждой строке на её длине с причинной маской —
  одним вызовом того же memory-efficient кернела, что выбирает SDPA на T4, но по `cu_seqlens` вместо
  маски с паддингом. Строки разных батчей не смешиваются.
- **Пересчёт MLP.** Для backward не хранятся активации MLP: gate\|up пересчитываются из нормированного
  входа одним умножением (≈ −1,1 ГБ на пике, ≈ +10–13% ко времени шага).
- **Меньше сохранённого.** Входы LoRA и MLP пересобираются в backward из `xhat` и битов масок;
  k и v хранятся с 2 головами, а не расширенными до 12.
- **Эмбеддинг в fp16 на время обучения.** На T4 модель грузится в fp16, peft лишь расширяет её до
  fp32, так что хранение в fp16 не меняет ни одного значения; после обучения вес возвращается бит в бит.
- **Одно рабочее пространство** для распакованных весов вместо новых буферов на каждый вес (меньше
  фрагментации аллокатора).
- **Компиляция в фоне.** Пока грузятся веса, фоновый поток того же процесса компилирует все варианты
  кернелов и готовит их запуск (`warmup.py`); первый шаг их не ждёт.

### Почему результат тот же

- Каждое умножение получает те же fp16-входы, что и в базе; остаточный поток остаётся в fp32.
- **Та же случайность.** Маски dropout LoRA рисует тот же `torch.native_dropout` на тензоре той же
  формы, dtype и в том же порядке, что и в базе (тест проверяет совпадение маски и состояния ГСЧ).
  CPU-ГСЧ наш код не трогает, так что порядок батчей и инициализация LoRA те же.
- **Те же выходы Soup.** `entropy`, `mean_token_accuracy` и `num_tokens` пишутся по формулам TRL,
  потери строк передаются в `rewind.jsonl` через атрибут `_soup_chunked_step`, который rewind Soup
  читает для шагов без логитов (#1693). Режим `stream_loss_chunk` при этом не включается.
- Батч, где у замаскированной позиции есть метка (база считает и её), идёт путём Soup целиком.

### Как подключается

`apply()` ставит три узких патча, и каждый возвращает исходный код, если перед ним не настоящая
4-битная Qwen2 с LoRA на CUDA (так проходят 79 тестов Soup с подставными модулями):
`SFTTrainerWrapper._setup_transformers` / `.train` (план включается только на время обучения),
`Qwen2ForCausalLM.forward` / `__init__` (слитый forward и фоновая компиляция) и
`trl.SFTTrainer.compute_loss` (метрики и rewind из слитого шага).

### Источники идей (код написан нами)

- Слитая потеря по большому словарю без полных логитов: Liger Kernel (FusedLinearCrossEntropy),
  Unsloth (fast cross entropy), Wijmans et al., *Cut Your Losses in Large-Vocabulary Language Models*, 2024.
- Online-softmax: Milakov & Gimelshein, *Online normalizer calculation for softmax*, 2018;
  Dao et al., *FlashAttention*, 2022.
- RMSNorm, SwiGLU и RoPE в одном проходе: Liger Kernel, Unsloth, урок Triton *Layer Normalization*.
- Формат NF4 и двойная квантизация: Dettmers et al., *QLoRA*, 2023; формулы распаковки —
  bitsandbytes 0.50.2.
- Пересчёт активаций в backward: Chen et al., *Training Deep Nets with Sublinear Memory Cost*, 2016.
- Формулы метрик без логитов и крючок rewind: `soup_cli/utils/stream_loss_chunk.py` (Soup 0.77.0).

## Как запустить

**На Kaggle (как жюри).** Импортировать `templates/kaggle_notebook.ipynb`
(File → Import Notebook), включить GPU T4 x2 и Internet, в первой ячейке указать
`REPO_URL = "https://github.com/mmeirbek/decentra26.git"` и `COMMIT = "FINAL_COMMIT"`, нажать
Run All (~15 минут). Сохранённая версия с результатами: KAGGLE_LINK.

**Локально на Linux с CUDA:**

```bash
pip install "soup-cli[train]==0.77.0" "transformers==5.16.1" "peft==0.20.0" "trl==0.29.1" \
    "bitsandbytes==0.50.2" "accelerate==1.14.0" pytest
python task/prepare_data.py
python -m pytest solution -q            # наши кернелы против PyTorch и bitsandbytes
python harness/run_task.py --repeats 2   # база и решение -> runs/result.json
python harness/check_rules.py            # 10 проверок -> runs/checks.json
```

**Без GPU** (так мы проверяли кернелы между прогонами): `dev/Dockerfile` собирает CPU-копию
окружения, где кернелы идут под интерпретатором Triton:

```bash
docker build -t soupk-cpu dev
docker run --rm -e TRITON_INTERPRET=1 -v "$PWD":/repo soupk-cpu python -m pytest solution -q
docker run --rm -e SOUP_APPLY_SOLUTION=1 -v "$PWD":/repo soupk-cpu python -m pytest tests -q
```

Папка `dev/` в решение не входит: сравнение шага на CPU (`compare_cpu.py`), отправка прогонов на
Kaggle через API (`kaggle_push.py`), подбор параметров запуска на T4 (`tune_probe.py`).

## Результаты и ограничения

RESULTS_DETAILS

### Ограничения и что не получилось

- Путь написан под этот случай: Qwen2, NF4 с двойной квантизацией, fp16 на T4, LoRA на q_proj и
  v_proj, SDPA. Для всего остального работает путь Soup без изменений.
- Пик памяти зависит от самого длинного батча скрытых данных (по реальным токенам); на открытых
  строках он ~1 000 токенов, батч из четырёх строк по 512 даст пик выше.
- Пересчёт MLP — сознательный обмен: ≈ −1,1 ГБ на пике за ≈ +10–13% ко времени шага.
- ~20 с вне обучения (импорты, CUDA, загрузка модели) мы не трогали: они одинаковы у базы и у нас.
- Умножения матриц остались за cuBLAS, а распаковка NF4 — отдельным кернелом перед ними: свой
  слитый «распаковка + умножение» на Triton для T4 мы не делали.
- Attention считает memory-efficient кернел PyTorch (тот же, что у базы через SDPA), а не наш.
- Компиляция кернелов идёт в фоновом потоке того же процесса; её время входит в замер, но
  перекрыто загрузкой весов.
