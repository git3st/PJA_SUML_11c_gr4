# PJA_SUML_11c_gr4

Repozytorium projektu SUML - lab 11c, grupa 4

## Tematyka projektu

Aplikacji do Bukmarcherki Szachowej "Checkmate Prophet"

### Problematyka

Celem projektu jest stworzenie aplikacji, która przy wykorzystaniu wdrożonego modelu uczenia maszynowego, umożliwi użytkownikowi wprowadzenie danych dotyczących rozgrywanej partii szachowej, a następnie na podstawie tych danych oceni, który zawodnik zwycięży.
Przedwsięzięcie realizowane w ramach przedmiotu SUML na Polsko-Japońskiej Akademii Technik Komputerowych w Warszawie.

### Dataset

- **Źródło:** <https://www.kaggle.com/datasets/shkarupylomaxim/chess-games-dataset-lichess-2017-may>  
- **Liczba rekordów:** 130922  
- **Liczba zmiennych:** 33  
- **Plik z danymi:** games_metadata_profile.csv  
- **Rozmiar pliku:** 189.64 MB  
- **Predykowana zmienna:** "Result"  

## Autorzy projektu

- Adam Kaczkowski - s23020  
- Wiktor Snochowski - s22748  
- Patryk Siedlik - s22811  

## Polityka branchowania i wprowadzania zmian w repozytorium

- Autor wdrażający funkcjonalność tworzy brancha o nazwie zgodnej ze standardem: "feature/[inicjał]/[funkcjonalność]". Na przykład: "feature/ak/ml".  
- Wykonanie commitów wdrażających zmiany do brancha.  
- Utworzenie żądania zmergowania brancha z **"main"**.  
- Zmergowanie feature brancha z main.

## Przygotowanie środowiska

Przygotowano 2 dostępne procedury dotyczące przygotowania środowiska python, niezbędnego do uruchomienia modelu oraz aplikacji:

- Przygotowanie środowiska w Condzie  
- Przygotowanie środowiska, za pomocą pip  

### Przygotowanie środowiska - pip

#### 1. Pobierz plik `pip-requirements.txt`, umieszczony w katalogu config w repozytorium projektu

#### 2. Wykorzystując komendę `pip`, zainstaluj niezbędne komponenty zawarte w pobranym pliku

`pip install -r requirements.txt`

### Przygotowanie środowiska - conda

**Uwaga!**
Procedura wspierana dla następujących wersji oprogramowania:  

- Anaconda 2024.02-1  
- Conda 24.3.0  

### Instalacja Anacondy

#### 1. Pobierz dystrybucję Anaconda z oficjalnej strony: <https://www.anaconda.com/download/>

#### 2. Zweryfikuj hash pliku instalacyjnego

- Hash pliku instalacyjnego sprawdzić na stronie: <https://docs.anaconda.com/free/anaconda/hashes/>  
- Wykonać następujące polecenie powershell, wskazując plik instalacyjny Anacondy: `Get-FileHash "Anaconda3-2024.02-1-Windows-x86_64.exe" -Algorithm SHA256`. Należy podmienić nazwę pliku, jeśli jest inna.  
- Porównaj uzyskany hash z hashem uwzględnionym na oficjalnej stronie Anacondy, podanej w punkcie **1.** procedury. Jeśli się zgadza, możesz przystąpić do realizacji kolejnego punktu procedury.  

#### 3. Zainstaluj Anacondę, korzystając z pobranego pliku instalacyjnego

#### 4. Zaktualizuj condę za pomocą polecenia

`conda update conda`

### Import środowiska Conda

**1.** Z katalogu **"config"** z repozytorium projektu pobierz pliki: *"import-conda-environment.ps1"* oraz *"environment.yaml"*.  
**2.** Umieść pobrane pliki w tej samej lokalizacji.  
**3.** Uruchom konsolę **Anaconda Powershell** i przejdź lokalizacji, w której znajdują się pobrane pliki.  
**4.** Uruchomić skrypt *`import-conda-environment.ps1`*.  
**5.** Po poprawnym imporcie środowiska, w konsoli powinien pojawić się komunikat: *"Environment PJA_SUML_11c_gr4 successfully created and activated."*.

### Uruchomienie aplikacji

Z poziomu lokalizacji folderu projektu, uruchomić aplikację poleceniem:
`streamlit run "src\deployment\app.py"`

### Uruchamianie klasy main generującej model

Aby uruchomić klasę main, która generuje model, należy wykonać następującą komendę z terminala, będąc w lokalizacji folderu projektu:

```bash
python src/main.py --file_prefix=<ścieżka_do_pliku> --num_files=<liczba_plików> --output_file=<ścieżka_do_pliku_wyjściowego> --use_automl --train=<wartość> --test=<wartość> --validation=<wartość> --seed=<wartość> --n_samples=<liczba_próbek> --time_limit=<limit_czasu> --n_estimators=<liczba_estymatorów> --n_estimators_pipeline=<liczba_estymatorów_pipeline> --random_state_pipeline=<wartość> --n_samples_evaluate=<liczba_próbek_do_ewaluacji> --random_state_evaluate=<wartość> --wandb_project=<nazwa_projektu> --wandb_api_key=<klucz_api_wandb>
```
Przykładowa komenda:

```bash
python src/main.py --file_prefix=data/01_raw_data/games_metadata_profile_2024_01 --num_files=16 --output_file=data/01_raw_data/full_dataset.csv --use_automl --train=0.8 --test=0.1 --validation=0.1 --seed=50 --n_samples=500 --time_limit=60 --n_estimators=100 --n_estimators_pipeline=100 --random_state_pipeline=42 --n_samples_evaluate=100 --random_state_evaluate=0 --wandb_project=Checkmate_Prophet --wandb_api_key=<twój_klucz_api_wandb>
```
