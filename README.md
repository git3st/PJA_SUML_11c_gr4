# PJA_SUML_11c_gr4

Repozytorium projektu SUML - lab 11c, grupa 4

## Tematyka projektu

Aplikacji do Bukmarcherki Szachowej

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

Przygotowano 2 różne procedury dotyczące przygotowania środowiska python, niezbędnego do uruchomienia modelu oraz aplikacji.

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

## TODO

- [x] Add dataset
- [x] Clean dataset
- [ ] Train model
- [x] Transform code into modules
