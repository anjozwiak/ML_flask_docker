
## 1. Budowa modelu

Model wylicza churn na bazie danych klientów z branży turystycznej: 
`python train.py`

Wynikiem jest plik zwracający model wraz z odpowiednim słownikiem `model.bin`.

## 2. Obraz w dockerze (Docker, Flask)

Tworzymy obraz na pliku Dockerfile

`docker build -t modelzal .`

## 3. Kontener w dockerze

Uruchomienie kontenera

`docker run -p 5000:5000 modelzal`

## 4. Odpytanie API

Model można odpytać ze zmiennych jak poniżej:
- Age - numeryczna
- FrequentFlyer {Yes, No}
- AnnualIncomeClass {Middle Income, Low Income, High Income}
- ServicesOpted - numeryczna
- AccountSyncedToSocialMedia {Yes, No}
- BookedHotelOrNot {Yes, No}

Dla przykładu: 
- Age 48
- FrequentFlyer No
- AnnualIncomeClass Middle Income
- ServicesOpted 6
- AccountSyncedToSocialMedia No
- BookedHotelOrNot Yes

W przeglądarce:
`http://localhost:5000/api/predict?&age=48&FF='No'&AI='Middle Income'&SO=6&SM='No'&HT='Yes'`


Rezultat:
```
{"features":[{"AccountSyncedToSocialMedia":"'No'","Age":48.0,"AnnualIncomeClass":"'Middle Income'","BookedHotelOrNot":"'Yes'","FrequentFlyer":"'No'","ServicesOpted":6.0}],"y_pred":0.09503643853592621}
```

`"y_pred":0.09503643853592621` oznacza, że churn wynosi poniżej 10% - niskie prawdopodobieństwo odejścia klienta
