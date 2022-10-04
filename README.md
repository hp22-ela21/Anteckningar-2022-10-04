# Anteckningar - 2022-10-04
Implementering av neuralt nätverk i C++ (del III) - Slutförande av strukt för dense-lager.

Filen dense_layer.hpp innehåller strukten dense-layer, som används för att implementera dense-lager. Denna strukt är inte komplett än.

Filen main.cpp demonstrerar träning av ett dense-lager innehållande tre noder samt fyra vikter per nod. 
Träningen sker under 50 epoker och efter varje epok sker utskrift av lagrets parametrar. 
Mellan varje epok genereras ca en sekunds fördröjning för att tydliggöra framstegen under träningen.