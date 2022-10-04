/********************************************************************************
* main.cpp: Demonstrerar implementering av dense-lager i C++.
********************************************************************************/
#include "dense_layer.hpp"

/********************************************************************************
* delay: Implementerar approximativ f�rdr�jning m�tt i millsekunder.
*        OBS! Precisionen �r inte exakt!
* 
*        - delay_time_ms: F�rdr�jningstid m�tt i millisekunder.
********************************************************************************/
static void delay(const std::size_t delay_time_ms)
{
   for (volatile std::size_t i = 0; i < delay_time_ms; ++i)
   {
      for (volatile std::size_t j = 0; j < 600000; ++j);
   }

   return;
}

/********************************************************************************
* main: Implementerar ett dense-lager inneh�llande tre noder samt fyra vikter 
*       per nod. Tr�ningsdata passeras f�r att tr�na lagret till att detektera
*       ett specifikt m�nster. Lagret tr�nas under 50 epoker med befintlig
*       tr�ningsdata. Efter varje epok sker utskrift av lagrets parametrar.
* 
*       Mellan varje epok f�rdr�js programmet ca en sekund f�r att vi skall
*       kunna se framstegen under tr�ning. Kommandot CLS anv�nds f�r att t�mma
*       sk�rmen innan varje ny utskrift, som passeras till funktionen system
*       fr�n C:s standardbibliotek stdlib.h.
********************************************************************************/
int main(void)
{
   const std::vector<double> x = { 1, 2, 3, 4 };
   const std::vector<double> yref = { 2, 4, 6 };
   dense_layer l1(3, 4);

   for (auto i = 0; i < 50; ++i)
   {
      l1.feedforward(x);
      l1.backpropagate(yref);
      l1.optimize(x, 0.01);

      system("CLS");
      l1.print();
      delay(1000);
   }

   l1.print(); 

   return 0;
}