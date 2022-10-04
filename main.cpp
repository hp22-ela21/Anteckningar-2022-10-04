/********************************************************************************
* main.cpp: Demonstrerar implementering av dense-lager i C++.
********************************************************************************/
#include "dense_layer.hpp"

/********************************************************************************
* delay: Implementerar approximativ fördröjning mätt i millsekunder.
*        OBS! Precisionen är inte exakt!
* 
*        - delay_time_ms: Fördröjningstid mätt i millisekunder.
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
* main: Implementerar ett dense-lager innehållande tre noder samt fyra vikter 
*       per nod. Träningsdata passeras för att träna lagret till att detektera
*       ett specifikt mönster. Lagret tränas under 50 epoker med befintlig
*       träningsdata. Efter varje epok sker utskrift av lagrets parametrar.
* 
*       Mellan varje epok fördröjs programmet ca en sekund för att vi skall
*       kunna se framstegen under träning. Kommandot CLS används för att tömma
*       skärmen innan varje ny utskrift, som passeras till funktionen system
*       från C:s standardbibliotek stdlib.h.
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