/********************************************************************************
* dense_layer.hpp: Inneh�ller funktionalitet f�r implementering av dense-lager
*                  i neurala n�tverk (dolda lager och utg�ngslager) via
*                  strukten dense_layer.
********************************************************************************/
#ifndef DENSE_LAYER_HPP_
#define DENSE_LAYER_HPP_

/* Inkluderingsdirektiv: */
#include <iostream> /* Funktionalitet f�r inmatning och utskrift. */
#include <iomanip>  /* I/O-manipulationer, s�som antalet decimaler vid utskrift. */
#include <vector>   /* Dynamiska vektorer. */
#include <cstdlib>  /* Randomiseringsfunktionen rand. */

/********************************************************************************
* dense_layer: Dense-lager f�r implementering i neurala n�tverk. Nodernas
*              parametrar (utsignal, bias, fel och vikter) lagras via vektorer.
********************************************************************************/
struct dense_layer
{
   std::vector<double> output;               /* Nodernas utsignaler. */
   std::vector<double> error;                /* Nodernas avvikelser/felv�rden. */
   std::vector<double> bias;                 /* Nodernas bias/vilov�rden. */
   std::vector<std::vector<double>> weights; /* Nodernas vikter. */

   /********************************************************************************
   * dense_layer: Skapar ett nytt tomt dense-lager.
   ********************************************************************************/
   dense_layer(void) { }

   /********************************************************************************
   * dense_layer: Skapar och initierar ett nytt dense-lager med angivet antal
   *              noder samt vikter per nod.
   *
   *              - num_nodes  : Antalet noder i det nya lagret.
   *              - num_weights: Antalet vikter per nod i det nya lagret.
   ********************************************************************************/
   dense_layer(const std::size_t num_nodes,
               const std::size_t num_weights)
   {
      this->resize(num_nodes, num_weights);
      return;
   }

   /********************************************************************************
   * ~dense_layer: Nollst�ller dense-lager automatiskt innan objektet raderas.
   ********************************************************************************/
   ~dense_layer(void)
   {
      this->clear();
      return;
   }

   /********************************************************************************
   * clear: Nollst�ller dense-lager genom att t�mma vektorer.
   ********************************************************************************/
   void clear(void)
   {
      this->output.clear();
      this->error.clear();
      this->bias.clear();
      this->weights.clear();
      return;
   }

   /********************************************************************************
   * resize: S�tter nytt antal noder och vikter per nod i angivet dense-lager.
   *         Alla bias och vikter tilldelas randomiserade startv�rden mellan 0 - 1,
   *         �vriga parametrar s�tts till 0. Funktionen fungerar enligt nedan.
   * 
   *         1. Eventuellt tidigare inneh�ll frig�rs innan omallokeringen.
   *         2. Vektorer f�r lagring av utsignaler, fel samt bias s�tts till att
   *            rymma flyttal f�r specificerat antal noder, d�r samtliga parametrar 
   *            tilldelas startv�rdet 0.
   *         3. Den tv�dimensionella vektorn weights s�tts till att rymma
   *            endimensionella vektorer inneh�llande flyttal f�r specificerat
   *            antal vikter. F�r varje nod l�ggs en s�dan vektor till, d�rav
   *            s�tts antalet endimensionella vektorer till antalet noder i lagret.
   *         4. Samtliga bias-v�rden och vikter s�tts till randomiserade 
   *            startv�rden mellan 0 - 1 via iteration i kombination med anrop 
   *            av medlemsfunktionen get_random.
   * 
   *         - num_nodes  : Nytt antal noder i dense-lagret.
   *         - num_weights: Nytt antal vikter per nod i dense-lagret.
   ********************************************************************************/
   void resize(const std::size_t num_nodes,
               const std::size_t num_weights)
   {
      this->clear();

      this->output.resize(num_nodes, 0);
      this->error.resize(num_nodes, 0);
      this->bias.resize(num_nodes, 0);
      
      this->weights.resize(num_nodes, std::vector<double>(num_weights, 0));

      for (std::size_t i = 0; i < this->num_nodes(); ++i)
      {
         this->bias[i] = this->get_random();

         for (std::size_t j = 0; j < this->num_weights(); ++j)
         {
            this->weights[i][j] = this->get_random();
         }
      }

      return;
   }

   /********************************************************************************
   * get_random: Returnerar ett randomiserat flyttal mellan 0 - 1.
   ********************************************************************************/
   double get_random(void) 
   {
      return static_cast<double>(std::rand()) / RAND_MAX;
   }

   /********************************************************************************
   * num_nodes: Returnerar antalet noder i aktuellt dense-lager.
   ********************************************************************************/
   std::size_t num_nodes(void) const
   {
      return this->output.size();
   }

   /********************************************************************************
   * num_weights: Returnerar antalet vikter per nod i aktuellt dense-lager.
   ********************************************************************************/
   std::size_t num_weights(void) const
   {
      if (this->weights.size())
      {
         return this->weights[0].size();
      }
      else
      {
         return 0;
      }
   }

   /********************************************************************************
   * feedforward: Ber�knar nya utsignaler igenom dense-lagret via nya insignaler,
   *              lagrade i en vektor. Ber�kningen genomf�rs enligt nedan:
   * 
   *              1. Itererar genom lagret fr�n nod till nod.
   *              2. Summan av nodens bias samt dess vikter adderas.
   *              3. Om summan �verstiger 0 �r noden aktiverad. D� s�tts nodens
   *                 utsignal till denna summa. Annars s�tts utsignalen till 0.
   * 
   *              - input: Nya insignaler till dense-lagret.
   ********************************************************************************/
   void feedforward(const std::vector<double>& input)
   {
      for (int i = 0; i < this->num_nodes(); ++i)
      {
         auto sum = this->bias[i];

         for (std::size_t j = 0; j < this->num_weights() && j < input.size(); ++j)
         {
            sum += input[j] * this->weights[i][j];
         }

         if (sum > 0)
         {
            this->output[i] = sum;
         }
         else
         {
            this->output[i] = 0;
         }
      }

      return;
   }

   /********************************************************************************
   * backpropagate: Ber�knar aktuella fel i utg�ngslagret genom att j�mf�ra
   *                referensv�rden fr�n tr�ningsdatan med utsignaler ur lagret.
   *                Enbart om noden �r aktiverad ber�knas fel. OBS! Denna funktion
   *                �r enbart avsedd f�r utg�ngslager.
   * 
   *                1. Iterera fr�n nod till nod i utg�ngslagret.
   *                2. Ber�kna avvikelse som differensen mellan aktuellt 
   *                   referensv�rde samt motsvarande utsignal.
   *                3. Om noden �r aktiverad sparas detta v�rde som aktuellt
   *                   fel / aktuell avvikelse p� noden. Annars r�knas inget fel.
   * 
   *                - reference: Referensv�rden fr�n tr�ningsdatan.                
   ********************************************************************************/
   void backpropagate(const std::vector<double>& reference)
   {
      for (std::size_t i = 0; i < this->num_nodes() && i < reference.size(); ++i)
      {
         const auto dev = reference[i] - this->output[i];

         if (this->output[i] > 0)
         {
            this->error[i] = dev;
         }
         else
         {
            this->error[i] = 0;
         }
      }
      return;
   }

   /********************************************************************************
   * backpropagate: Ber�knar aktuella fel i det dolda lagret genom att summera
   *                felen i n�sta lager multiplicerat med vikterna mellan noderna. 
   *                OBS! Denna funktion �r enbart avsedd f�r dolda lager.
   * 
   *                1. Iterara genom lagret fr�n f�rsta nod till sista nod.
   *                2. F�r varje iteration genom lagret, iterera fr�n f�rsta
   *                   till sista nod i n�sta lager och summera felet f�r denna
   *                   nod samt vikten mellan noderna.
   *                3. Om noden �r aktiverad (utsignalen �r �ver 0) s� sparas det 
   *                   ber�knade v�rdet. Annars om noden inte �r aktiverad s� 
   *                   sparas inte v�rdet (en inaktiverad nod har inte bidragit
   *                   till aktuell utsignal och har d�rmed inte orsakat fel).
   * 
   *                - next_layer: Referens till n�sta lager.
   ********************************************************************************/
   void backpropagate(const dense_layer& next_layer)
   {
      for (std::size_t i = 0; i < this->num_nodes(); ++i)
      {
         auto dev = 0.0;

         for (std::size_t j = 0; j < next_layer.num_nodes(); ++j)
         {
            dev += next_layer.error[j] * next_layer.weights[j][i];
         }

         if (this->output[i] > 0)
         {
            this->error[i] = dev; 
         }
         else
         {
            this->error[i] = 0;
         }
      }

      return;
   }

   /********************************************************************************
   * optimize: Justerar bias och vikter i aktuellt dense-lager utefter ber�knade
   *           fel samt angiven l�rhastighet.
   * 
   *           1.  Iterera genom lagret fr�n f�rsta till sista nod.
   *           2.  Justera aktuellt nods bias genom att addera aktuellt fel
   *               multiplicerat med l�rhastigheten.
   *           3.  Iterera genom nodens vikter fr�n f�rsta till sista och addera
   *               aktuellt fel multiplicerat med l�rhastigheten g�nger insignalen.
   * 
   *           - input:         Referens till vektor inneh�llande lagrets insignaler
   *                            (kan vara utsignaler fr�n f�reg�ende lager eller
   *                            fr�n ing�ngslagret).
   *           - learning_rate: L�rhastigheten, avg�r justeringsgraden vid fel.
   ********************************************************************************/
   void optimize(const std::vector<double>& input,
                 const double learning_rate)
   {
      for (std::size_t i = 0; i < this->num_nodes(); ++i)
      {
         this->bias[i] += this->error[i] * learning_rate;

         for (std::size_t j = 0; j < this->num_weights() && j < input.size(); ++j)
         {
            this->weights[i][j] += this->error[i] * learning_rate * input[j];
         }
      }
      return;
   }

   /********************************************************************************
   * print: Skriver ut information om aktuellt dense-lager via angiven utstr�m,
   *        d�r standardutenheten std::cout anv�nds som default f�r utskrift
   *        i terminalen.
   * 
   *        - ostream: Referens till angiven utstr�m (default = std::cout).
   ********************************************************************************/
   void print(std::ostream& ostream = std::cout)
   {
      ostream << "--------------------------------------------------------------------------------\n";

      ostream << "Number of nodes: " << this->num_nodes() << "\n";
      ostream << "Number of weights per node: " << this->num_weights() << "\n\n";

      ostream << "Output: ";
      this->print_line(this->output, ostream);

      ostream << "Error: ";
      this->print_line(this->error, ostream);

      ostream << "Bias: ";
      this->print_line(this->bias, ostream);

      ostream << "\nWeights:\n";

      for (std::size_t i = 0; i < this->num_nodes(); ++i)
      {
         ostream << "\tNode " << i + 1 << ": ";
         this->print_line(this->weights[i], ostream);
      }
      
      ostream << "--------------------------------------------------------------------------------\n\n";
      return;
   }

   /********************************************************************************
   * print_line: Skriver ut flyttal ur en vektor p� en enda rad och avslutas med
   *             en ny rad. Varje flyttal skrivs ut med tv� decimalers precision.
   * 
   *             - data   : Referens till vektorn vars inneh�ll ska skrivas ut.
   *             - ostream: Referens till den utstr�m som utskrift ska ske till.
   ********************************************************************************/
   void print_line(const std::vector<double>& data, 
                   std::ostream& ostream)
   {
      for (auto& i : data)
      {
         ostream << std::setprecision(2) << i << " ";
      }

      ostream << "\n";
      return;
   }
};

#endif /* DENSE_LAYER_HPP_ */